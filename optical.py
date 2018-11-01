"""
The four electromagnetic liouvillians I am studying for the vibronic dimer are:
- no secular approximation
- a secular approximation
- an approximation which says that the enlarged system eigenstates are the same as the
    uncoupled system eigenstates (found in electronic_lindblad)

"""
import time
import multiprocessing
from functools import partial
import numpy as np
from numpy import sqrt
from numpy import pi
from qutip import Qobj, basis, spost, spre, sprepost, steadystate, tensor
import qutip.parallel as par

#from dimer_weak_phonons import cauchyIntegrands, integral_converge, Gamma
from utils import *
import phonons as RC

reload(RC)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def coth(x):
    return float(sympy.coth(x))

def cauchyIntegrands(omega, beta, J, alpha, wc, ver):
    # J_overdamped(omega, alpha, wc)
    # Function which will be called within another function where J, beta and
    # the eta are defined locally
    F = 0
    if ver == 1:
        F = J(omega, alpha, wc)*(coth(beta*omega/2.)+1)
    elif ver == -1:
        F = J(omega, alpha, wc)*(coth(beta*omega/2.)-1)
    elif ver == 0:
        F = J(omega, alpha, wc)
    return F

def integral_converge(f, a, omega):
    x = 30
    I = 0
    while abs(f(x))>0.01:
        print a, x
        I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
        a+=30
        x+=30
    return I # Converged integral

def Gamma(omega, beta, J, alpha, wc, imag_part=True):
    G = 0
    # Here I define the functions which "dress" the integrands so they
    # have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, alpha, wc, 1)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, alpha, wc, -1)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, alpha, wc, 0)))
    w='cauchy'
    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega,alpha, wc)
        if imag_part:
            G += (1j/2.)*(integral_converge(F_m, 0,omega))
            G -= (1j/2.)*(integral_converge(F_p, 0,-omega))

        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        G = (np.pi/2)*(2*alpha/beta)
        # The limit as omega tends to zero is zero for superohmic case?
        if imag_part:
            G += -(1j)*integral_converge(F_0, -1e-12,0)
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),alpha, wc)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0,-abs(omega))
            G -= (1j/2.)*integral_converge(F_p, 0,abs(omega))
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G

def L_non_rwa(H_vib, sigma, PARAMS, silent=False, site_basis=True):
    ti = time.time()
    A = sigma + sigma.dag()
    w_1 = PARAMS['w_1']
    alpha = PARAMS['alpha_EM']

    beta = beta_f(PARAMS['T_EM'])

    eVals, eVecs = H_vib.eigenstates()
    J=J_minimal
    d_dim = len(eVals)
    G = 0
    for i in xrange(d_dim):
        for j in xrange(d_dim):
            eta = eVals[i]-eVals[j]
            aij = A.matrix_element(eVecs[i].dag(), eVecs[j])
            g = Gamma(eta, beta, J, alpha, w_1, imag_part=False)
            if (abs(g)>0) and (abs(aij)>0):
                G+=g*aij*eVecs[i]*(eVecs[j].dag())
    eVecs = np.transpose(np.array([v.dag().full()[0] for v in eVecs])) # get into columns of evecs
    eVecs_inv = sp.linalg.inv(eVecs) # has a very low overhead
    if site_basis:
        G = to_site_basis(G, eVals, eVecs, eVecs_inv)
    else:
        A = to_eigenbasis(A, eVals, eVecs, eVecs_inv)
        H_vib = to_eigenbasis(H_vib, eVals, eVecs, eVecs_inv)
    G_dag = G.dag()
    # Initialise liouvilliian
    L =  qt.spre(A*G) - qt.sprepost(G, A)
    L += qt.spost(G_dag*A) - qt.sprepost(A, G_dag)
    if not silent:
        print "Full optical Liouvillian took {} seconds.".format(time.time()- ti)
    return -L*0.5

def nonRWA_function(idx_list, **kwargs):

    A = kwargs['A']
    eVecs = kwargs['eVecs']
    eVals = kwargs['eVals']
    T = kwargs['T_EM']
    w_1, alpha, J = kwargs['w_1'], kwargs['alpha_EM'], kwargs['J']
    beta = beta_f(kwargs['T_EM'])

    op_contrib = 0*A
    for i, j in idx_list:
        eta = eVals[i]-eVals[j]
        g = Gamma(eta, beta, J, alpha, w_1, imag_part=False)
        aij = A.matrix_element(eVecs[i].dag(), eVecs[j])
        if (abs(g)>0) and (abs(aij)>0):
            op_contrib+= g*aij*eVecs[i]*(eVecs[j].dag())
    return op_contrib


def L_non_rwa_par(H_vib, sigma, args, silent=False, site_basis=True):
    Gamma, T, w_1, J, num_cpus = args['alpha_EM'], args['T_EM'], args['w_1'],args['J'], args['num_cpus']
    #Construct non-secular liouvillian
    ti = time.time()
    dim_ham = H_vib.shape[0]

    A = sigma + sigma.dag()
    eVals, eVecs = H_vib.eigenstates()
    kwargs = dict(args)
    kwargs.update({'eVals':eVals, 'eVecs':eVecs, 'A':A})
    l = dim_ham*range(dim_ham) # Perform two loops in one
    i_j_gen = [(i,j) for i,j in zip(sorted(l), l)]
    i_j_gen = chunks(i_j_gen, 1024)
    pool = multiprocessing.Pool(num_cpus)
    Out = pool.imap_unordered(partial(nonRWA_function,**kwargs), i_j_gen)
    pool.close()
    pool.join()
    
    G = Qobj(np.sum(np.array([x for x in Out])), dims=H_vib.dims)
    eVecs = np.transpose(np.array([v.dag().full()[0] for v in eVecs])) # get into columns of evecs
    eVecs_inv = sp.linalg.inv(eVecs) # has a very low overhead
    if site_basis:
        G = to_site_basis(G, eVals, eVecs, eVecs_inv)
    else:
        A = to_eigenbasis(A, eVals, eVecs, eVecs_inv)
        H_vib = to_eigenbasis(H_vib, eVals, eVecs, eVecs_inv)
    G_dag = G.dag()

    L =  qt.spre(A*G) - qt.sprepost(G, A)
    L += qt.spost(G_dag*A) - qt.sprepost(A, G_dag)
    
    if not silent:
        print "It took ", time.time()-ti, " seconds to build the Non-secular RWA Liouvillian"
    return -0.5*L

def nonsecular_function(args, **kwargs):
    i, j = args[0], args[1]
    A = kwargs['A']
    eVecs = kwargs['eVecs']
    eVals = kwargs['eVals']
    T = kwargs['T_EM']
    w_1, Gamma, J = kwargs['w_1'], kwargs['alpha_EM'], kwargs['J']
    eps_ij = abs(eVals[i]-eVals[j])

    A_ij = A.matrix_element(eVecs[i].dag(), eVecs[j])
    A_ji = (A.dag()).matrix_element(eVecs[j].dag(), eVecs[i])
    Occ = Occupation(eps_ij, T)
    zero = 0*A

    # 0.5*np.pi*alpha*(N+1)
    if abs(A_ij)>0 or abs(A_ji)>0:
        IJ = eVecs[i]*eVecs[j].dag()
        JI = eVecs[j]*eVecs[i].dag()

        if eps_ij == 0:
            JN = Gamma/(2*pi*w_1*beta_f(T))
            r_up = 2*pi*JN
            r_down = 2*pi*JN
        else:
            r_up = 2*pi*J(eps_ij, Gamma, w_1)*Occ
            r_down = 2*pi*J(eps_ij, Gamma, w_1)*(Occ+1)
        return Qobj(r_up*A_ji*JI), Qobj(r_down*A_ji*JI), Qobj(r_down*A_ij*IJ), Qobj(r_up*A_ij*IJ)
    else:
        return zero, zero, zero, zero



def L_nonsecular_par(H_vib, A, args, site_basis=True):
    Gamma, T, w_1, J, num_cpus = args['alpha_EM'], args['T_EM'], args['w_1'],args['J'], args['num_cpus']
    #Construct non-secular liouvillian
    ti = time.time()
    dim_ham = H_vib.shape[0]

    eVals, eVecs = sorted_eig(H_vib)
    kwargs = dict(args)
    kwargs.update({'eVals':eVals, 'eVecs':eVecs, 'A':A})
    l = dim_ham*range(dim_ham) # Perform two loops in one
    i_j_gen = ((i,j) for i,j in zip(sorted(l), l))
    pool = multiprocessing.Pool(num_cpus)
    Out = pool.imap_unordered(partial(nonsecular_function,**kwargs), i_j_gen)
    pool.close()
    pool.join()
    X_ops = np.sum(np.array([x for x in Out]), axis=0)
    
    eVecs_inv = sp.linalg.inv(eVecs) # has a very low overhead
    if site_basis:
        for j, op in enumerate(X_ops):
            X_ops[j] = to_site_basis(op, eVals, eVecs, eVecs_inv)
    else:
        A = to_eigenbasis(A, eVals, eVecs, eVecs_inv)
        H_vib = to_eigenbasis(H_vib, eVals, eVecs, eVecs_inv)
    X1, X2, X3, X4 = X_ops[0], X_ops[1], X_ops[2], X_ops[3]
    L = spre(A*X1) -sprepost(X1,A)+spost(X2*A)-sprepost(A,X2)
    L+= spre(A.dag()*X3)-sprepost(X3, A.dag())+spost(X4*A.dag())-sprepost(A.dag(), X4)
    #print np.sum(X1.full()), np.sum(X2.full()), np.sum(X3.full()), np.sum(X4.full())
    print "It took ", time.time()-ti, " seconds to build the Non-secular RWA Liouvillian"

    return -0.25*L

def L_nonsecular(H_vib, A, args, site_basis=True):
    Gamma, T, w_1, J = args['alpha_EM'], args['T_EM'], args['w_1'],args['J']
    #Construct non-secular liouvillian
    ti = time.time()
    dim_ham = H_vib.shape[0]
    eVals, eVecs = sorted_eig(H_vib)
    l = dim_ham*range(dim_ham) # Perform two loops in one
    X1, X2, X3, X4 = 0,0,0,0
    for i,j in zip(sorted(l), l):
        eps_ij = abs(eVals[i]-eVals[j])
        A_ij = A.matrix_element(eVecs[i].dag(), eVecs[j])
        A_ji = (A.dag()).matrix_element(eVecs[j].dag(), eVecs[i])

        # 0.5*np.pi*alpha*(N+1)
        if abs(A_ij)>0 or abs(A_ji)>0:
            Occ = Occupation(eps_ij, T)
            IJ = eVecs[i]*eVecs[j].dag()
            JI = eVecs[j]*eVecs[i].dag()
            r_up = 0
            r_down = 0
            if eps_ij == 0:
                JN = Gamma/(2*pi*w_1*beta_f(T))
                r_up = 2*pi*JN
                r_down = 2*pi*JN
            else:
                r_up = 2*pi*J(eps_ij, Gamma, w_1)*Occ
                r_down = 2*pi*J(eps_ij, Gamma, w_1)*(Occ+1)
            X3+= r_down*A_ij*IJ
            X4+= r_up*A_ij*IJ
            X1+= r_up*A_ji*JI
            X2+= r_down*A_ji*JI
    
    L = spre(A*X1) -sprepost(X1,A)+spost(X2*A)-sprepost(A,X2)
    L+= spre(A.dag()*X3)-sprepost(X3, A.dag())+spost(X4*A.dag())-sprepost(A.dag(), X4)
    #print np.sum(X1.full()), np.sum(X2.full()), np.sum(X3.full()), np.sum(X4.full())
    print "It took ", time.time()-ti, " seconds to build the Non-secular RWA Liouvillian"
    return -0.25*L

def secular_function(args, eVecs=[], eVals=[], T_EM=0.,
                            w_1=1., alpha_EM=0.,
                            J = None, col_op=None):
    i, j = args[0], args[1]
    L = 0
    lam_ij = col_op.matrix_element(eVecs[i].dag(), eVecs[j])
    lam_ji = col_op.dag().matrix_element(eVecs[j].dag(), eVecs[i])
    lam_ij_sq = lam_ij*lam_ji
    eps_ij = abs(eVals[i]-eVals[j])

    if abs(lam_ij_sq)>0:
        IJ = eVecs[i]*eVecs[j].dag()
        JI = eVecs[j]*eVecs[i].dag()
        JJ = eVecs[j]*eVecs[j].dag()
        II = eVecs[i]*eVecs[i].dag()
        Occ = Occupation(eps_ij, T)
        r_up = 0
        r_down = 0
        if eps_ij == 0:
            JN = Gamma/(2*pi*w_1*beta_f(T))
            r_up = 2*pi*JN
            r_down = 2*pi*JN
        else:
            r_up = 2*pi*J(eps_ij, alpha_EM, w_1)*Occ
            r_down = 2*pi*J(eps_ij, alpha_EM, w_1)*(Occ+1)

        s1 = r_up*(spre(II) + spost(II) - 2*sprepost(JI, IJ))
        s2 = r_down*(spost(JJ)+ spre(JJ) - 2*sprepost(IJ,JI))
        L = lam_ij_sq*(s1+s2)
    return Qobj(L)

def L_secular(H_vib, A, args, silent=False):
    '''
    Initially assuming that the vibronic eigenstructure has no
    degeneracy and the secular approximation has been made
    '''
    ti = time.time()
    #num_cpus = args['num_cpus']
    dim_ham = H_vib.shape[0]
    eVals, eVecs = sorted_eig(H_vib)
    #print [(i, ev) for i, ev in enumerate(eVals)] # Understanding manifold structure
    #names = ['eVals', 'eVecs', 'A', 'w_1', 'Gamma', 'T', 'J']
    T = args['T_EM']
    w_1, Gamma, J = args['w_1'], args['alpha_EM'], args['J']
    l = dim_ham*range(dim_ham)
    i_j_gen = ((i,j) for i,j in zip(sorted(l), l))
    L = 0
    for i, j in i_j_gen:
        lam_ij = A.matrix_element(eVecs[i].dag(), eVecs[j])
        lam_ji = A.dag().matrix_element(eVecs[j].dag(), eVecs[i])
        #lam_mn = (A.dag()).matrix_element(eVecs[n].dag(), eVecs[m])
        lam_ij_sq = lam_ij*lam_ji
        if abs(lam_ij_sq)>0:
            eps_ij = abs(eVals[i]-eVals[j])
            IJ = eVecs[i]*eVecs[j].dag()
            JI = eVecs[j]*eVecs[i].dag()
            JJ = eVecs[j]*eVecs[j].dag()
            II = eVecs[i]*eVecs[i].dag()
            Occ = Occupation(eps_ij, T)
            r_up = 0
            r_down = 0
            if eps_ij == 0:
                JN = Gamma/(2*pi*w_1*beta_f(T))
                r_up = 2*pi*JN
                r_down = 2*pi*JN
            else:
                r_up = 2*pi*J(eps_ij, Gamma, w_1)*Occ
                r_down = 2*pi*J(eps_ij, Gamma, w_1)*(Occ+1)
            L += Qobj(lam_ij_sq*(r_up*(spre(II) + spost(II) - 2*sprepost(JI, IJ))+r_down*(spost(JJ)+ spre(JJ) - 2*sprepost(IJ,JI))))
    if not silent:
        print "It took ", time.time()-ti, " seconds to build the secular RWA Liouvillian"
    return -np.sum(L)*0.25

def L_secular_par(H_vib, A, args):
    '''
    Initially assuming that the vibronic eigenstructure has no
    degeneracy and the secular approximation has been made
    '''
    ti = time.time()
    num_cpus = args['num_cpus']
    dim_ham = H_vib.shape[0]
    eVals, eVecs = H_vib.eigenstates()
    #print [(i, ev) for i, ev in enumerate(eVals)] # Understanding manifold structure
    #names = ['eVals', 'eVecs', 'A', 'w_1', 'Gamma', 'T', 'J']
    kwargs = dict(args)
    kwargs.update({'eVals':eVals, 'eVecs':eVecs, 'A':A})
    #for name in names:
    #    kwargs[name] = eval(name)
    l = dim_ham*range(dim_ham)
    i_j_gen = ((i,j) for i,j in zip(sorted(l), l))
    pool = multiprocessing.Pool(num_cpus)
    L = pool.imap_unordered(partial(secular_function,**kwargs), i_j_gen)
    pool.close()
    pool.join()
    print "It took ", time.time()-ti, " seconds to build the secular RWA Liouvillian"
    L = [l for l in L]
    return -np.sum(L)*0.25

def L_phenom(I, args):
    ti = time.time()
    eps, V, w_xx = args['bias'], args['V'], args['w_xx']
    mu, gamma, w_1, J, T = args['mu'], args['alpha_EM'], args['w_1'], args['J'], args['T_EM']
    H_sub = qt.Qobj([[0,0,0,0],[0,w_1,V,0],[0,V,w_1-eps,0],[0,0,0,(2*(w_1-eps))+eps]])
    energies, states = check.exciton_states(args)
    dark, lm = states[0], energies[0]
    bright, lp = states[1], energies[1]
    OO = basis(4,0)
    XX = basis(4,3)
    eta = np.sqrt(4*V**2+eps**2)
    pre_1 = (sqrt(eta-eps)+mu*sqrt(eta+eps))/sqrt(2*eta) # A_wxx_lp
    pre_2 = -(sqrt(eta+eps)-mu*sqrt(eta-eps))/sqrt(2*eta) # A_wxx_lm
    pre_3 = (sqrt(eta+eps)+mu*sqrt(eta-eps))/sqrt(2*eta) # A_lp
    pre_4 = (sqrt(eta-eps)-mu*sqrt(eta+eps))/sqrt(2*eta) # A_lm
    #print pre_p, pre_p
    A_lp, A_wxx_lp = pre_3*tensor(OO*bright.dag(), I),  pre_1*tensor(bright*XX.dag(),I)
    A_lm, A_wxx_lm = pre_4*tensor(OO*dark.dag(), I),  pre_2*tensor(dark*XX.dag(),I)
    L = rate_up(lp, T, gamma, J, w_1)*lin_construct(A_lp.dag())
    L += rate_up(lm, T, gamma, J, w_1)*lin_construct(A_lm.dag())
    L += rate_down(lp, T, gamma, J, w_1)*lin_construct(A_lp)
    L += rate_down(lm, T, gamma, J, w_1)*lin_construct(A_lm)
    L += rate_up(w_xx-lp, T, gamma, J, w_1)*lin_construct(A_wxx_lp.dag())
    L += rate_up(w_xx-lm, T, gamma, J, w_1)*lin_construct(A_wxx_lm.dag())
    L += rate_down(w_xx-lp, T, gamma, J, w_1)*lin_construct(A_wxx_lp)
    L += rate_down(w_xx-lm, T, gamma, J, w_1)*lin_construct(A_wxx_lm)
    #print [(i, cf) for i, cf in enumerate(coeffs)]

    print "It took {} seconds to build the phenomenological Liouvillian".format(time.time()-ti)
    return L

def L_phenom_old(I, args):
    ti = time.time()
    eps, V, w_xx = args['bias'], args['V'], args['w_xx']
    mu, gamma, w_1, J, T = args['mu'], args['alpha_EM'], args['w_1'], args['J'], args['T_EM']
    H_sub = qt.Qobj([[0,0,0,0],[0,w_1,V,0],[0,V,w_1-eps,0],[0,0,0,(2*(w_1-eps))+eps]])
    energies, states = check.exciton_states(args)
    dark, lm = states[0], energies[0]
    bright, lp = states[1], energies[1]
    OO = basis(4,0)
    XX = basis(4,3)
    eta = np.sqrt(4*V**2+eps**2)
    pre_p = (sqrt(eta-eps)+mu*sqrt(eta+eps))/sqrt(2*eta)
    pre_m = -(sqrt(eta+eps)-mu*sqrt(eta-eps))/sqrt(2*eta)
    pre_p = (sqrt(eta-eps)+mu*sqrt(eta+eps))/sqrt(2*eta)
    pre_m = -(sqrt(eta+eps)-mu*sqrt(eta-eps))/sqrt(2*eta)
    #print pre_p, pre_p
    A_lp, A_wxx_lp = pre_p*tensor(OO*bright.dag(), I),  pre_p*tensor(bright*XX.dag(),I)
    A_lm, A_wxx_lm = pre_m*tensor(OO*dark.dag(), I),  pre_m*tensor(dark*XX.dag(),I)
    L = rate_up(lp, T, gamma, J, w_1)*lin_construct(A_lp.dag())
    L += rate_up(lm, T, gamma, J, w_1)*lin_construct(A_lm.dag())
    L += rate_down(lp, T, gamma, J, w_1)*lin_construct(A_lp)
    L += rate_down(lm, T, gamma, J, w_1)*lin_construct(A_lm)
    L += rate_up(w_xx-lp, T, gamma, J, w_1)*lin_construct(A_wxx_lp.dag())
    L += rate_up(w_xx-lm, T, gamma, J, w_1)*lin_construct(A_wxx_lm.dag())
    L += rate_down(w_xx-lp, T, gamma, J, w_1)*lin_construct(A_wxx_lp)
    L += rate_down(w_xx-lm, T, gamma, J, w_1)*lin_construct(A_wxx_lm)
    coeffs = [pre_m*rate_up(lm, T, gamma, J, w_1), pre_m*rate_down(lm, T, gamma, J, w_1),
                    pre_p*rate_up(lp, T, gamma, J, w_1), pre_p*rate_down(lp, T, gamma, J, w_1),
                    pre_m*rate_up(w_xx-lm, T, gamma, J, w_1),pre_m*rate_down(w_xx-lm, T, gamma, J, w_1),
                    pre_p*rate_up(w_xx-lp, T, gamma, J, w_1), pre_p*rate_down(w_xx-lp, T, gamma, J, w_1)
                    ]
    #print [(i, cf) for i, cf in enumerate(coeffs)]

    print "It took {} seconds to build the phenomenological Liouvillian".format(time.time()-ti)
    return L


if __name__ == "__main__":
    '''
    ev_to_inv_cm = 8065.5
    w_1, w_2 = 1.4*ev_to_inv_cm, 1.*ev_to_inv_cm
    eps = (w_1 + w_2)/2 # Hack to make the spectral density work
    V = 200.
    w_xx = w_1+w_2+V
    T_1, T_2 = 300., 300.
    wRC_1, wRC_2 = 300., 300.
    alpha_1, alpha_2 = 10./pi, 10./pi
    wc =53.
    N_1, N_2= 4, 4
    exc = int((N_1+N_2)*0.75)
    mu=1
    T_EM = 6000.
    Gamma_EM = 6.582E-4*ev_to_inv_cm

    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()

    L_RC, H_vib, A_1, A_2, A_EM, wRC_1, wRC_2 = RC.RC_mapping_UD(
                                w_1, w_2, w_xx, V, T_1, T_2, wRC_1, wRC_2,
                                alpha_1, alpha_2, wc,  N_1, N_2, exc, mu=1, num_cpus=1)
    print "*******   L_RC  *******"
    #print "Is L_RC a completely positive map? -", L_RC.iscp
    #print "Is it trace-preserving? -", L_RC.istp
    # Non-secular version
    print "*******   L_NS  *******"
    L_ns = L_nonsecular(H_vib, A_EM, eps, Gamma_EM, T_EM, J_minimal)
    #print "Is L_ns a completely positive map? -", L_ns.iscp
    #print "Is it trace-preserving? -", L_ns.istp
    ss_ns = steadystate(H_vib, [L_RC + L_ns]).ptrace(0)
    print "Non-sec steady-state dimer DM is, "
    print ss_ns
    #print "trace = ",ss_ns.tr()
    # Secular version
    print "*******   L_S  *******"
    L_s = L_secular(H_vib, A_EM, eps, Gamma_EM, T_EM, J_minimal)
    print "dimer_driving_liouv is finished."
    #print "Is L_s a completely positive map? -", L_s.iscp
    #print "Is it trace-preserving? -", L_s.istp
    ss_s = steadystate(H_vib, [L_RC + L_s]).ptrace(0)
    print "Sec steady-state dimer DM is, "
    print ss_s
    #print ss_s.tr()

    # Naive version
    """
    L_n = electronic_lindblad(w_xx, w_1, w_1-w_2, V, mu, Gamma_EM,
                                T_EM, N_1, N_1,  N_1+N_2, J_minimal)
    print "Is L_n a completely positive map? -", L_n.iscp
    print "Is it trace-preserving? -", L_n.istp
    ss_n = steadystate(H_vib, [L_RC + L_n]).ptrace(0)
    print "Naive steady-state dimer DM is, "
    print ss_n
    print ss_n.tr()
    """
    real_therm = (((-1./(0.695*T_EM))*H_vib).expm().ptrace(0))/(((-1./(0.695*T_EM))*H_vib).expm().tr())
    # This is just a thermal state of the TLS-RC with respect to the electromagnetic bath only.
    print real_therm
    #print L_RC.dims == L_ns.dims, L_RC.dims == L_s.dims, L_ns.dims ==L_s.dims, L_n.dims == L_RC.dims
    '''
