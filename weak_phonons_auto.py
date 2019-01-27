import numpy as np
import scipy as sp
from scipy import integrate
import qutip as qt
from qutip import destroy, tensor, qeye, spost, spre, sprepost, basis
import time
from utils import J_minimal, beta_f, J_minimal_hard, J_underdamped, J_multipolar, exciton_states, rate_down, rate_up, lin_construct
import sympy
from numpy import pi, sqrt



def coth(x):
    return float(sympy.coth(x))

def cauchyIntegrands(omega, beta, J, Gamma, w0, ver, alpha=0.):
    # J_overdamped(omega, alpha, wc)
    # Function which will be called within another function where J, beta and
    # the eta are defined locally
    F = 0
    if ver == 1:
        F = J(omega, alpha, w0, Gamma=Gamma)*(coth(beta*omega/2.)+1)
    elif ver == -1:
        F = J(omega, alpha, w0, Gamma=Gamma)*(coth(beta*omega/2.)-1)
    elif ver == 0:
        F = J(omega, alpha, w0, Gamma=Gamma)
    return F

def int_conv(f, a, inc, omega):
        x = inc
        I = 0.
        I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
        while abs(f(x))>1E-4:
            #print ince x, f(x), a, omega
            a+=inc
            x+=inc
            I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
            #time.sleep(0.1)
        #print "Integral converged to {} with step size of {}".format(I, inc)
        return I # Converged integral

def integral_converge(f, a, omega):
    for inc in [400., 200., 100., 50., 25., 10, 5., 1, 0.5, 0.1]:
        try:
            return int_conv(f, a, inc, omega)
        except:
            if inc == 0.1:
                raise ValueError("Integrals couldn't converge with omega={}".format(omega))
            else:
                pass
                
    

def DecayRate(omega, beta, J, Gamma, w0, imag_part=True, c=1, alpha=0.):
    G = 0
    # Here I define the functions which "dress" the integrands so they
    # have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, 1 , alpha=alpha)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, -1, alpha=alpha)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, 0,  alpha=alpha)))
    w = np.linspace(-300, 300, 1000)
    #import matplotlib.pyplot as plt
    #plt.plot(w, F_0(w))
    #plt.show()
    w='cauchy'
    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega, alpha, w0, Gamma=Gamma)
        if imag_part:
            G += (1j/2.)*(integral_converge(F_m, 0,omega))
            G -= (1j/2.)*(integral_converge(F_p, 0,-omega))

        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        if J == J_underdamped:
            G = (pi*alpha*Gamma)/(beta*(w0**2))
        elif J == J_multipolar:
            G=0.
        else:
            print "Assuming J_minimal"
            G = (np.pi/2)*(2*Gamma/beta)
            # G = Gamma/(2*beta*w0)
        # The limit as omega tends to zero is zero for superohmic case?
        if imag_part:
            G += -(1j)*integral_converge(F_0, -1e-5,0.)
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),alpha, w0, Gamma=Gamma)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0,-abs(omega))
            G -= (1j/2.)*integral_converge(F_p, 0,abs(omega))
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G


def L_wc_auto(H_vib, A, w_0, Gamma, T, J, principal=False, 
                                silent=False, alpha=0.):
    import optical as opt
    reload(opt)
    ti = time.time()
    eVals, eVecs = H_vib.eigenstates()
    d_dim = len(eVals)
    beta = beta_f(T)
    G = 0
    for i in xrange(d_dim):
        for j in xrange(d_dim):
            eta = eVals[i]-eVals[j]
            aij = A.matrix_element(eVecs[i].dag(), eVecs[j])
            g = DecayRate(eta, beta, J, Gamma, w_0, imag_part=True, alpha=alpha)
            if (abs(g)>0) and (abs(aij)>0):
                G+=g*aij*eVecs[i]*(eVecs[j].dag())
                
    G_dag = G.dag()
    # Initialise liouvilliian
    L =  qt.spre(A*G) - qt.sprepost(G, A)
    L += qt.spost(G_dag*A) - qt.sprepost(A, G_dag)
    if not silent:
        print "Full optical Liouvillian took {} seconds.".format(time.time()- ti)
    return L

def get_wc_H_and_L(PARAMS,silent=False, threshold=0., H = None):
    import optical as opt
    reload(opt)
    w_1 = PARAMS['w_1']
    w_2 = PARAMS['w_2']
    OO, XO, OX = basis(3,0), basis(3,1), basis(3,2)
    sigma_m1 =  OO*XO.dag()
    sigma_m2 =  OO*OX.dag()
    eps = PARAMS['bias']
    V = PARAMS['V']
    if H == None:
        H = PARAMS['H_sub'] #w_1*XO*XO.dag() + w_2*OX*OX.dag() + V*(OX*XO.dag() + XO*OX.dag())
        H += 0.5*pi*PARAMS['alpha_1']*sigma_m1.dag()*sigma_m1
        H += 0.5*pi*PARAMS['alpha_2']*sigma_m2.dag()*sigma_m2
    ti = time.time()
    L = L_wc_auto(H, sigma_m1.dag()*sigma_m1, PARAMS['w0_1'], PARAMS['Gamma_1'], PARAMS['T_1'], J_underdamped, principal=True, 
                            silent=False, alpha=PARAMS['alpha_1'])
    L += L_wc_auto(H, sigma_m2.dag()*sigma_m2, PARAMS['w0_2'], PARAMS['Gamma_2'], PARAMS['T_2'], J_underdamped, principal=True, 
                            silent=False, alpha=PARAMS['alpha_2'])
    print "auto wc phonon liouvs took {} seconds".format(time.time() - ti)
    
    mu = PARAMS['mu']

    sigma = sigma_m1 + mu*sigma_m2
    if abs(PARAMS['alpha_EM'])>0:
        L +=  L_sec_wc_SES(PARAMS)
        #L += opt.L_BMME(H, sigma, PARAMS, ME_type='secular', site_basis=True, silent=silent)
    #H += 0.5*pi*(PARAMS['alpha_1']*sigma_m1.dag()*sigma_m1 + PARAMS['alpha_2']*sigma_m2.dag()*sigma_m2)
    return H, -L

def L_sec_wc_SES(args):
    ti = time.time()
    eps, V, w_xx = args['bias'], args['V'], args['w_xx']
    mu, gamma, w_1, J, T = args['mu'], args['alpha_EM'], args['w_1'], args['J'], args['T_EM']
    energies, states = exciton_states(args)
    dark, lm = states[0], energies[0]
    bright, lp = states[1], energies[1]
    OO = basis(3,0)
    eta = np.sqrt(4*V**2+eps**2)
    pre_1 = (sqrt(eta-eps)+mu*sqrt(eta+eps))/sqrt(2*eta) # A_wxx_lp
    pre_2 = -(sqrt(eta+eps)-mu*sqrt(eta-eps))/sqrt(2*eta) # A_wxx_lm
    pre_3 = (sqrt(eta+eps)+mu*sqrt(eta-eps))/sqrt(2*eta) # A_lp
    pre_4 = (sqrt(eta-eps)-mu*sqrt(eta+eps))/sqrt(2*eta) # A_lm
    #print pre_p, pre_p
    A_lp = pre_3*OO*bright.dag()
    A_lm= pre_4*OO*dark.dag()
    L = rate_up(lp, T, gamma, J, w_1)*lin_construct(A_lp.dag())
    L += rate_up(lm, T, gamma, J, w_1)*lin_construct(A_lm.dag())
    L += rate_down(lp, T, gamma, J, w_1)*lin_construct(A_lp)
    L += rate_down(lm, T, gamma, J, w_1)*lin_construct(A_lm)
    #print [(i, cf) for i, cf in enumerate(coeffs)]

    print "It took {} seconds to build the phenomenological Liouvillian".format(time.time()-ti)
    return L