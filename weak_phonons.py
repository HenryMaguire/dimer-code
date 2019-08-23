import numpy as np
import scipy as sp
from scipy import integrate
import qutip as qt
from qutip import destroy, tensor, qeye, spost, spre, sprepost, basis
import time
from utils import (J_minimal, beta_f, J_minimal_hard, J_multipolar, lin_construct, 
                    exciton_states, rate_up, rate_down, Occupation)
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
        F = J(omega, Gamma, w0, alpha=alpha)*(coth(beta*omega/2.)+1)
    elif ver == -1:
        F = J(omega, Gamma, w0, alpha=alpha)*(coth(beta*omega/2.)-1)
    elif ver == 0:
        F = J(omega, Gamma, w0, alpha=alpha)
    return F

def int_conv(f, a, inc, omega, tol=1E-4):
        x = inc
        I = 0.
        while abs(f(x))>tol:
            #print inc, x, f(x), a, omega
            I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
            a+=inc
            x+=inc
            #time.sleep(0.1)
        #print "Integral converged to {} with step size of {}".format(I, inc)
        return I # Converged integral

def integral_converge(f, a, omega, tol=2e-3):
    for inc in [300., 200., 100., 50., 25., 10, 5., 1, 0.5, 0.3, 0.2, 0.1]:
        inc += np.random.random()/10
        try:
            return int_conv(f, a, inc, omega, tol=tol)
        except:
            if inc < 0.1:
                raise ValueError("Integrals couldn't converge")
            else:
                pass

def DecayRate(omega, beta, J, Gamma, w0, imag_part=True, alpha=0., tol=1e-4):
    G = 0
    # Here I define the functions which "dress" the integrands so they
    # have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, 1 , alpha=alpha)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, -1, alpha=alpha)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, 0,  alpha=alpha)))
    w='cauchy'
    if beta> 0.01:
        tol=1e-6
    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega, Gamma, w0, alpha=alpha)
        if imag_part:
            G += (1j/2.)*(integral_converge(F_m, 0,omega, tol=tol))
            G -= (1j/2.)*(integral_converge(F_p, 0,-omega, tol=tol))

        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        G = (pi*alpha*Gamma)/(beta*(w0**2))
        if imag_part:
            G += -(1j)*integral_converge(F_0, -1e-12,0., tol=tol)
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),Gamma, w0, alpha=alpha)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0,-abs(omega), tol=tol)
            G -= (1j/2.)*integral_converge(F_p, 0,abs(omega), tol=tol)
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G

def L_weak_phonon_auto(H_vib, A, w_0, Gamma, T_EM, J, principal=False, 
                                silent=False, alpha=0.):
    
    ti = time.time()
    beta = beta_f(T_EM)
    eVals, eVecs = H_vib.eigenstates()
    #J=J_minimal # J_minimal(omega, Gamma, omega_0)
    d_dim = len(eVals)
    G = 0
    for i in range(d_dim):
        for j in range(d_dim):
            eta = eVals[i]-eVals[j]
            s = eVecs[i]*(eVecs[j].dag())
            
            #print A.matrix_element(eVecs[i].dag(), eVecs[j])
            overlap = A.matrix_element(eVecs[i].dag(), eVecs[j])
            if abs(overlap)>0:
                dr = DecayRate(eta, beta, J, Gamma, w_0, imag_part=principal, alpha=alpha)
                s*= overlap*dr
                G+=s
    G_dag = G.dag()
    # Initialise liouvilliian
    L =  qt.spre(A*G) - qt.sprepost(G, A)
    L += qt.spost(G_dag*A) - qt.sprepost(A, G_dag)
    if not silent:
        print("Calculating non-RWA Liouvilliian took {} seconds.".format(time.time()-ti))
    return -L


def _J_underdamped(omega, Gamma, omega_0, alpha=0.):
    return alpha*Gamma*(omega_0**2)*omega/(((omega_0**2)-(omega**2))**2+(Gamma*omega)**2)

def weak_phonon(H_sub, PARAMS, secular=False, shift=True, tol=1e-6):
    c_ops = PARAMS['coupling_ops']
    L = 0
    J = _J_underdamped
    """for i in range(2):
        #print c_ops[i], PARAMS['w0_'+str(i+1)], PARAMS['Gamma_'+str(i+1)], PARAMS['alpha_'+str(i+1)]
        if secular:
            l =L_full_secular(H_sub, c_ops[i], PARAMS['w0_'+str(i+1)], 
                        PARAMS['Gamma_'+str(i+1)], PARAMS['T_'+str(i+1)], J,
                        silent=True, alpha=PARAMS['alpha_'+str(i+1)])
            print( "Secular phonon dissipator: ", l)
            L += l
        else:
            L_ = L_weak_phonon_auto(H_sub, c_ops[i], PARAMS['w0_'+str(i+1)], 
                        PARAMS['Gamma_'+str(i+1)], PARAMS['T_'+str(i+1)], J,
                        principal=True, silent=True, alpha=PARAMS['alpha_'+str(i+1)]) # need principal value parts
            L+= L_"""
    L = L_wc_analytic(PARAMS, shift=shift, tol=tol)
    return L

def L_sec_wc_SES(args, silent=True):
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
    if not silent:
        print("It took {} seconds to build the phenomenological Liouvillian".format(time.time()-ti))
    return L



def L_full_secular(H_vib, A, w_0, Gamma, T, J, time_units='cm', silent=False, alpha=0.):
    '''
    Does not assume that the vibronic eigenstructure has no
    degeneracy. Must be of the form
    '''
    alpha *= 2
    ti = time.time()
    d = H_vib.shape[0]
    L = 0
    eVals, eVecs = H_vib.eigenstates()
    A_dag = A.dag()
    terms = 0
    for l in range(int(d)):
        for m in range(int(d)):
            for p in range(int(d)):
                for q in range(int(d)):
                    secular_freq = (eVals[l]-eVals[m]) - (eVals[p]-eVals[q])
                    if abs(secular_freq) <1E-10:
                        terms+=1
                        A_lm = A.matrix_element(eVecs[l].dag(), eVecs[m])
                        A_lm_star = A_dag.matrix_element(eVecs[m].dag(), eVecs[l])
                        A_pq = A.matrix_element(eVecs[p].dag(), eVecs[q])
                        A_pq_star = A_dag.matrix_element(eVecs[q].dag(), eVecs[p])
                        coeff_1 = A_lm*A_pq_star
                        coeff_2 = A_lm_star*A_pq
                        eps_pq = abs(eVals[p]-eVals[q])
                        Occ = Occupation(eps_pq, T, time_units)
                        # omega, Gamma, omega_0, alpha=0.)
                        r_up = np.pi*J(eps_pq, Gamma, w_0, alpha=alpha)*Occ
                        r_down = np.pi*J(eps_pq, Gamma, w_0, alpha=alpha)*(Occ+1)
                        LM = eVecs[l]*eVecs[m].dag()
                        ML = LM.dag()
                        PQ = eVecs[p]*eVecs[q].dag()
                        QP = PQ.dag()
                        """
                        if abs(secular_freq) !=0:
                            print (abs(secular_freq), r_up, A_lm, A_lm_star,
                                   A_pq, A_pq_star, r_down, l,m,p,q, m==q, l==p)
                        """
                        if abs(r_up*coeff_1)>0:
                            L+= r_up*coeff_1*(spre(LM*QP)-sprepost(QP,LM))
                        if abs(r_up*coeff_2)>0:
                            L+= r_up*coeff_2*(spost(PQ*ML)- sprepost(ML,PQ))
                        if abs(r_down*coeff_1)>0:
                            L+= r_down*coeff_1*(spre(ML*PQ)-sprepost(PQ, ML))
                        if abs(r_down*coeff_2)>0:
                            L+= r_down*coeff_2*(spost(QP*LM)-sprepost(LM, QP))
    if not silent:
        print ("It took ", time.time()-ti, " seconds to build the secular Liouvillian")
        print ("Secular approximation kept {:0.2f}% of total ME terms. \n".format(100*float(terms)/(d*d*d*d)))
    return -L


def L_wc_analytic(PARAMS, shift=True, tol=1e-7):
    energies, states = exciton_states(PARAMS, shift=shift)
    dark_proj = states[0]*states[0].dag()
    bright_proj = states[1]*states[1].dag()
    ct_p = states[1]*states[0].dag()
    ct_m = states[0]*states[1].dag()
    cross_term = (ct_p + ct_m)
    epsilon = PARAMS['bias']
    V = PARAMS['V']
    if shift:

        epsilon = PARAMS['shifted_bias']
    
    eta = sqrt(epsilon**2 + 4*V**2)

    # Bath 1
    G = (lambda x: (DecayRate(x, beta_f(PARAMS['T_1']), _J_underdamped, 
                        PARAMS['Gamma_1'], PARAMS['w0_1'], imag_part=True, 
                        alpha=PARAMS['alpha_1'], tol=tol)))
    G_0 = G(0.)
    G_p = G(eta)
    G_m = G(-eta)

    site_1 = (0.5/eta)*((eta+epsilon)*bright_proj + (eta-epsilon)*dark_proj + 2*V*cross_term)

    Z_1 = (0.5/eta)*(G_0*((eta+epsilon)*bright_proj + (eta-epsilon)*dark_proj) + 2*V*(ct_p*G_p + ct_m*G_m))
    # Bath 2
    G = (lambda x: (DecayRate(x, beta_f(PARAMS['T_2']), _J_underdamped, 
                        PARAMS['Gamma_2'], PARAMS['w0_2'], imag_part=True, 
                        alpha=PARAMS['alpha_2'], tol=tol)))
    site_2 = (0.5/eta)*((eta-epsilon)*bright_proj + (eta+epsilon)*dark_proj - 2*V*cross_term)
    G_0 = G(0.)
    G_p = G(eta)
    G_m = G(-eta)

    Z_2 = (0.5/eta)*(G_0*((eta-epsilon)*bright_proj + (eta+epsilon)*dark_proj) - 2*V*(ct_p*G_p + ct_m*G_m))

    L =  - qt.spre(site_1*Z_1) + qt.sprepost(Z_1, site_1)
    L += -qt.spost(Z_1.dag()*site_1) + qt.sprepost(site_1, Z_1.dag())

    L +=  - qt.spre(site_2*Z_2) + qt.sprepost(Z_2, site_2)
    L += -qt.spost(Z_2.dag()*site_2) + qt.sprepost(site_2, Z_2.dag())

    return L