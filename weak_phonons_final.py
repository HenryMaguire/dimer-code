import numpy as np
import scipy as sp
from scipy import integrate
import qutip as qt
from qutip import destroy, tensor, qeye, spost, spre, sprepost, basis
import time
from utils import J_minimal, beta_f, J_minimal_hard, J_multipolar, lin_construct, exciton_states, rate_up, rate_down
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

def int_conv(f, a, inc, omega):
        x = inc
        I = 0.
        while abs(f(x))>1E-3:
            #print inc, x, f(x), a, omega
            I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
            a+=inc
            x+=inc
            #time.sleep(0.1)
        #print "Integral converged to {} with step size of {}".format(I, inc)
        return I # Converged integral

def integral_converge(f, a, omega):
    for inc in [300., 200., 100., 50., 25., 10, 5., 1, 0.5]:
        inc += np.random.random()/10
        try:
            return int_conv(f, a, inc, omega) 
        except:
            if inc == 0.5:
                raise ValueError("Integrals couldn't converge")
            else:
                pass
                
    

def DecayRate(omega, beta, J, Gamma, w0, imag_part=True, c=1, alpha=0.):
    G = 0
    # Here I define the functions which "dress" the integrands so they
    # have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, 1 , alpha=alpha)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, -1, alpha=alpha)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, Gamma, w0, 0,  alpha=alpha)))
    w='cauchy'
    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega, Gamma, w0, alpha=alpha)
        if imag_part:
            G += (1j/2.)*(integral_converge(F_m, 0,omega))
            G -= (1j/2.)*(integral_converge(F_p, 0,-omega))

        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        if J == _J_underdamped:
            G = (pi*alpha*Gamma)/(beta*(w0**2))
        elif J == J_multipolar:
            G=0.
        else:
            print "Assuming J_minimal"
            G = (np.pi/2)*(2*Gamma/beta)
            # G = Gamma/(2*beta*w0)
        # The limit as omega tends to zero is zero for superohmic case?
        if imag_part:
            G += -(1j)*integral_converge(F_0, -1e-12,0.)
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),Gamma, w0, alpha=alpha)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0,-abs(omega))
            G -= (1j/2.)*integral_converge(F_p, 0,abs(omega))
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
    for i in xrange(d_dim):
        for j in xrange(d_dim):
            eta = eVals[i]-eVals[j]
            s = eVecs[i]*(eVecs[j].dag())
            
            #print A.matrix_element(eVecs[i].dag(), eVecs[j])
            overlap = A.matrix_element(eVecs[i].dag(), eVecs[j])
            if abs(overlap)>0:
                s*= overlap*DecayRate(eta, beta, J, Gamma, w_0, imag_part=principal, alpha=alpha)
            G+=s
    G_dag = G.dag()
    # Initialise liouvilliian
    L =  qt.spre(A*G) - qt.sprepost(G, A)
    L += qt.spost(G_dag*A) - qt.sprepost(A, G_dag)
    if not silent:
        print "Calculating non-RWA Liouvilliian took {} seconds.".format(time.time()-ti)
    return -L


def _J_underdamped(omega, Gamma, omega_0, alpha=0.):
    return alpha*Gamma*(omega_0**2)*omega/(((omega_0**2)-(omega**2))**2+(Gamma*omega)**2)

def weak_phonon(H_sub, PARAMS):
    c_ops = PARAMS['coupling_ops']
    L = 0
    for i in range(2):
        #print c_ops[i], PARAMS['w0_'+str(i+1)], PARAMS['Gamma_'+str(i+1)], PARAMS['alpha_'+str(i+1)]
        L+= L_weak_phonon_auto(H_sub, c_ops[i], PARAMS['w0_'+str(i+1)], 
                        PARAMS['Gamma_'+str(i+1)], PARAMS['T_'+str(i+1)], _J_underdamped, 
                        principal=True, silent=True, alpha=PARAMS['alpha_'+str(i+1)]) # need principal value parts
    return -L

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
