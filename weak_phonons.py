# -*- coding: utf-8 -*-
"""
Weak-coupling spin-boson model solution
written in Python 2.7
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from utils import *
import sympy
from qutip import basis
import time
#import ctypes



def cauchyIntegrands(omega, beta, J, alpha, Gamma, omega_0, ver):
    # J_overdamped(omega, alpha, wc)
    # Function which will be called within another function where J, beta and
    # the eta are defined locally
    F = 0
    if ver == 1:
        F = J(omega, alpha, Gamma, omega_0)*(coth(beta*omega/2.)+1)
    elif ver == -1:
        F = J(omega, alpha, Gamma, omega_0)*(coth(beta*omega/2.)-1)
    elif ver == 0:
        F = J(omega, alpha, Gamma, omega_0)
    return F

def integral_converge(f, a, omega):
    x = 30
    I = 0
    while abs(f(x))>0.001:
        #print a, x
        I += integrate.quad(f, a, x, weight='cauchy', wvar=omega)[0]
        a+=30
        x+=30
    return I # Converged integral

def Decay(omega, beta, J, alpha, Gamma, omega_0, imag_part=True):
    G = 0
    # Here I define the functions which "dress" the integrands so they
    # have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, alpha, Gamma, omega_0, 1)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, alpha, Gamma, omega_0, -1)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, alpha, Gamma, omega_0, 0)))
    w='cauchy'
    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega, alpha, Gamma, omega_0)
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
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),alpha, Gamma, omega_0)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0,-abs(omega))
            G -= (1j/2.)*integral_converge(F_p, 0,abs(omega))
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G

def commutate(A, A_i, anti = False):
    if anti:
        return qt.spre(A*A_i) - qt.sprepost(A_i,A) + qt.sprepost(A, A_i.dag()) - qt.spre(A_i.dag()*A)
    else:
        return qt.spre(A*A_i) - qt.sprepost(A_i,A) - qt.sprepost(A, A_i.dag()) + qt.spre(A_i.dag()*A)


def auto_L(PARAMS, A, T, alpha):
    eig = zip(*check.exciton_states(PARAMS))
    L = 0
    beta = beta_f(T)
    for eig_i in eig:
        for eig_j in eig:
            omega = eig_i[0]-eig_j[0]
            A_ij = eig_i[1]*eig_j[1].dag()*A.matrix_element(eig_i[1].dag(), eig_j[1])
            L += Gamma(omega, beta, J_underdamped, alpha, PARAMS['wc'], imag_part=False) * commutate(A, A_ij)
            # Imaginary part
            G = Gamma(omega, beta, J_underdamped, alpha, PARAMS['wc'], imag_part=True)
            print G
            L += G.imag * commutate(A, A_ij, anti=True)
    return -0.5*L
"""

def L_weak_phonon(PARAMS):
    w_1 = PARAMS['w_1']
    w_2 = PARAMS['w_2']
    w_xx = PARAMS['w_xx']
    OO = basis(4,0)

    eps = PARAMS['bias']
    V = PARAMS['V']
    alpha_1 = PARAMS['alpha_1']
    alpha_2 = PARAMS['alpha_2']
    wc = PARAMS['wc']

    energies, states = exciton_states(PARAMS)
    psi_m = states[0]
    psi_p = states[1]
    eta = np.sqrt(eps**2 + 4*V**2)

    PARAMS['beta_1'] = beta_1 = beta_f(PARAMS['T_1'])
    PARAMS['beta_2'] = beta_2 = beta_f(PARAMS['T_2'])
    MM = psi_m*psi_m.dag()
    PP =psi_p*psi_p.dag()
    MP = psi_m*psi_p.dag()
    PM = psi_p*psi_m.dag()
    XX_proj = basis(4,3)*basis(4,3).dag()
    J=J_overdamped
    
    site_1 = (0.5*((eta-eps)*MM + (eta+eps)*PP) +V*(PM + MP))/eta
    Z_1 = (Gamma(0, beta_1, J, alpha_1, wc)*((eta-eps)*MM + (eta+eps)*PP))/(2.*eta)
    Z_1 += (V/eta)*Gamma(eta, beta_1, J, alpha_1, wc)*PM
    Z_1 += (V/eta)*Gamma(-eta, beta_1, J, alpha_1, wc)*MP

    site_2 = (0.5*((eta+eps)*MM + (eta-eps)*PP) -V*(PM + MP))/eta
    Z_2 = (Gamma(0, beta_2, J, alpha_2, wc)*((eta+eps)*MM + (eta-eps)*PP))/(2.*eta)
    Z_2 -= (V/eta)*Gamma(eta, beta_2, J, alpha_2, wc)*PM
    Z_2 -= (V/eta)*Gamma(-eta, beta_2, J, alpha_2, wc)*MP
    print site_1, site_2
    # Initialise liouvilliian
    L =  qt.spre(site_1*Z_1) - qt.sprepost(Z_1, site_1)
    L += qt.spost(Z_1.dag()*site_1) - qt.sprepost(site_1, Z_1.dag())
    L +=  qt.spre(site_2*Z_2) - qt.sprepost(Z_2, site_2)
    L += qt.spost(Z_2.dag()*site_2) - qt.sprepost(site_2, Z_2.dag())

    L_xx = (alpha_1+alpha_2)*(qt.spre(XX_proj) + qt.spost(XX_proj)
                            -2*qt.sprepost(XX_proj,XX_proj))
    L+=L_xx
    # Second attempt
    return L
"""

def exciton_states(PARS):
    w_1, w_2, V, bias = PARS['w_1'], PARS['w_2'],PARS['V'], PARS['bias']
    v_p, v_m = 0, 0
    eta = np.sqrt(4*(V**2)+bias**2)
    lam_p = w_2+(bias+eta)*0.5
    lam_m = w_2+(bias-eta)*0.5
    v_m = np.array([0., -(w_1-lam_p)/V, -1])
    #v_p/= /(1+(V/(w_2-lam_m))**2)
    v_m/= np.sqrt(np.dot(v_m, v_m))
    v_p = np.array([0, V/(w_2-lam_m),1.])

    v_p /= np.sqrt(np.dot(v_p, v_p))
    #print  np.dot(v_p, v_m) < 1E-15
    return [lam_m, lam_p], [qt.Qobj(v_m), qt.Qobj(v_p)]

def L_weak_phonon_SES(PARAMS, silent=False):
    ti = time.time()
    w_1 = PARAMS['w_1']
    w_2 = PARAMS['w_2']
    OO = basis(3,0)

    eps = PARAMS['bias']
    V = PARAMS['V']

    energies, states = exciton_states(PARAMS)
    psi_m = states[0]
    psi_p = states[1]
    eta = np.sqrt(eps**2 + 4*V**2)

    PARAMS['beta_1'] = beta_1 = beta_f(PARAMS['T_1'])
    PARAMS['beta_2'] = beta_2 = beta_f(PARAMS['T_2'])
    MM = psi_m*psi_m.dag()
    PP = psi_p*psi_p.dag()
    MP = psi_m*psi_p.dag()
    PM = psi_p*psi_m.dag()
    J = J_underdamped
    site_1 = (0.5*((eta-eps)*MM + (eta+eps)*PP) +V*(PM + MP))/eta
    Z_1 = (Decay(0, beta_1, J, PARAMS['alpha_1'], PARAMS['Gamma_1'], PARAMS['w0_1'])*((eta-eps)*MM + (eta+eps)*PP))/(2.*eta)
    Z_1 += (V/eta)*Decay(eta, beta_1, J, PARAMS['alpha_1'], PARAMS['Gamma_1'], PARAMS['w0_1'])*PM
    Z_1 += (V/eta)*Decay(-eta, beta_1, J, PARAMS['alpha_1'], PARAMS['Gamma_1'], PARAMS['w0_1'])*MP
    site_2 = (0.5*((eta+eps)*MM + (eta-eps)*PP) -V*(PM + MP))/eta

    Z_2 = (Decay(0, beta_2, J, PARAMS['alpha_2'], PARAMS['Gamma_2'], PARAMS['w0_2'])*((eta+eps)*MM + (eta-eps)*PP))/(2.*eta)
    Z_2 -= (V/eta)*Decay(eta, beta_2, J, PARAMS['alpha_2'], PARAMS['Gamma_2'], PARAMS['w0_2'])*PM
    Z_2 -= (V/eta)*Decay(-eta, beta_2, J, PARAMS['alpha_2'], PARAMS['Gamma_2'], PARAMS['w0_2'])*MP
    # Initialise liouvilliian
    L =  qt.spre(site_1*Z_1) - qt.sprepost(Z_1, site_1)
    L += qt.spost(Z_1.dag()*site_1) - qt.sprepost(site_1, Z_1.dag())
    L +=  qt.spre(site_2*Z_2) - qt.sprepost(Z_2, site_2)
    L += qt.spost(Z_2.dag()*site_2) - qt.sprepost(site_2, Z_2.dag())
    # Second attempt
    #print site_1, site_2
    if not silent:
        print "Weak coupling Liouvillian took {:0.2f} seconds".format(time.time()-ti)
    return -L

def get_wc_H_and_L(PARAMS,silent=False, threshold=0.):
    import optical as opt
    w_1 = PARAMS['w_1']
    w_2 = PARAMS['w_2']
    OO, XO, OX = basis(3,0), basis(3,1), basis(3,2)
    sigma_m1 =  OO*XO.dag()
    sigma_m2 =  OO*OX.dag()
    eps = PARAMS['bias']
    V = PARAMS['V']
    H = w_1*XO*XO.dag() + w_2*OX*OX.dag() + V*(OX*XO.dag() + XO*OX.dag())
    L = L_weak_phonon_SES(PARAMS, silent=False)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    sigma = sigma_m1 + mu*sigma_m2
    
    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_BMME(H, sigma, PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
    
    return H, L

def get_dynamics(w_2=100., bias=10., V=10., alpha_1=1., alpha_2=1., end_time=1):

    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    OO_p = OO*OO.dag()
    XO_p = XO*XO.dag()
    OX_p = OX*OX.dag()
    XX_p = XX*XX.dag()
    site_coherence = OX*XO.dag()
    w_1 = w_2 + bias
    dipole_1, dipole_2 = 1., 1.
    mu = w_2*dipole_2/(w_1*dipole_1)

    T_1, T_2 = 300., 300. # Phonon bath temperature

    wc = 1*53.08 # Ind.-Boson frame phonon cutoff freq
    w0_2, w0_1 = 500., 500. # underdamped SD parameter omega_0
    w_xx = w_2 + w_1

    J = J_overdamped

    PARAM_names = ['w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2', 'wc',
                    'w0_1', 'w0_2', 'alpha_1', 'alpha_2', 'J', 'dipole_1','dipole_2']

    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)
    print PARAMS
    L = auto_L(PARAMS, XO_p, T_1, alpha_1)+auto_L(PARAMS, OX_p, T_2, alpha_2)#_weak_phonon(PARAMS)

    H_dim = w_1*XO_p + w_2*OX_p + w_xx*XX_p + V*(site_coherence + site_coherence.dag())
    energies, states = check.exciton_states(PARAMS)
    N_en, N_st = H_dim.eigenstates()
    bright = states[1]*states[1].dag()
    dark = states[0]*states[0].dag()
    exciton_coherence = states[0]*states[1].dag()
    ops = [OO_p, XO_p, OX_p, XX_p]
    # Expectation values and time increments needed to calculate the dynamics
    expects = ops + [dark, bright, exciton_coherence, site_coherence]
    opts = qt.Options(num_cpus=1, nsteps=6000)
    timelist = np.linspace(0,end_time,4000*end_time)
    DATA = qt.mesolve(H_dim, XO_p, timelist, [L], e_ops=expects, progress_bar=True, options=opts)
    '''
    J_1 = lambda x : J_overdamped(x, alpha_1, wc)
    J_2 = lambda x : J_overdamped(x, alpha_2, wc)
    J_12 = lambda x : J_overdamped(x, alpha_1+alpha_2, wc)
    DATA = qt.bloch_redfield.brmesolve(H_dim, XO_p, timelist, [site_1, site_2], expects, [J_1, J_2], options=opts)'''
    fig = plt.figure(figsize=(12,8))
    plot_eig_dynamics(DATA, timelist, expects, fig.add_subplot(211), title='', ss_dm = False)
    plot_coherences(DATA, timelist, expects, fig.add_subplot(212), title='', ss_dm = False)
    plt.savefig("weak_coupling_test.pdf")
    plt.show()
    print "figure saved at weak_coupling_test.pdf"
    return DATA

if __name__ == "__main__":

    # w_2, bias, V, T_EM, alpha_EM, alpha_1, alpha_2, end_time
    DATA = get_dynamics(w_2=12400., bias=10., V=100., alpha_1=5., alpha_2=5., end_time=10)
