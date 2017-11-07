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
import dimer_tests as check
from dimer_plotting import plot_dynamics, plot_eig_dynamics, plot_coherences
#import ctypes


def coth(x):
    return float(sympy.coth(x).evalf())


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
        #print a, x
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
        print "greater than"
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega,alpha, wc)
        if imag_part:
            G += (1j/2.)*(integral_converge(F_m, 0,omega))
            G -= (1j/2.)*(integral_converge(F_p, 0,-omega))

        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        print "equal"
        G = (np.pi/2)*(2*alpha/beta)
        # The limit as omega tends to zero is zero for superohmic case?
        if imag_part:
            G += -(1j)*integral_converge(F_0, -1e-12,0)
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        print "less than"
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega),alpha, wc)
        if imag_part:
            G += (1j/2.)*integral_converge(F_m, 0,-abs(omega))
            G -= (1j/2.)*integral_converge(F_p, 0,abs(omega))
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G



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

    energies, states = check.exciton_states(PARAMS)
    psi_m = states[0]
    psi_p = states[1]
    eta = np.sqrt(eps**2 + 4*V**2)

    beta_1 = beta_f(PARAMS['T_1'])
    beta_2 = beta_f(PARAMS['T_2'])
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
    print site_1 + site_2
    # Initialise liouvilliian
    L =  qt.spre(site_1*Z_1) - qt.sprepost(Z_1, site_1)
    L += qt.spost(Z_1.dag()*site_1) - qt.sprepost(site_1, Z_1.dag())
    L +=  qt.spre(site_2*Z_2) - qt.sprepost(Z_2, site_2)
    L += qt.spost(Z_2.dag()*site_2) - qt.sprepost(site_2, Z_2.dag())

    L_xx = (alpha_1+alpha_2)*(qt.spre(XX_proj) + qt.spost(XX_proj)
                            -2*qt.sprepost(XX_proj,XX_proj))
    L+=L_xx
    return -L

def get_dynamics(w_2=100., bias=10., V=10., alpha_1=1., alpha_2=1., end_time=1):
    from qutip import basis
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

    J = J_minimal

    PARAM_names = ['w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2', 'wc',
                    'w0_1', 'w0_2', 'alpha_1', 'alpha_2', 'J', 'dipole_1','dipole_2']

    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)
    L = L_weak_phonon(PARAMS)
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
    fig = plt.figure(figsize=(12,8))
    plot_dynamics(DATA, timelist, expects, fig.add_subplot(211), title='', ss_dm = False)
    plot_coherences(DATA, timelist, expects, fig.add_subplot(212), title='', ss_dm = False)
    plt.savefig("weak_coupling_text.pdf")
    return DATA

if __name__ == "__main__":

    # w_2, bias, V, T_EM, alpha_EM, alpha_1, alpha_2, end_time
    DATA = get_dynamics(w_2=100., bias=10., V=20., alpha_1=1., alpha_2=1., end_time=1)
