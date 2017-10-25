# -*- coding: utf-8 -*-
"""
Weak-coupling spin-boson model solution
written in Python 2.7
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#import ctypes


def coth(x):
    return (np.exp(2*x)+1)/(np.exp(2*x)-1)

def cauchyIntegrands(omega, beta, J, alpha, wc, ver):
    # Function which will be called within another function where J, beta and the eta are defined locally
    F = 0
    if ver == 1:
        F = J(omega)*(coth(beta*omega/2.)+1)
    elif ver == -1:
        F = J(omega)*(coth(beta*omega/2.)-1)
    elif ver == 0:
        F = J(omega)
    return F

def Gamma(omega, beta, J, alpha, wc):
    G = 0
    # Here I define the functions which "dress" the integrands so they have only 1 free parameter for Quad.
    F_p = (lambda x: (cauchyIntegrands(x, beta, J, alpha, wc, 1)))
    F_m = (lambda x: (cauchyIntegrands(x, beta, J, alpha, wc, -1)))
    F_0 = (lambda x: (cauchyIntegrands(x, beta, J, alpha, wc, 0)))
    n = 21
    print "Cauchy int. convergence checks: ", F_0(4*n), F_m(4*n), F_p(4*n)
    w='cauchy'
    if omega>0.:
        # These bits do the Cauchy integrals too
        G = (np.pi/2)*(coth(beta*omega/2.)-1)*J(omega)

        G += (1j/2.)*(integrate.quad(F_m, 0, n, weight=w, wvar=omega)[0]
                    +integrate.quad(F_m, n, 2*n, weight=w, wvar=omega)[0]
                    +integrate.quad(F_m, 2*n, 3*n, weight=w, wvar=omega)[0]
                    +integrate.quad(F_m, 3*n, 4*n, weight=w, wvar=omega)[0]
                    - integrate.quad(F_p, 0, n, weight=w, wvar=-omega)[0]
                    -integrate.quad(F_p, n, 2*n, weight=w, wvar=-omega)[0]
                    -integrate.quad(F_p, 2*n, 3*n, weight=w, wvar=-omega)[0]
                    -integrate.quad(F_p, 3*n, 4*n, weight=w, wvar=-omega)[0])
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=omega), integrate.quad(F_p, 0, n, weight='cauchy', wvar=-omega)
    elif omega==0.:
        G = (np.pi/2)*(2*)
        # The limit as omega tends to zero is zero for superohmic case?
        G = -(1j)*(integrate.quad(F_0, -1e-12, n, weight=w, wvar=0)[0]
                    +integrate.quad(F_0, n, 2*n, weight=w, wvar=0)[0]
                    +integrate.quad(F_0, 2*n, 3*n, weight=w, wvar=0)[0]
                    +integrate.quad(F_0, 3*n, 4*n, weight=w, wvar=0)[0])
        #print (integrate.quad(F_0, -1e-12, 20, weight='cauchy', wvar=0)[0])
    elif omega<0.:
        G = (np.pi/2)*(coth(beta*abs(omega)/2.)+1)*J(abs(omega))
        G += (1j/2.)*(integrate.quad(F_m, 0, n, weight=w, wvar=-abs(omega))[0]
                    +integrate.quad(F_m, n, 2*n, weight=w, wvar=-abs(omega))[0]
                    +integrate.quad(F_m, 2*n, 3*n, weight=w, wvar=-abs(omega))[0]
                    +integrate.quad(F_m, 3*n, 4*n, weight=w, wvar=-abs(omega))[0]
                    - integrate.quad(F_p, 0, n, weight=w, wvar=abs(omega))[0]
                    -integrate.quad(F_p, n, 2*n, weight=w, wvar=abs(omega))[0]
                    -integrate.quad(F_p, 2*n, 3*n, weight=w, wvar=abs(omega))[0]
                    -integrate.quad(F_p, 3*n, 4*n, weight=w, wvar=abs(omega))[0])
        #print integrate.quad(F_m, 0, n, weight='cauchy', wvar=-abs(omega)), integrate.quad(F_p, 0, n, weight='cauchy', wvar=abs(omega))
    return G



def liouvillian(PARAMS):

    OO = basis(4,0)

    eps = PARAMS['bias']
    V = PARAMS['V']
    alpha_1 = PARAMS['alpha_1']
    alpha_2 = PARAMS['alpha_2']
    wc = PARAMS['wc']
    H_dim = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag() + V*(XO*OX.dag() + OX*XO.dag())
    energies, states = check.exciton_states(PARAMS)
    psi_m = states[0]
    psi_p = states[1]
    print energies[1]-energies[0], np.sqrt(eps**2 + V**2)
    eta = energies[1]-energies[0]
    L = 0 # Initialise liouvilliian
    Z = 0 # initialise operator Z
    beta_1 = beta_f(PARAMS['T_1'])
    beta_2 = beta_f(PARAMS['T_2'])
    MM = psi_m*psi_m.dag()
    PP =psi_p*psi_p.dag()
    MP = psi_m*psi_p.dag()
    PM = psi_p*psi_m.dag()
    XX_proj = XX*XX.dag()
    J=J_overdamped
    site_1 = 0.5*((eta-eps)*MM + (eta+eps)*PP)/eta +(V/eta)*(PM + MP)
    site_2 = 0.5*((eta+eps)*MM + (eta-eps)*PP)/eta -(V/eta)*(PM + MP)
    Z_1 = 0.5*Gamma(0, beta_1, J, alpha_1, wc)*((eta-eps)*MM + (eta+eps)*PP)/eta
    Z_1 += (V/eta)*(Gamma(eta, beta_1, J, alpha_1, wc)*PM+Gamma(-eta, beta_1, J, alpha_1, wc)*MP)

    Z_2 = 0.5*Gamma(0, beta_1, J, alpha_2, wc)*((eta+eps)*MM + (eta-eps)*PP)/eta
    Z_2 -= (V/eta)*(Gamma(eta, beta_1, J, alpha_2, wc)*PM+Gamma(-eta, beta_1, J, alpha_2, wc)*MP)

    L +=  qt.spre(site_1*Z_1) - qt.sprepost(Z_1, site_1)
    L += qt.spost(Z_1.dag()*site_1) - qt.sprepost(site_1, Z_1.dag())
    L +=  qt.spre(site_2*Z_2) - qt.sprepost(Z_2, site_2)
    L += qt.spost(Z_2.dag()*site_2) - qt.sprepost(site_2, Z_2.dag())

    L_xx = (alpha_1+alpha_2)*(qt.spre(XX_proj) + qt.spost(XX_proj)
                            -2*qt.sprepost(XX_proj,XX_proj))
    L+=L_xx
    return -L

"""
plt.figure()
omega= np.linspace(0,50, 1000)
plt.plot(omega,J_superohm(omega))
plt.title("Spectral density")
"""
"""
epsilon = 1.
delta = 2*np.pi #*10**(-12)
T = 10.

L = liouvillian(epsilon, delta, J_overdamped, T)

H = qt.Qobj([[-epsilon/2., delta/2.],[delta/2., epsilon/2.]])


rho = qt.fock_dm(2,1)
timelist = np.linspace(0.,16., 10000)
expect_list = [qt.fock_dm(2,0), qt.fock_dm(2,1)]
DATA = qt.mesolve(H, rho, timelist, [L], expect_list)
DATA_A = (open("Data/pop5d2pi.dat").read()).split('\n')
pop_A, time_A = [], []
for item in DATA_A:
    li = item.split('\t')
    time_A.append(float(li[0]))
    pop_A.append(float(li[1]))

plt.figure()
plt.plot(timelist, DATA.expect[1], label='H')
plt.plot(time_A, pop_A, label="Ahsan's data")
plt.legend()

"""
"""
fo = open("WCSB_0p025.dat", "w")
fo.write("% Parameters: \n alpha=0.025, omega_c=2.2, T=10, delta = pi, epsilon=1% \n")
for item in DATA.expect[1]:
    fo.write(item)
"""
#fo.write(DATA.expect[1])
#fo.write("\n")
#fo.write(timelist)

#DATA_J = qt.mesolve(H, rho, timelist, [L_Jake], expect_list)
#plt.plot(timelist, DATA.expect[0])



#plt.plot(timelist, DATA_J.expect[1], label='J')
#plt.ylim(0,1)
#print liuo(100, 10, J, T=10.).eigenenergies()
#plt.legend()
#plt.hold()




#plt.show()
