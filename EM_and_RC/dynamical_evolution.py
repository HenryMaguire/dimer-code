##### a function to construct and solve the RC master equation ####

# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg

from scipy import *


#import plotting packages
import matplotlib.pyplot as plt

import Optical_liouvillian as opli

def rate_up(w, beta, gamma):
	n = 1 / (exp(beta * w) - 1.)
	rate = 0.5 * pi * gamma * n
	return rate

def rate_down(w, beta, gamma):
	n = 1 / (exp(beta * w) - 1.)
	rate = 0.5 * pi * gamma * (n + 1. )
	return rate


# define the site basis is:
biexc = basis(4, 0)
site1 = basis(4, 1)
site2 = basis(4, 2)
gr = basis(4, 3)

# and in density matrix form
# rhogr = gr * (gr.dag())
# rhobi = bi * (bi.dag())
# rhopsi_p = b1 * b1.dag()
# rhopsi_m = b2 * b2.dag()

# the system parameters that can be defined globally are
w2 = 1500. # splitting for site 1
V = 100.
mu = 0.
gamma = 0.1
EM_temp = 500.0
k_b = 0.695
thermal_energy = k_b * EM_temp
beta = float(1. / thermal_energy)

eps = 100.0


# the eigenstate splitting is given by:
eta = sqrt(eps ** 2. + 4. * V ** 2.)


#w2 = w1 + eps
wXX =  2 * w2 + eps

#

#

#print(lam_p, lam_m)

HDim = (w2 + eps) * site1 * site1.dag() + (w2 ) * site2 * site2.dag() + wXX * biexc * biexc.dag()
HDim = HDim + V * (site1 * site2.dag() + site2 * site1.dag())


psi_p = (sqrt(eta + eps) * site1 + sqrt(eta - eps) * site2) / sqrt(2 * eta)
psi_m = (- sqrt(eta - eps) * site1 + sqrt(eta + eps) * site2) / sqrt(2 * eta)
lam_p = 0.5 * (2 * w2 + eps + eta)
lam_m = 0.5 * (2 * w2 + eps - eta)
vals, vecs = HDim.eigenstates()
print(vals,[lam_p,lam_m])
print(vecs)
print([psi_p,psi_m])

#build optical dissipator
L_EM = opli.EM_dissipator(wXX, w2, eps, V, mu, gamma, EM_temp,4,4)
#print(L_EM)

#need initial conditions:
rho_1 = psi_m * psi_m.dag()#site1 * site1.dag()




no_steps = 10000
propto = 100.
time_steps = linspace(0., propto, no_steps)

expec = [gr * gr.dag(), psi_m * psi_m.dag(), psi_p * psi_p.dag(), biexc * biexc.dag(), psi_p * psi_m.dag()]

time_ev = mesolve(HDim, rho_1, time_steps, [L_EM], expec)
#
#
#
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(time_steps, time_ev.expect[0], label = 'Ground')
ax.plot(time_steps, time_ev.expect[1], label = 'Dark')
ax.plot(time_steps, time_ev.expect[2], label = 'Bright')
ax.plot(time_steps, time_ev.expect[3], label = 'Biexciton')
ax.legend(loc = 0)
plt.xlabel('Time', fontsize = 22)
plt.ylabel('Population', fontsize = 22)
plt.savefig('Populations_eps={0}.pdf'.format(eps))
plt.show()

#
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(time_steps, real(time_ev.expect[4]), label = 'Re bright')
ax.plot(time_steps, imag(time_ev.expect[4]), label = 'Im bright')
ax.legend(loc=2,prop={'size':6})
plt.xlabel('Time', fontsize = 22)
plt.ylabel('Coherences', fontsize = 22)
plt.show()
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(time_steps, real(time_ev.expect[6]), label = 'Re BD')
# ax.plot(time_steps, imag(time_ev.expect[6]), label = 'Im BD')
# ax.legend(loc = 0)
# plt.xlabel('Time', fontsize = 22)
# plt.ylabel('Coherences', fontsize = 22)
# plt.show()
