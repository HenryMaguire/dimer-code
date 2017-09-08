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
bi = Qobj([1., 0., 0., 0.])
b1 = Qobj([0., 1., 0., 0.])
b2 = Qobj([0., 0., 1., 0.])
gr = Qobj([0., 0., 0., 1.])
 
# and in density matrix form 
rhogr = gr * (gr.dag())
rhobi = bi * (bi.dag())


# the system parameters that can be defined globally are
w1 = 50. # splitting for site 1
V = 20.
mu = 0.
gamma = 1.
EM_temp = 50.0
k_b = 0.695
thermal_energy = k_b * EM_temp
beta = float(1. / thermal_energy)
	


ground = []
bright = []
dark = []
biexc = []



val1 = []
val2 = []
val3 = []
val4 = []

bias_list = [0.] #linspace(0.001,10.,50)

for k in range(len(bias_list)):
	
	eps = bias_list[k]
	
	# the eigenstate splitting is given by:
	eta = sqrt(eps ** 2. + 4. * V ** 2.)
	

	# first we define the eigenstates:
	# psi_p =  (sqrt( eta - eps) * b1 + sqrt( eta + eps) * b2) / sqrt(2 * eta)
# 	psi_m =  (- sqrt(eta + eps) * b1 + sqrt(eta - eps) * b2) / sqrt(2 * eta)
# 	
# 	rhopsi_p = psi_p * (psi_p.dag())
# 	rhopsi_m = psi_m * (psi_m.dag())
	
	w2 = w1 + eps
	wXX =  w1 + w2 

	# H4ls =  wXX * bi * (bi.dag()) + w2 *  b1 * (b1.dag()) +  w1 *  b2 * (b2.dag())+ V * ( b1 * (b2.dag()) +  b2 * (b1.dag()) )
# 	vals, vecs = H4ls.eigenstates()
# 	#print(vals)


	lam_p = 0.5 * (2 * w1 + eps + eta)
	lam_m = 0.5 * (2 * w1 + eps - eta)
	
	#print(lam_p, lam_m)
	
	Hsys = wXX * bi * (bi.dag()) + lam_p *  b1 * (b1.dag()) +  lam_m *  b2 * (b2.dag())#+ V * ( b1 * (b2.dag()) +  b2 * (b1.dag()) ) 

	#build optical dissipator
	L_EM = opli.EM_dissipator(wXX, w1, eps, V, mu, gamma, EM_temp)

	steady = steadystate(Hsys, [L_EM])
	
	ground = ground + [(rhogr * steady).tr()]
	bright = bright + [((b1 * b1.dag()) * steady).tr()]
	dark =  dark + [((b2 * b2.dag()) * steady).tr()]
	biexc = biexc + [(rhobi * steady).tr()]
	
	print(ground, bright, dark, biexc)






# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(bias_list, dark, label = 'Dark')
# ax.plot(bias_list, bright, label = 'Bright')
# ax.plot(bias_list, ground, label = 'ground')
# ax.plot(bias_list, biexc, label = 'biexc')
# ax.legend(loc = 0)
# plt.xlabel('Bias', fontsize = 22)
# plt.ylabel('Population', fontsize = 22)
# plt.show()