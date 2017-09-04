import pickle
import os
# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg

from scipy import *
##### a function to construct and solve the RC master equation ####
import RC_function as RC_f
reload(RC_f)

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

def save_obj(obj, name ):
	with open(name + '.pickle', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# define the site basis is:
biexc = basis(4, 0)
site1 = basis(4, 1)
site2 = basis(4, 2)
gr = basis(4, 3)

# and in density matrix form
# rhogr = gr * (gr.dag())
# rhobi = bi * (bi.dag())
# rhopsi_p = b1 * b1.dag()
# rhopsi_m = b2 * b2.dag()#


#data_maker(100., 0., 20, 50, 1, 0., 0., 5, '2a', 1, )
#data_maker(100., 10., 20, 50, 1, 0., 0., 10, '2b', 1)
#data_maker(1500., 50., 100, 5700, 0.1, 0., 0., 2, 1, '4ab', 0, make_new_data=True)
#data_maker(1500., 50., 100, 5700, 0.1, 2., 2., 3, 4, '4cd', 0, make_new_data=True)

def data_maker(w2, eps, V, T_EM, alpha_EM, alpha_1, alpha_2, N, propto, figure_num, initial, make_new_data=False):
	T = 300.
	k_b = 0.695
	thermal_energy = k_b * T
	beta = float(1. / thermal_energy)
	wc = 53.08
	excitations = N*2
	if N>4:
		excitations = N/2

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
	#print(vals,[lam_p,lam_m])
	#print(vecs)
	#print([psi_p,psi_m])

	#build optical dissipator
	#L_EM = opli.EM_dissipator(wXX, w2, eps, V, mu, gamma, EM_temp,4,4)
	#print(L_EM)

	#need initial conditions:
	#rho_1 = psi_m * psi_m.dag()#site1 * site1.dag()


	no_steps = 4000*propto

	#time_steps = linspace(0., propto, no_steps)

	#expec = [gr * gr.dag(), psi_m * psi_m.dag(), psi_p * psi_p.dag(), biexc * biexc.dag(), psi_p * psi_m.dag()]
	#time_ev = mesolve(HDim, rho_1, time_steps, [L_EM], expec)
	time_ev = RC_f.RCdimerfunction_ENR(wXX, w2, eps, V, T, T_EM, wc, alpha_1, alpha_EM, N, excitations, propto, no_steps, initial)
	data_dir = 'DATA/QHE_notes_fig{}/N{}_exc{}/'.format(figure_num, N, excitations)
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)
	save_obj(time_ev, data_dir+'jake_data')
	"""
	#
	#
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.plot(time_steps, time_ev.expect[0], label = 'Ground')
	ax.plot(time_steps, time_ev.expect[1], label = 'Site 1')
	ax.plot(time_steps, time_ev.expect[2], label = 'Site 2')
	#ax.plot(time_steps, time_ev.expect[4], label = 'Dark')
	#ax.plot(time_steps, time_ev.expect[5], label = 'Bright')
	ax.plot(time_steps, time_ev.expect[3], label = 'Biexciton')
	ax.legend(loc = 0)
	plt.xlabel('Time', fontsize = 22)
	plt.ylabel('Population', fontsize = 22)
	plt.savefig('Populations_eps={0}.pdf'.format(eps))
	plt.show()

	#
	fig = plt.figure()
	ax = plt.subplot(111)
	ax.plot(time_steps, real(time_ev.expect[6]), label = 'Re bright')
	ax.plot(time_steps, imag(time_ev.expect[6]), label = 'Im bright')
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
	"""
if __name__ == "__main__":
    '''figure 2'''

    data_maker(100., 0., 20, 50, 1., 0., 0., 2, 1, '2a', 1,  make_new_data=True)
    data_maker(100., 10., 20, 50, 1., 0., 0., 2, 10, '2b', 1, make_new_data=True)

    '''figure 4'''
    data_maker(1500., 50., 100., 5700., 0.1, 2., 2., 4, 1, '4ab', 0, make_new_data=True)
    data_maker(1500., 50., 100., 5700., 0.1, 2., 2., 4, 4, '4cd', 0, make_new_data=True)
    '''figure 5'''
    data_maker(1500., 50., 100, 5700, 0.1, 100/pi, 100/pi, 5, 1, '5ab-p', 0, make_new_data=True)
    data_maker(1500., 50., 100, 5700, 0.1, 100/pi, 100/pi, 5, 4, '5cd-p', 0, make_new_data=True)
    data_maker(1500., 50., 100, 5700, 0.1, 100., 100., 5, 1, '5ab', 0, make_new_data=True)
    data_maker(1500., 50., 100, 5700, 0.1, 100., 100., 5, 4, '5cd', 0, make_new_data=True)
