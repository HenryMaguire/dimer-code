##### a function to construct and solve the RC master equation ####

# make qutip available in the rest of the notebook
import qutip as qt
from qutip import Qobj, tensor, basis
#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg
from utils import J_multipolar, J_flat, rate_up, rate_down
from scipy import *
from dimer_tests import exciton_states



def lin_construct(O):
	Od = O.dag()
	L = 2. * qt.spre(O) * qt.spost(Od) - qt.spre(Od * O) - qt.spost(Od * O)
	return L

"""
def rate_up(w, beta, gamma):
	n = 1 / (exp(beta * w) - 1.)
	rate = 0.5 * pi * n *J_flat(w, gamma, w)
	return rate

def rate_down(w, beta, gamma):
	n = 1 / (exp(beta * w) - 1.)
	rate = 0.5 * pi * (n + 1. )*J_flat(w, gamma, w)
	return rate
"""

def	EM_dissipator(states, wXX, w2, eps, V, mu, gamma, EM_temp, J, N, exc):
#
# A function  to build the Liouvillian describing the processes due to the
# electromagnetic field (without Lamb shift contributions). The important
# parameters to consider here are:
#
#	wXX = biexciton splitting
#	w1 = splitting of site 1
#	eps = bias between site 1 and 2
#	V = tunnelling rate between dimer
#	mu = scale factor for dipole moment of site 2
# 	gamma = bare coupling to the environment.
#	EM_temp =  temperature of the electromagnetic environment
# 	N = the number of states in the RC
#	exc = number of excitations kept in the ENR basis
########################################################

	#important definitions for the the ENR functions:
	# the dimension list for the RCs is:
	dims = [N] * 2
	#2 is the number of modes taken
	w1 = w2
	#and dimension of the sysetm:
	Nsys = 4

	#Load the ENR dictionaries
	nstates, state2idx, idx2state = qt.enr_state_dictionaries(dims, exc)


	#boltzmann constant in eV
	k_b = 0.695
	thermal_energy = k_b * EM_temp
	beta = 1 / thermal_energy

	# the site basis is:
	bi = basis(4,3)
	b2 = basis(4,2)
	b1 = basis(4,1)
	gr = basis(4,0)
	# the eigenstate splitting is given by:
	eta = sqrt(eps ** 2. + 4. * V ** 2.)

	# and the eigenvalues are:
	lam_p = 0.5 * (2 * w2 + eps + eta)
	lam_m = 0.5 * (2 * w2 + eps - eta)

	# first we define the eigenstates:
	psi_p = (sqrt( eta - eps) * b1 + sqrt( eta + eps) * b2) / sqrt(2 * eta)
	psi_m = (- sqrt(eta + eps) * b1 + sqrt(eta - eps) * b2) / sqrt(2 * eta)
	dark = states[0]
	bright= states[1]
	psi_m = dark
	psi_p = bright
	# Now the system eigenoperators
	#ground -> dressed state transitions
	Alam_p = (sqrt( eta - eps) + mu * sqrt(eta + eps)) / sqrt(2 * eta)
	print Alam_p
	Alam_p *= gr * (psi_p.dag())
	Alam_p = tensor( Alam_p, qt.enr_identity(dims, exc))

	Alam_m = - (sqrt( eta + eps) - mu * sqrt(eta - eps)) / sqrt(2 * eta)
	print Alam_m
	Alam_m *= gr * (psi_m.dag())
	Alam_m = tensor(Alam_m, qt.enr_identity(dims, exc))

	#print(Alam_m)
	#dressed state -> biexciton transitions
	Alam_p_bi = (sqrt( eta - eps) + mu * sqrt(eta + eps)) / sqrt(2 * eta)
	print Alam_p_bi
	Alam_p_bi *= (psi_p) * (bi.dag())
	Alam_p_bi = tensor( Alam_p_bi, qt.enr_identity(dims, exc))


	Alam_m_bi = - (sqrt( eta + eps) - mu * sqrt(eta - eps)) / sqrt(2 * eta)
	print Alam_m_bi
	Alam_m_bi *= (psi_m) * (bi.dag())
	Alam_m_bi = tensor( Alam_m_bi, qt.enr_identity(dims, exc))
	print J
	# Now the dissipators and there associated rates are are given by:
	gam_p_emm = rate_down(lam_p, EM_temp, gamma, J, w1)
	L1_emission = lin_construct(Alam_p)

	gam_p_abs = rate_up(lam_p, EM_temp, gamma, J, w1)
	L1_absorption = lin_construct(Alam_p.dag())

	gam_m_emm = rate_down(lam_m, EM_temp, gamma, J, w1)
	L2_emission = lin_construct(Alam_m)

	gam_m_abs = rate_up(lam_m, EM_temp, gamma, J, w1)
	L2_absorption = lin_construct(Alam_m.dag())

	gam_bi_p_emm = rate_down(wXX-lam_p, EM_temp, gamma, J, w1)
	L3_emission = lin_construct(Alam_p_bi)

	gam_bi_p_abs = rate_up(wXX-lam_p, EM_temp, gamma, J, w1)
	L3_absorption = lin_construct(Alam_p_bi.dag())

	gam_bi_m_emm = rate_down(wXX-lam_m, EM_temp, gamma, J, w1)
	L4_emission = lin_construct(Alam_m_bi)

	gam_bi_m_abs = rate_up(wXX-lam_m, EM_temp, gamma, J, w1)
	L4_absorption = lin_construct(Alam_m_bi.dag())

	#So the Liouvillian
	Li = gam_p_emm * L1_emission + gam_p_abs * L1_absorption
	Li = Li + gam_m_emm * L2_emission + gam_m_abs * L2_absorption
	Li = Li + gam_bi_p_emm * L3_emission + gam_bi_p_abs * L3_absorption
	Li = Li + gam_bi_m_emm * L4_emission + gam_bi_m_abs * L4_absorption

	return Li
