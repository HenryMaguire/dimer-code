"""
Jake's electronic Lindblads
"""
import numpy as np
from numpy import pi
import scipy as sp
from qutip import Qobj, basis, destroy, tensor, qeye, spost, spre, sprepost, enr_state_dictionaries, enr_identity
import time
from dimer_driving_liouv import rate_up, rate_down

def J_multipolar(omega, Gamma, omega_0):
    return Gamma*(omega**3)/(2*np.pi*(omega_0**3))

def J_minimal(omega, Gamma, omega_0):
    return Gamma*omega/(2*np.pi*omega_0)

def J_flat(omega, Gamma, omega_0):
    return Gamma

def rate_up(w, n, gamma):
    rate = 0.5 * pi * n
    return rate

def rate_down(w, n, gamma):
    rate = 0.5 * pi * (n + 1. )
    return rate

def lin_construct(O):
    Od = O.dag()
    L = 2. * spre(O) * spost(Od) - spre(Od * O) - spost(Od * O)
    return L
def	electronic_lindblad(wXX, w1, eps, V, mu, gamma, EM_temp, N_1, N_2,  exc):
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
	dims = [N_1, N_2]
	#2 is the number of modes taken

	#and dimension of the sysetm:
	Nsys = 4

	#Load the ENR dictionaries
	nstates, state2idx, idx2state = enr_state_dictionaries(dims, exc)


	#boltzmann constant in eV
	k_b = 0.695
	thermal_energy = k_b * EM_temp
	beta = 1 / thermal_energy

	# the site basis is:
	bi = Qobj(np.array([1., 0., 0., 0.]))
	b1 = Qobj(np.array([0., 1., 0., 0.]))
	b2 = Qobj(np.array([0., 0., 1., 0.]))
	gr = Qobj(np.array([0., 0., 0., 1.]))

	# the eigenstate splitting is given by:
	eta = np.sqrt(eps ** 2. + 4. * V ** 2.)

	# and the eigenvalues are:
	lam_p = 0.5 * (2 * w1 + eps + eta)
	lam_m = 0.5 * (2 * w1 + eps - eta)

	# first we define the eigenstates:
	psi_p = (np.sqrt( eta - eps) * b1 + np.sqrt( eta + eps) * b2) / np.sqrt(2 * eta)
	psi_m = (- np.sqrt(eta + eps) * b1 + np.sqrt(eta - eps) * b2) / np.sqrt(2 * eta)

	# Now the system eigenoperators
	#ground -> dressed state transitions
	Alam_p = (np.sqrt( eta - eps) + (1 - mu) * np.sqrt(eta + eps)) / np.sqrt(2 * eta) * gr * (psi_p.dag())
	Alam_p = tensor(Alam_p, enr_identity(dims, exc))

	Alam_m = - (np.sqrt( eta + eps) - (1 - mu) * np.sqrt(eta - eps)) / np.sqrt(2 * eta) * gr * (psi_m.dag())
	Alam_m = tensor(Alam_m, enr_identity(dims, exc))

	#print(Alam_m)
	#dressed state -> biexciton transitions
	Alam_p_bi = (np.sqrt( eta - eps) + (1 - mu) * np.sqrt(eta + eps)) / np.sqrt(2 * eta) * (psi_p) * (bi.dag())
	Alam_p_bi = tensor(Alam_p_bi,enr_identity(dims, exc))

	Alam_m_bi = - (np.sqrt( eta + eps) - (1 - mu) * np.sqrt(eta - eps)) / np.sqrt(2 * eta)  * (psi_m) * (bi.dag())
	Alam_m_bi = tensor( Alam_m_bi,enr_identity(dims, exc))

	# Now the dissipators and there associated rates are are given by:
	gam_p_emm = rate_down(lam_p, beta, gamma)
	L1_emission = lin_construct(Alam_p)

	gam_p_abs = rate_up(lam_p, beta, gamma)
	L1_absorption = lin_construct(Alam_p.dag())

	gam_m_emm = rate_down(lam_m, beta, gamma)
	L2_emission = lin_construct(Alam_m)

	gam_m_abs = rate_up(lam_m, beta, gamma)
	L2_absorption = lin_construct(Alam_m.dag())

	gam_bi_p_emm = rate_down(wXX-lam_p, beta, gamma)
	L3_emission = lin_construct(Alam_p_bi)

	gam_bi_p_abs = rate_up(wXX-lam_p, beta, gamma)
	L3_absorption = lin_construct(Alam_p_bi.dag())

	gam_bi_m_emm = rate_down(wXX-lam_m, beta, gamma)
	L4_emission = lin_construct(Alam_m_bi)

	gam_bi_m_abs = rate_up(wXX-lam_m, beta, gamma)
	L4_absorption = lin_construct(Alam_m_bi.dag())


	#So the Liouvillian
	Li = gam_p_emm * L1_emission + gam_p_abs * L1_absorption
	Li = Li + gam_m_emm * L2_emission + gam_m_abs * L2_absorption
	Li = Li + gam_bi_p_emm * L3_emission + gam_bi_p_abs * L3_absorption
	Li = Li + gam_bi_m_emm * L4_emission + gam_bi_m_abs * L4_absorption

	return Li
