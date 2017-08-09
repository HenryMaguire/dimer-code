##### a function to construct and solve the RC master equation ####

# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg

from scipy import *

import enr_functions as excres

#import functions that builds Liouvillian
import RC_liouvillian_single as lio
import RC_liouvillian as twolio
import ENR_Liouvillian_func as enrlio
import Optical_liouvillian as opli
reload(excres)
reload(opli)

import time as time


def RCfunction(eps, V, Temp, wc, alpha, N, propto, stepsize):



	#Now define system operators:
	#TLS Hamiltonian:

	TLSid = qeye(2)
	sm = destroy(2)
	sp = sm.dag()
	sx = sm + sp
	sz = sigmaz()

	HTLS = 0.5 * (eps * sz + V * sx)


	#find eigenvalues of this matrix
	TLSeivals, TLSeivecs=HTLS.eigenstates()

	eta = TLSeivals[1]-TLSeivals[0] #energy splitting
	#print(eta)

	# define inverse temperature
	beta = 1. / (0.695 * Temp)



	# reaction coordinate parameters are:
	#calculate the RC parameters:
	gamma = 2.#sqrt((eps ** 2. + V ** 2.)) / (2. * pi * wc)  # free parameter that we fix to the system splitting
	wRC = 2. * pi * gamma * wc #RC splitting
	kappa= sqrt(pi * alpha * wRC / 2.)  # coupling strength between the TLS and RC
	#print(kappa)
	#print(wRC)
	#Now we need to define RC operators:
	b = tensor(destroy(N), qeye(2))
	bd = b.dag()



	#Hamiltonian for the TLS and RC:
	HamRC =  kappa * tensor(qeye(N), sz) * (b + bd) + wRC * bd * b + tensor(qeye(N), HTLS)

	#print(HamRC)
	#call function that  constructs the dissipator
	#st = time.time()

	LRC = lio.Liou(gamma, beta, b + bd, 2 * N, HamRC)

	#en = time.time()

	#print('the time taken in to build the dissipator was {0} seconds'.format(en-st))


	# Now we need initial states:
	#TLS
	inTLS = 0.5 * (qeye(2) + sz)#TLSeivecs[1] * TLSeivecs[1].dag()

	#RC
	RCnb = (1 / (exp( beta * wRC)-1))
	inRC = thermal_dm(N, RCnb)

	#initial state of the TLS-RC
	TLSRCunitaryin = tensor(inRC, inTLS)




	#The expectation values we want to analyse for the RC;
	#These are the excitonic states (eigenstates of the TLS Hamiltonian)
	#excitonRC = tensor(qeye(N), TLSeivecs[0] * TLSeivecs[0].dag())
	#ExcohRC = tensor(qeye(N), TLSeivecs[0] * TLSeivecs[1].dag())
	sitepop = tensor(qeye(N), 0.5 * (qeye(2) + sz))
	sitecoh = tensor(qeye(N), sp)

	unsteps = linspace(0, propto, stepsize)

	#ODE solver:
	RCsolution = mesolve(HamRC, TLSRCunitaryin, unsteps, [LRC], [sitepop, sitecoh])

	return RCsolution






def RCdimerfunction_ENR(wXX, w2, eps, V, Temp, wc, alpha, N, excitations, propto, stepsize):
#
# 	"""
# 	A function to calculate the RC dynamics using the excitation number restricted basis,
# 	for Dimer/two RC model. We assume that both environments are identical.
# 	The important definition to us are:
# 	wXX = Biexciton energy
# 	w2 = splitting for the lowest energy state
# 	eps = bias between the two molecules
# 	V = coherent coupling
# 	Temp = temperature of the phonon environment
# 	alpha, wc = coupling strength and cut - off frequency
# 	N = max number of basis states
# 	excitations = number of excitations to keep in the basis
# 	propto, stepsize = time evolution parameters
# 	"""

	#Now define system operators:
	#the dimer basis states:
	site1 = Qobj(array([0., 1.,0.,0.]))
	site2 = Qobj(array([0., 0.,1.,0.]))
	biexc = Qobj(array([1.,0.,0.,0.]))
	ground = Qobj(array([0., 0.,0.,1.]))



	# define inverse temperature
	beta = 1. / (0.695 * Temp)



	# reaction coordinate parameters are:
	#calculate the RC parameters:
	gamma = 2. #sqrt((eps ** 2. + V ** 2.)) / (2. * pi * wc1)  # free parameter that we fix to the system splitting
	wRC = 2. * pi * gamma * wc #RC splitting
	print wRC
	kappa = sqrt(pi * alpha * wRC / 2.)  # coupling strength between the TLS and RC



# 	#start defining the excitation restricted states:
# 	# the dimension list for the RCs is:
	dims = [N] * 2
#	#2 is the number of modes taken

# 	#and dimension of the sysetm:
	Nsys = 4
#
# 	#Load the ENR dictionaries
	nstates, state2idx, idx2state = excres._enr_state_dictionaries(dims, excitations)
#
 	# Now construct the destruction operators in the ENR:
 	#start with a temporary definition:
	atemp = excres.enr_destroy(dims, excitations)
 	#which outputs a list due to the list nature of dims


 	#Then tensor product with the identity for the system:
	a = [tensor(aa, qeye(Nsys)) for aa in atemp]
 	#again this is a list

	#then the  dimension of the hilbert space after the ENR truncation:
	NHil = nstates * Nsys
	print('With the ENR we have a Hilbert space of {0}'.format(NHil))
	print('While the full basis has {0}'.format(4 * N * N))





 	#now we need to define the system operators in terms of ENR basis
	s1s1d = tensor(excres.enr_unit(dims, excitations), site1 * site1.dag())
	s2s2d = tensor(excres.enr_unit(dims, excitations), site2 * site2.dag())
	grgrd = tensor(excres.enr_unit(dims, excitations), ground * ground.dag())
	bibid = tensor(excres.enr_unit(dims,excitations), biexc * biexc.dag())
	s1s2d = tensor(excres.enr_unit(dims,excitations), site1 * site2.dag())
	s2s1d = tensor(excres.enr_unit(dims,excitations), site2 * site1.dag())

 	# and Now the Dimer Hamiltonian

	HDim = (w2 + eps) * site1 * site1.dag() + w2 * site2 * site2.dag() + wXX * biexc * biexc.dag()
	HDim = HDim + V * (site1 * site2.dag() + site2 * site1.dag())

	# define the eigenstates
	eta = sqrt(eps ** 2. + 4. * V ** 2.)
	psi1 = (sqrt(eta + eps) * site1 + sqrt(eta - eps) * site2) / sqrt(2 * eta)
	psi2 = - ( - sqrt(eta - eps) * site1 + sqrt(eta + eps) * site2) / sqrt(2 * eta)
	energies, states = HDim.eigenstates()
	psi1, psi2 = states[1], states[2]
	expeclist = [grgrd] #[s1s1d, s2s2d, grgrd, bibid]
	expeclist = expeclist + [tensor(excres.enr_unit(dims,excitations), psi1 * psi1.dag())]#coherence between ground and dark state
	expeclist = expeclist + [tensor(excres.enr_unit(dims,excitations), psi2 * psi2.dag())]#coherence between ground and bright
	expeclist = expeclist + [bibid]
	expeclist = expeclist + [tensor(excres.enr_unit(dims,excitations), psi1 * psi2.dag())]#coherence between dark and biexc

	#Hamiltonian for the TLS and RC:
	HamRC = tensor(excres.enr_unit(dims, excitations),HDim) + kappa * ( s1s1d * (a[0] + a[0].dag()) + s2s2d * (a[1] + a[1].dag()))
	HamRC = HamRC + sum([wRC * aa.dag() * aa for aa in a])

 	#call function that constructs the dissipator
	st = time.time()
	EM_temp = 5700.0
	gam_op = 0.1/pi
 	#LRC = twolio.Liou((b1 + bd1), (b2 + bd2), gamma1, gamma2, beta, dimHil, HamRC)
	LRC = enrlio.enr_Liouvillian(HamRC, a, NHil, gamma, beta)
	LOP = opli.EM_dissipator(wXX, w2, eps, V, 0, gam_op, EM_temp, N, excitations)
	en = time.time()

	print('the time taken in to build the dissipator was {0} seconds'.format(en-st))


 	# Now we need initial states:
 	#Dimer
	inDim=grgrd

 	#RCs initial states
	RCnb = (1 / (exp( beta * wRC)-1))
	print 'began'
	rhoRCs = excres.enr_therm(dims, excitations, RCnb)
	RCsinitial = tensor(rhoRCs,qeye(Nsys))


 	#initial state of the Dim-RC
	psi0 = RCsinitial * inDim

	unsteps = linspace(0, propto, stepsize)

 	#ODE solver:
	RCsolution = mesolve(HamRC, psi0, unsteps, [LRC+LOP], expeclist, options = (Options(nsteps = 15000)))

# 	#write to file:
# 	file_name ='data/RC_dynamics_N={2}_expec={3}_alpha={0}wc={1}.dat'.format(alpha * pi / 2., wc, N, excitations)
# 	data = vstack((unsteps, RCsolution.expect[0], RCsolution.expect[1], RCsolution.expect[2],\
# 	RCsolution.expect[3], RCsolution.expect[4], RCsolution.expect[5], real(RCsolution.expect[6]), imag(RCsolution.expect[6])) )
# 	file_data_store(file_name, data.T, numtype="real",sep=",")

	return RCsolution
