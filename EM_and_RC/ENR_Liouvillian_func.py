
# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg

from scipy import *


import enr_functions as excres

#construct master equation for restricted ENR basis
def enr_Liouvillian(H, a, Nmax, gamma, beta):
	"""
	A function to construct the Liouvillian in the ENR basis.
	The parameters are defined such that:
	H = system Hamiltonian 
	a = the list of coupling matrices
	Nmax = the number of dimensions of the total RC RC Dim Hilbert space in the ENR basis 
	"""

	L=0.
	PsipreEta=0.
	PsipreX=0.
	PsipreXList=[]
	PsipreEtaList=[]

	all_energy, all_state = H.eigenstates()
	#Apre = [spre((aa+aa.dag())) for aa in a]
	#Apost = [spost(aa+aa.dag()) for aa in a]
	for aa in a:
    
		PsipreEta=0.
		PsipreX=0.
		for j in range(Nmax):
			for k in range(Nmax):
				xa=aa.dag()+aa
				A=xa.matrix_element(all_state[j].dag(),all_state[k])
				delE=(all_energy[j]-all_energy[k])

				if absolute(A) > 0.0:
					if abs(delE)>0.0:
						X =0.5 * pi * gamma *(all_energy[j]-all_energy[k])*(cosh((all_energy[j]-all_energy[k]) * beta / 2)/(sinh((all_energy[j]-all_energy[k]) * beta / 2.)))*A

 
						eta = 0.5 * pi * gamma * (all_energy[j]-all_energy[k])*A

						PsipreX = PsipreX + X * all_state[j] * all_state[k].dag()



						PsipreEta=PsipreEta+eta*all_state[j]*all_state[k].dag()





					else:
						X = pi * gamma * A / beta 
						PsipreX = PsipreX + X * all_state[j] * all_state[k].dag()
						

		Apre=spre(aa+aa.dag())
		Apost=spost(aa+aa.dag())
		L=L+(-spre((aa+aa.dag())*PsipreX))
		L=L+(Apre*spost(PsipreX))


		L=L+(spre(PsipreX)*Apost)
		L=L+(-spost(PsipreX*(aa+aa.dag())))
		L=L+(spre((aa+aa.dag())*PsipreEta))
		L=L+(Apre*spost(PsipreEta))
		L=L+(-spre(PsipreEta)*Apost)
		L=L+(-spost(PsipreEta*(aa+aa.dag())))           


	return L
