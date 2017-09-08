
# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg

from scipy import *



def Liou(B1, B2, gamma1, gamma2, beta, dimHil, Ham):

	# RC master equation with respect to the TLS eigenstructure:
	eigvals, eigvecs = Ham.eigenstates()

	# empty out operators
	L1=0
	PsipreXi1=0
	PsipreX1=0
	L2=0
	PsipreXi2=0
	PsipreX2=0



	Apre1 = spre(B1)
	Apost1 = spost(B2)

	Apre2 = spre(B1)
	Apost2 = spost(B2)

	#define the deliminators

	for j in range(dimHil):
		for k in range(dimHil):
			# writes the RC operator in the eigen basis by finding the overlap of each eigenstate with A
			A1 = (B1).matrix_element(eigvecs[j].dag(), eigvecs[k])
			A2 = (B2).matrix_element(eigvecs[j].dag(), eigvecs[k])

			#difference in eigenvalues
			lamb = eigvals[j]-eigvals[k]

			if absolute(A1) > 0.0:
				if absolute(lamb) > 0.0:
					X1 = 0.5 * pi * gamma1 * lamb * ((cosh(lamb * beta/2))/sinh(lamb * beta/2)) * A1
					xi1 = 0.5 * pi * gamma1 * lamb * A1
	

					PsipreX1 = PsipreX1 + X1 * eigvecs[j] * eigvecs[k].dag() # construct the matrices and add to existing matrix
					PsipreXi1 = PsipreXi1 + xi1 * eigvecs[j] * eigvecs[k].dag()

				else:
					X1 = pi * gamma1 * A1 / beta
					PsipreX1 = PsipreX1 + X1 * eigvecs[j] * eigvecs[k].dag()
	


			if absolute(A2) > 0.0:
				if absolute(lamb) > 0.0:

					X2 = 0.5 * pi * gamma2 * lamb * ((cosh(lamb * beta / 2.))/sinh(lamb * beta / 2.)) * A2
					xi2 = 0.5 * pi * gamma2 * lamb * A2

					PsipreX2 = PsipreX2 + X2 * eigvecs[j] * eigvecs[k].dag() # construct the matrices and add to existing matrix
					PsipreXi2 = PsipreXi2 + xi2 * eigvecs[j] * eigvecs[k].dag()

				else:

					X2 = pi * gamma2 * A2 / beta
					PsipreX2 = PsipreX2 + X2 * eigvecs[j] * eigvecs[k].dag()



	L1 = L1 + ( - spre((B1) * PsipreX1))
	L1 = L1 + (Apre1 * spost(PsipreX1))
	L1 = L1 + ( spre(PsipreX1) * Apost1 )
	L1 = L1 + ( - spost(PsipreX1 * (B1)))

	L1 = L1 + (spre((B1) * PsipreXi1))
	L1 = L1 + (Apre1 * spost(PsipreXi1))
	L1 = L1 + ( - spre(PsipreXi1) * Apost1)
	L1 = L1 + ( - spost(PsipreXi1 * (B1)))           

	L2 = L2 + ( - spre((B2) * PsipreX2))
	L2 = L2 + (Apre2 * spost(PsipreX2))
	L2 = L2 + ( spre(PsipreX2) * Apost2 )
	L2 = L2 + ( - spost(PsipreX2 * (B2)))

	L2 = L2 + (spre((B2) * PsipreXi2))
	L2 = L2 + (Apre2 * spost(PsipreXi2))
	L2 = L2 + ( - spre(PsipreXi2) * Apost2)
	L2 = L2 + ( - spost(PsipreXi2 * (B2)))           
	 
	L = L1 + L2
	return L

          
 