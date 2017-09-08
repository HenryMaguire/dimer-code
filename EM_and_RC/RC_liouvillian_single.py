
# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg

from scipy import *



def Liou(gamma, beta, coupop, dimHil, Ham):

    # find eigenvalues and vectors of the input Hamiltonian
    TLSbigvals, TLSbigvecs = Ham.eigenstates()

    # empty out operators
    L=0
    PsipreXi=0
    PsipreX=0

    # define the RC superoperators for pre- and post- multiplication of the density matrix

    Apre=spre(coupop)
    Apost=spost(coupop)

    #define the deliminators

    for j in range(dimHil):
        for k in range(dimHil):
             # writes the RC operator in the eigen basis by finding the overlap of each eigenstate with A
            A = (coupop).matrix_element(TLSbigvecs[j].dag(),TLSbigvecs[k])
            
            #difference in eigenvalues
            lamb = TLSbigvals[j] - TLSbigvals[k]
            
            if absolute(A) > 0.0:
                if absolute(lamb) > 0.0:
                    # construct the matrices and add to existing matrix
                    X = 0.5 * pi * gamma * lamb * ((cosh(lamb * beta / 2)) / sinh(lamb * beta / 2)) * A
                
                    xi = 0.5 * pi * gamma * lamb * A
                
                    PsipreX = PsipreX + X * TLSbigvecs[j] * dag(TLSbigvecs[k])
                
                    PsipreXi = PsipreXi + xi * TLSbigvecs[j] * dag(TLSbigvecs[k])
                
                else:
                    X = pi * gamma * A / beta
                    
                    
                    PsipreX = PsipreX + X * TLSbigvecs[j] * dag(TLSbigvecs[k])
                
                
                
    L=L+(-spre((coupop) * PsipreX))
    L=L+(Apre * spost(PsipreX))
    L=L+(spre(PsipreX) * Apost )
    L=L+(-spost(PsipreX * (coupop)))
                
    L=L+(spre((coupop) * PsipreXi))
    L=L+(Apre*spost(PsipreXi))
    L=L+(-spre(PsipreXi)*Apost)
    L=L+(-spost(PsipreXi*(coupop)))           
    return L

                