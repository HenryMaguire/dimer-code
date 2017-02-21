"""
The four electromagnetic liouvillians I am studying for the vibronic dimer are:
- no secular approximation
- a secular approximation
- an approximation which says that the enlarged system eigenstates are the same as the
    uncoupled system eigenstates (found in electronic_lindblad)

"""

import numpy as np
from numpy import pi
import scipy as sp
from qutip import basis, destroy, tensor, qeye, spost, spre, sprepost
import time

import dimer_UD_liouv as RC
reload(RC)


def Occupation(omega, T, time_units='cm'):
    conversion = 0.695
    if time_units == 'ev':
        conversion == 8.617E-5
    if time_units == 'ps':
        conversion == 0.131
    else:
        pass
    n =0.
    if T ==0. or omega ==0.: # stop divergences safely
        n = 0.
    else:
        try:
            beta = 1. / (conversion*T)
            n = float(1./(sp.exp(omega*beta)-1))
        except RuntimeError:
            n = 0.
    return n


def J_multipolar(omega, Gamma, omega_0):
    return Gamma*(omega**3)/(2*np.pi*(omega_0**3))

def J_minimal(omega, Gamma, omega_0):
    return Gamma*omega/(2*np.pi*omega_0)

def J_flat(omega, Gamma, omega_0):
    return Gamma

def rate_up(w, n, gamma):
    rate = 0.5 * pi * gamma * n
    return rate

def rate_down(w, n, gamma):
    rate = 0.5 * pi * gamma * (n + 1. )
    return rate

def L_nonsecular(H_vib, A, eps, Gamma, T, J, time_units='cm'):
    #Construct non-secular liouvillian
    ti = time.time()
    d = H_vib.shape[0]
    evals, evecs = H_vib.eigenstates()
    X1, X2, X3, X4 = 0, 0, 0, 0
    for i in range(int(d)):
        for j in range(int(d)):
            eps_ij = abs(evals[i]-evals[j])
            A_ij = A.matrix_element(evecs[i].dag(), evecs[j])
            A_ji = (A.dag()).matrix_element(evecs[j].dag(), evecs[i])
            Occ = Occupation(eps_ij, T, time_units)
            IJ = evecs[i]*evecs[j].dag()
            JI = evecs[j]*evecs[i].dag()
            # 0.5*np.pi*alpha*(N+1)
            if abs(A_ij)>0 or abs(A_ji)>0:
                r_up = 2*np.pi*J(eps_ij, Gamma, eps)*Occ
                r_down = 2*np.pi*J(eps_ij, Gamma, eps)*(Occ+1)
                X3+= r_down*A_ij*IJ
                X4+= r_up*A_ij*IJ
                X1+= r_up*A_ji*JI
                X2+= r_down*A_ji*JI

    L = spre(A*X1) -sprepost(X1,A)+spost(X2*A)-sprepost(A,X2)
    L+= spre(A.dag()*X3)-sprepost(X3, A.dag())+spost(X4*A.dag())-sprepost(A.dag(), X4)
    print "It took ", time.time()-ti, " seconds to build the Non-secular RWA Liouvillian"
    return -0.5*L


def L_vib_lindblad(H_vib, A, eps, Gamma, T, J, time_units='cm'):
    '''
    Initially assuming that the vibronic eigenstructure has no
    degeneracy and the secular approximation has been made
    '''
    ti = time.time()
    d = H_vib.shape[0]
    ti = time.time()
    L = 0
    eig = H_vib.eigenstates()
    eVals = eig[0]
    eVecs = eig[1] # come out like kets
    l = 0
    occs=[]
    for i in range(int(d)):
        l = 0
        for j in range(int(d)):
            t_0 = time.time() # initial time reference for tracking slow calculations
            lam_ij = A.matrix_element(eVecs[i].dag(), eVecs[j])
            #lam_mn = (A.dag()).matrix_element(eVecs[n].dag(), eVecs[m])
            lam_ij_sq = lam_ij*lam_ij.conjugate()
            eps_ij = abs(eVals[i]-eVals[j])
            if lam_ij_sq>0:
                IJ = eVecs[i]*eVecs[j].dag()
                JI = eVecs[j]*eVecs[i].dag()
                JJ = eVecs[j]*eVecs[j].dag()
                II = eVecs[i]*eVecs[i].dag()

                Occ = Occupation(eps_ij, T, time_units)
                r_up = 2*np.pi*J(eps_ij, Gamma, eps)*Occ
                r_down = 2*np.pi*J(eps_ij, Gamma, eps)*(Occ+1)

                T1 = r_up*spre(II)+r_down*spre(JJ)
                T2 = r_up.conjugate()*spost(II)+r_down.conjugate()*spost(JJ)
                T3 = (r_up*sprepost(JI, IJ)+r_down*sprepost(IJ,JI))
                L += lam_ij_sq*(0.5*(T1 + T2) - T3)
                l+=1

    print "It took ", time.time()-ti, " seconds to build the vibronic Lindblad Liouvillian"
    return -L

if __name__ == "__main__":
    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()

    L_RC, H_vib, A_1, A_2, A_EM, wRC_1, wRC_2 = RC.RC_mapping_UD(sigma_m1, sigma_m2, 9000, 9000, 200, 100., 100., 300., 300., 10., 10., 53,  2) # test that it works
    L_ns = L_nonsecular(H_vib, A_EM, 0.3, 100.)
    L_s = L_vib_lindblad(H_vib, A_EM, 0.3, 100.)
    print "dimer_driving_liouv is finished."
    print "Is L_s a completely positive map? -", L_s.iscp
    print "Is it trace-preserving? -", L_s.istp

    print "Is L_ns a completely positive map? -", L_ns.iscp
    print "Is it trace-preserving? -", L_ns.istp
