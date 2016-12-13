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
    if time_units == 'ps': # allows conversion to picoseconds, I can't remember what the exact number is though and cba to find it
        conversion == 7.13
    else:
        pass
    n =0.
    if T ==0. or omega ==0.: # stop divergences safely
        n = 0.
    else:
        try:
            beta = 1. / (conversion* T)
            n = float(1./(sp.exp(omega*beta)-1))
        except ZeroDivisionError:
            n = 0
    return n


def rate_up(w, n, gamma):
    rate = 0.5 * pi * gamma * n
    return rate

def rate_down(w, n, gamma):
    rate = 0.5 * pi * gamma * (n + 1. )
    return rate

def L_nonsecular(H_vib, sig, alpha, T, time_units='cm'):
    #Construct non-secular liouvillian
    ti = time.time()
    d = H_vib.shape[0]
    evals, evecs = H_vib.eigenstates()
    X1, X2, X3, X4 = 0, 0, 0, 0
    for i in range(int(d)):
        for j in range(int(d)):
            eps_ij = abs(evals[i]-evals[j])
            sig_ij = sig.matrix_element(evecs[i].dag(), evecs[j])
            sig_ji = (sig.dag()).matrix_element(evecs[j].dag(), evecs[i])
            Occ = Occupation(eps_ij, T, time_units)
            IJ = evecs[i]*evecs[j].dag()
            JI = evecs[j]*evecs[i].dag()

            if abs(sig_ij.real)>0 or abs(sig_ji.real)>0:
                if abs(eps_ij.real)>0:
                    X3+= rate_down(eps_ij, Occ, alpha)*sig_ij*IJ
                    X4+= rate_up(eps_ij, Occ, alpha)*sig_ij*IJ
                    X1+= rate_up(eps_ij, Occ, alpha)*sig_ji*JI
                    X2+= rate_down(eps_ij, Occ, alpha)*sig_ji*JI

    L = spre(sig*X1) -sprepost(X1,sig)+spost(X2*sig)-sprepost(sig,X2)
    L+= spre(sig.dag()*X3)-sprepost(X3, sig.dag())+spost(X4*sig.dag())-sprepost(sig.dag(), X4)
    print "It took ", time.time()-ti, " seconds to build the Non-secular RWA Liouvillian"
    return -L

def L_vib_lindblad(H_vib, sig, alpha, T, time_units='cm'):
    '''
    Initially assuming that the vibronic eigenstructure has no
    degeneracy and the secular approximation has been made
    '''
    d = H_vib.shape[0]
    ti = time.time()
    L = 0
    eig = H_vib.eigenstates()
    eVals = eig[0]
    eVecs = eig[1] # come out like kets
    l = 0
    occs=[]
    for m in range(int(d)):
        #print " Liouvillian is ", (float(m)/H_vib.shape[0])*100, " percent complete after ", int(time.time()-ti), " seconds. (", int((time.time()-ti)/60.), " minutes)."
        #print "There were ", l, " non-zero contributions."
        l = 0
        for n in range(int(d)):
            t_0 = time.time() # initial time reference for tracking slow calculations
            lam_nm = sig.matrix_element(eVecs[m].dag(), eVecs[n])
            #lam_mn = A.matrix_element(eVecs[n].dag(), eVecs[m])
            lam_nm_sq = lam_nm*lam_nm.conjugate()
            eps_mn = abs(eVals[m]-eVals[n])
            if lam_nm_sq>0:
                if abs(eps_mn.real)>0:
                    MN = eVecs[m]*eVecs[n].dag()
                    NM = eVecs[n]*eVecs[m].dag()
                    NN = eVecs[n]*eVecs[n].dag()
                    MM = eVecs[m]*eVecs[m].dag()

                    Occ = Occupation(eps_mn, T, time_units)
                    g2 = rate_up(eps_mn, Occ, alpha)
                    g1 = rate_down(eps_mn, Occ, alpha)

                    #T1 = 0.5*rate_up(eps_mn, Occ)*(spre(NN) - 2*sprepost(MN, NM)) + 0.5*rate_down(eps_mn, Occ)*(spre(MM) - 2*sprepost(NM, MN))
                    T1 = g2*lam_nm_sq*(spre(MM))+g1*lam_nm_sq*(spre(NN))
                    T2 = g2.conjugate()*lam_nm_sq*(spost(MM))+g1.conjugate()*lam_nm_sq*(spost(NN))
                    T3 = 2*(g2*lam_nm_sq*sprepost(NM, MN)+g1*lam_nm_sq*(sprepost(MN,NM)))
                    L += (T1 + T2 - T3)
                    l+=1

    print "It took ", time.time()-ti, " seconds to build the vibronic Lindblad Liouvillian"
    #eMatrix = np.array(eMatrix).reshape((H_vib.shape[0], H_vib.shape[0]))
    #plt.imshow(eMatrix)
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
