"""
The four electromagnetic liouvillians I am studying for the vibronic dimer are:
- no secular approximation
- a secular approximation
- an approximation which says that the enlarged system eigenstates are the same as the
    uncoupled system eigenstates (found in electronic_lindblad)

"""
import time

import numpy as np
from numpy import pi
from qutip import Qobj, basis, spost, spre, sprepost, steadystate
import qutip.parallel as par

from utils import *
import dimer_phonons as RC

reload(RC)

def nonsecular_function(i,j, eVals=[], eVecs=[], A=0, eps=1000., Gamma=1.,T=0., J=J_minimal):
    X1, X2, X3, X4 = 0, 0, 0, 0
    eps_ij = abs(eVals[i]-eVals[j])
    A_ij = A.matrix_element(eVecs[i].dag(), eVecs[j])
    A_ji = (A.dag()).matrix_element(eVecs[j].dag(), eVecs[i])
    Occ = Occupation(eps_ij, T)
    IJ = eVecs[i]*eVecs[j].dag()
    JI = eVecs[j]*eVecs[i].dag()
    # 0.5*np.pi*alpha*(N+1)
    if abs(A_ij)>0 or abs(A_ji)>0:
        r_up = 2*pi*J(eps_ij, Gamma, eps)*Occ
        r_down = 2*pi*J(eps_ij, Gamma, eps)*(Occ+1)
        X3= r_down*A_ij*IJ
        X4= r_up*A_ij*IJ
        X1= r_up*A_ji*JI
        X2= r_down*A_ji*JI
    return Qobj(X1), Qobj(X2), Qobj(X3), Qobj(X4)

def secular_function(i,j, eVals=[], eVecs=[], A=0, eps=1000., Gamma=1.,T=0., J=J_minimal):
    L = 0
    lam_ij = A.matrix_element(eVecs[i].dag(), eVecs[j])
    #lam_mn = (A.dag()).matrix_element(eVecs[n].dag(), eVecs[m])
    lam_ij_sq = lam_ij*lam_ij.conjugate()
    eps_ij = abs(eVals[i]-eVals[j])
    if lam_ij_sq>0:
        IJ = eVecs[i]*eVecs[j].dag()
        JI = eVecs[j]*eVecs[i].dag()
        JJ = eVecs[j]*eVecs[j].dag()
        II = eVecs[i]*eVecs[i].dag()

        Occ = Occupation(eps_ij, T)
        r_up = 2*pi*J(eps_ij, Gamma, eps)*Occ
        r_down = 2*pi*J(eps_ij, Gamma, eps)*(Occ+1)

        T1 = r_up*spre(II)+r_down*spre(JJ)
        T2 = r_up.conjugate()*spost(II)+r_down.conjugate()*spost(JJ)
        T3 = (r_up*sprepost(JI, IJ)+r_down*sprepost(IJ,JI))
        L = lam_ij_sq*(0.5*(T1 + T2) - T3)
    return Qobj(L)


def L_nonsecular(H_vib, A, eps, Gamma, T, J, num_cpus=1):
    #Construct non-secular liouvillian
    ti = time.time()
    dim_ham = H_vib.shape[0]
    eVals, eVecs = H_vib.eigenstates()
    names = ['eVals', 'eVecs', 'A', 'eps', 'Gamma', 'T', 'J']
    kwargs = dict()
    for name in names:
        kwargs[name] = eval(name)
    l = dim_ham*range(dim_ham) # Perform two loops in one
    X1, X2, X3, X4 = par.parfor(nonsecular_function, sorted(l), l,
                                            num_cpus=num_cpus, **kwargs)
    X1, X2, X3, X4 = np.sum(X1), np.sum(X2), np.sum(X3), np.sum(X4)
    L = spre(A*X1) -sprepost(X1,A)+spost(X2*A)-sprepost(A,X2)
    L+= spre(A.dag()*X3)-sprepost(X3, A.dag())+spost(X4*A.dag())-sprepost(A.dag(), X4)
    print "It took ", time.time()-ti, " seconds to build the Non-secular RWA Liouvillian"
    return -0.5*L


def L_secular(H_vib, A, eps, Gamma, T, J, num_cpus=1):
    '''
    Initially assuming that the vibronic eigenstructure has no
    degeneracy and the secular approximation has been made
    '''
    ti = time.time()
    dim_ham = H_vib.shape[0]
    eVals, eVecs = H_vib.eigenstates()
    names = ['eVals', 'eVecs', 'A', 'eps', 'Gamma', 'T', 'J']
    kwargs = dict()
    for name in names:
        kwargs[name] = eval(name)
    l = dim_ham*range(dim_ham) # Perform two loops in one
    L = np.sum(par.parfor(secular_function, sorted(l), l,
                                            num_cpus=num_cpus, **kwargs))

    print "It took ", time.time()-ti, " seconds to build the vibronic Lindblad Liouvillian"
    return -L


if __name__ == "__main__":
    ev_to_inv_cm = 8065.5
    w_1, w_2 = 1.4*ev_to_inv_cm, 1.*ev_to_inv_cm
    eps = (w_1 + w_2)/2 # Hack to make the spectral density work
    V = 200.
    w_xx = w_1+w_2+V
    T_1, T_2 = 300., 300.
    wRC_1, wRC_2 = 300., 300.
    alpha_1, alpha_2 = 10./pi, 10./pi
    wc =53.
    N_1, N_2= 4, 4
    exc = int((N_1+N_2)*0.75)
    mu=1
    T_EM = 6000.
    Gamma_EM = 6.582E-4*ev_to_inv_cm

    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()

    L_RC, H_vib, A_1, A_2, A_EM, wRC_1, wRC_2 = RC.RC_mapping_UD(
                                w_1, w_2, w_xx, V, T_1, T_2, wRC_1, wRC_2,
                                alpha_1, alpha_2, wc,  N_1, N_2, exc, mu=1, num_cpus=1)
    print "*******   L_RC  *******"
    #print "Is L_RC a completely positive map? -", L_RC.iscp
    #print "Is it trace-preserving? -", L_RC.istp
    # Non-secular version
    print "*******   L_NS  *******"
    L_ns = L_nonsecular(H_vib, A_EM, eps, Gamma_EM, T_EM, J_minimal)
    #print "Is L_ns a completely positive map? -", L_ns.iscp
    #print "Is it trace-preserving? -", L_ns.istp
    ss_ns = steadystate(H_vib, [L_RC + L_ns]).ptrace(0)
    print "Non-sec steady-state dimer DM is, "
    print ss_ns
    #print "trace = ",ss_ns.tr()
    # Secular version
    print "*******   L_S  *******"
    L_s = L_secular(H_vib, A_EM, eps, Gamma_EM, T_EM, J_minimal)
    print "dimer_driving_liouv is finished."
    #print "Is L_s a completely positive map? -", L_s.iscp
    #print "Is it trace-preserving? -", L_s.istp
    ss_s = steadystate(H_vib, [L_RC + L_s]).ptrace(0)
    print "Sec steady-state dimer DM is, "
    print ss_s
    #print ss_s.tr()

    # Naive version
    """
    L_n = electronic_lindblad(w_xx, w_1, w_1-w_2, V, mu, Gamma_EM,
                                T_EM, N_1, N_1,  N_1+N_2, J_minimal)
    print "Is L_n a completely positive map? -", L_n.iscp
    print "Is it trace-preserving? -", L_n.istp
    ss_n = steadystate(H_vib, [L_RC + L_n]).ptrace(0)
    print "Naive steady-state dimer DM is, "
    print ss_n
    print ss_n.tr()
    """
    real_therm = (((-1./(0.695*T_EM))*H_vib).expm().ptrace(0))/(((-1./(0.695*T_EM))*H_vib).expm().tr())
    # This is just a thermal state of the TLS-RC with respect to the electromagnetic bath only.
    print real_therm
    #print L_RC.dims == L_ns.dims, L_RC.dims == L_s.dims, L_ns.dims ==L_s.dims, L_n.dims == L_RC.dims
