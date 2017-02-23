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
from qutip import Qobj, basis, destroy, tensor, qeye, spost, spre, sprepost, steadystate, enr_state_dictionaries, enr_identity
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
    beta = 0.
    if T ==0.: # First calculate beta
        n = 0.
        beta = np.infty
    else:
        # no occupation yet, make sure it converges
        beta = 1. / (conversion*T)
        if sp.exp(omega*beta)-1 ==0.:
            n = 0.
        else:
            n = float(1./(sp.exp(omega*beta)-1))
    return n


def J_multipolar(omega, Gamma, omega_0):
    return Gamma*(omega**3)/(2*np.pi*(omega_0**3))

def J_minimal(omega, Gamma, omega_0):
    return Gamma*omega/(2*np.pi*omega_0)

def J_flat(omega, Gamma, omega_0):
    return Gamma

def rate_up(w, n, gamma, J, w_0):
    rate = 0.5 * pi * n * J(w, gamma, w_0)
    return rate

def rate_down(w, n, gamma, J, w_0):
    rate = 0.5 * pi * (n + 1. ) * J(w, gamma, w_0)
    return rate

def lin_construct(O):
    Od = O.dag()
    L = 2. * spre(O) * spost(Od) - spre(Od * O) - spost(Od * O)
    return L

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

def	electronic_lindblad(wXX, w1, eps, V, mu, gamma, T, N_1, N_2,  exc, J):
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
    ti = time.time()
    #important definitions for the the ENR functions:
    # the dimension list for the RCs is:
    dims = [N_1, N_2]
    #2 is the number of modes taken

    #and dimension of the sysetm:
    Nsys = 4

    #Load the ENR dictionaries
    nstates, state2idx, idx2state = enr_state_dictionaries(dims, exc)


    #boltzmann constant in eV

    # the site basis is:
    bi = Qobj(np.array([0., 0., 0., 1.]))
    b1 = Qobj(np.array([0., 0., 1., 0.]))
    b2 = Qobj(np.array([0., 1., 0., 0.]))
    gr = Qobj(np.array([1., 0., 0., 0.]))

    # the eigenstate splitting is given by:
    eta = np.sqrt(eps ** 2. + 4. * V ** 2.)
    w_0 = w1 + 0.5*eps
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
    n = Occupation(lam_p, T)
    gam_p_emm = rate_down(lam_p, n, gamma, J, w_0)
    L1_emission = lin_construct(Alam_p)
    gam_p_abs = rate_up(lam_p, n, gamma, J, w_0)
    L1_absorption = lin_construct(Alam_p.dag())

    n = Occupation(lam_m, T)
    gam_m_emm = rate_down(lam_m, n, gamma, J, w_0)
    L2_emission = lin_construct(Alam_m)
    gam_m_abs = rate_up(lam_m, n, gamma, J, w_0)
    L2_absorption = lin_construct(Alam_m.dag())

    n = Occupation(wXX-lam_p, T)
    gam_bi_p_emm = rate_down(wXX-lam_p, n, gamma, J, w_0)
    L3_emission = lin_construct(Alam_p_bi)
    gam_bi_p_abs = rate_up(wXX-lam_p, n, gamma, J, w_0)
    L3_absorption = lin_construct(Alam_p_bi.dag())

    n = Occupation(wXX-lam_m, T)
    gam_bi_m_emm = rate_down(wXX-lam_m, n, gamma, J, w_0)
    L4_emission = lin_construct(Alam_m_bi)
    gam_bi_m_abs = rate_up(wXX-lam_m, n, gamma, J, w_0)
    L4_absorption = lin_construct(Alam_m_bi.dag())


    #So the Liouvillian
    Li = gam_p_emm * L1_emission + gam_p_abs * L1_absorption
    Li = Li + gam_m_emm * L2_emission + gam_m_abs * L2_absorption
    Li = Li + gam_bi_p_emm * L3_emission + gam_bi_p_abs * L3_absorption
    Li = Li + gam_bi_m_emm * L4_emission + gam_bi_m_abs * L4_absorption
    print "Naive Lindblad took ",time.time()-ti," seconds to compute"
    return Li

if __name__ == "__main__":
    ev_to_inv_cm = 8065.5
    w_1, w_2 = 1.4*ev_to_inv_cm, 1.*ev_to_inv_cm
    eps = (w_1 + w_2)/2 # Hack to make the spectral density work
    V = 200.
    w_xx = w_1+w_2+V
    T_1, T_2 = 300., 300.
    wRC_1, wRC_2 = 300., 300.
    alpha_1, alpha_2 = 100./np.pi, 100./np.pi
    wc =53.
    N_1, N_2= 3, 3

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

    L_RC, H_vib, A_1, A_2, A_EM, wRC_1, wRC_2 = RC.RC_mapping_UD(w_1, w_2, w_xx, V, T_1, T_2, wRC_1, wRC_2, alpha_1, alpha_2, wc,  N_1, N_2=N_2, mu=1, time_units='cm') # test that it works

    print "Is L_RC a completely positive map? -", L_RC.iscp
    print "Is it trace-preserving? -", L_RC.istp
    # Non-secular version
    L_ns = L_nonsecular(H_vib, A_EM, eps, Gamma_EM, T_EM, J_minimal, time_units='cm')
    print "Is L_ns a completely positive map? -", L_ns.iscp
    print "Is it trace-preserving? -", L_ns.istp
    ss_ns = steadystate(H_vib, [L_RC + L_ns]).ptrace(0)
    print "Non-sec steady-state dimer DM is, "
    print ss_ns
    print "trace = ",ss_ns.tr()
    # Secular version
    L_s = L_vib_lindblad(H_vib, A_EM, eps, Gamma_EM, T_EM, J_minimal, time_units='cm')
    print "dimer_driving_liouv is finished."
    print "Is L_s a completely positive map? -", L_s.iscp
    print "Is it trace-preserving? -", L_s.istp
    ss_s = steadystate(H_vib, [L_RC + L_s]).ptrace(0)
    print "Sec steady-state dimer DM is, "
    print ss_s
    print ss_s.tr()

    # Naive version
    L_n = electronic_lindblad(w_xx, w_1, w_1-w_2, V, mu, Gamma_EM, T_EM, N_1, N_1,  N_1+N_2, J_minimal)
    print "Is L_n a completely positive map? -", L_n.iscp
    print "Is it trace-preserving? -", L_n.istp
    ss_n = steadystate(H_vib, [L_RC + L_n]).ptrace(0)
    print "Naive steady-state dimer DM is, "
    print ss_n
    print ss_n.tr()
    real_therm = (((-1./(0.695*T_EM))*H_vib).expm().ptrace(0))/(((-1./(0.695*T_EM))*H_vib).expm().tr())
    # This is just a thermal state of the TLS-RC with respect to the electromagnetic bath only.

    #print L_RC.dims == L_ns.dims, L_RC.dims == L_s.dims, L_ns.dims ==L_s.dims, L_n.dims == L_RC.dims
