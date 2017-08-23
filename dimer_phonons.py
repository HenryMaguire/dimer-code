from qutip import Qobj,basis, ket, mesolve, qeye, tensor, thermal_dm, destroy, steadystate, spost, spre, sprepost, enr_destroy, enr_identity, steadystate, to_super
import qutip.parallel as par
from sympy.functions import coth
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy import pi
import time
from utils import *


def dimer_ham_RC(w_1, w_2, w_xx, V, mu, Omega_1,
                Omega_2, kap_1, kap_2, N_1, N_2, exc):
    """ Builds RC Hamiltonian in excitation restricted subspace

    Input: System splitting, RC freq., system-RC coupling
    and Hilbert space dimension Output: Hamiltonian, all
    collapse operators in the vibronic Hilbert space
    """
    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    SIGMA_1 = OX*XX.dag() + OO*XO.dag()
    SIGMA_2 = XO*XX.dag() + OO*OX.dag()
    assert SIGMA_1*OX == SIGMA_2*XO
    #I_RC_1 = qeye(N_1)
    #I_RC_2 = qeye(N_2)
    I_dim = qeye(4)
    I = enr_identity([N_1,N_2], exc)
    H_dim_sub = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag()
    H_dim_sub += V*(XO*OX.dag() + OX*XO.dag())
    print H_dim_sub
    H_dim = tensor(H_dim_sub, I)
    A_EM = tensor(SIGMA_1+mu*SIGMA_2, I)

    atemp = enr_destroy([N_1,N_2], exc)

    a_RC_exc = [tensor(I_dim, aa) for aa in atemp] # annhilation ops in exc restr basis
    A_1 = a_RC_exc[0].dag() + a_RC_exc[0]
    A_2 = a_RC_exc[1].dag() + a_RC_exc[1]
    H_I1 = kap_1*tensor(SIGMA_1.dag()*SIGMA_1, I)*A_1
    H_I2 = kap_2*tensor(SIGMA_2.dag()*SIGMA_2, I)*A_2

    H_RC1 = Omega_1*a_RC_exc[0].dag()*a_RC_exc[0]
    H_RC2 = Omega_2*a_RC_exc[1].dag()*a_RC_exc[1]

    H_S = H_dim + H_RC1 + H_RC2 + H_I1 + H_I2
    return H_S, A_1, A_2, A_EM

def operator_func(j, k, eVals=[], eVecs=[], A_1=[], A_2=[],
                    gamma_1=1., gamma_2=1., beta_1=0.4, beta_2=0.4):
    """ For parallelising the Liouvillian contruction
    """
    Chi_1, Xi_1, Chi_2, Xi_2 = 0, 0, 0, 0 # Initialise operators
    try:
        # eigenvalue difference, needs to be real for coth and hermiticity
        e_jk = (eVals[j] - eVals[k]).real
        # Overlap of collapse operator for site 1 with vibronic eigenbasis
        J, K = eVecs[j], eVecs[k]
        A_jk_1 = A_1.matrix_element(J.dag(), K)
        outer_eigen = J*K.dag()

        if sp.absolute(A_jk_1) > 0:
            if sp.absolute(e_jk) > 0 and sp.absolute(beta_1) > 0:
                Chi_1 = 0.5*np.pi*e_jk*gamma_1 * float(coth(e_jk * beta_1 / 2).evalf())*A_jk_1*outer_eigen
                Xi_1 = 0.5*np.pi*e_jk*gamma_1 * A_jk_1 * outer_eigen
            else:
                Chi_1 = np.pi*gamma_1*A_jk_1*outer_eigen/beta_1 # Just return coefficients which are left over
                #Xi += 0 #since J_RC goes to zero
        A_jk_2 = A_2.matrix_element(eVecs[j].dag(), eVecs[k])
        if sp.absolute(A_jk_2) > 0:
            if sp.absolute(e_jk) > 0 and sp.absolute(beta_2)>0:
                # e_jk*gamma is the spectral density
                Chi_2 = 0.5*np.pi*e_jk*gamma_2 * float(coth(e_jk * beta_2 / 2).evalf())*A_jk_2*outer_eigen
                Xi_2 = 0.5*np.pi*e_jk*gamma_2 * A_jk_2 * outer_eigen
            else:
                # If e_jk is zero, coth diverges but J goes to zero so limit taken seperately
                Chi_2 = np.pi*gamma_2*A_jk_2*outer_eigen/beta_2 # Just return coefficients which are left over
                #Xi += 0 #since J_RC goes to zero
    except e:
        print e
    #if type(Chi_1) != type(1):
    #    print Chi_1.dims
    return Qobj(Chi_1), Qobj(Xi_1), Qobj(Chi_2), Qobj(Xi_2)

def RCME_operators_par(H_0, A_1, A_2, gamma_1, gamma_2, beta_1, beta_2, num_cpus=0):
    dim_ham = H_0.shape[0]
    eVals, eVecs = H_0.eigenstates()
    names = ['eVals', 'eVecs', 'A_1', 'A_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2']
    kwargs = dict()
    for name in names:
        kwargs[name] = eval(name)
    l = dim_ham*range(dim_ham) # Perform two loops in one
    Chi_1, Xi_1, Chi_2, Xi_2 = par.parfor(operator_func,  sorted(l), l, num_cpus=num_cpus, **kwargs)
    return H_0, sum(Chi_1), sum(Xi_1), sum(Chi_2), sum(Xi_2)


def RCME_operators(H_0, A_1, A_2, gamma_1, gamma_2, beta_1, beta_2, num_cpus=0):
    # This function will be passed a TLS-RC hamiltonian, RC operator, spectral density and beta
    # outputs all of the operators needed for the RCME (underdamped)
    dim_ham = H_0.shape[0]
    Chi_1, Chi_2 = 0, 0 # Initiate the operators
    Xi_1, Xi_2 = 0, 0
    eVals, eVecs = H_0.eigenstates()
    #print H_0
    #EigenDiffs = []
    #ti = time.time()
    for j in range(dim_ham):
        for k in range(dim_ham):
            e_jk = eVals[j] - eVals[k] # eigenvalue difference
            #EigenDiffs.append(e_jk)
            A_jk_1 = A_1.matrix_element(eVecs[j].dag(), eVecs[k])
            outer_eigen = eVecs[j] * (eVecs[k].dag())
            if sp.absolute(A_jk_1) > 0:
                if sp.absolute(e_jk) > 0 and sp.absolute(beta_1) > 0:
                    Chi_1 += 0.5*np.pi*e_jk*gamma_1 * float(coth(e_jk * beta_1 / 2).evalf())*A_jk_1*outer_eigen
                    Xi_1 += 0.5*np.pi*e_jk*gamma_1 * A_jk_1 * outer_eigen
                else:
                    Chi_1 += np.pi*gamma_1*A_jk_1*outer_eigen/beta_1 # Just return coefficients which are left over
                    #Xi += 0 #since J_RC goes to zero
            A_jk_2 = A_2.matrix_element(eVecs[j].dag(), eVecs[k])
            if sp.absolute(A_jk_2) > 0:
                if sp.absolute(e_jk) > 0 and sp.absolute(beta_2)>0:
                    #print e_jk
                    # If e_jk is zero, coth diverges but J goes to zero so limit taken seperately
                    #print beta_2, e_jk
                    Chi_2 += 0.5*np.pi*e_jk*gamma_2 * float(coth(e_jk * beta_2 / 2).evalf())*A_jk_2*outer_eigen # e_jk*gamma is the spectral density
                    Xi_2 += 0.5*np.pi*e_jk*gamma_2 * A_jk_2 * outer_eigen
                else:
                    Chi_2 += np.pi*gamma_2*A_jk_2*outer_eigen/beta_2 # Just return coefficients which are left over
                    #Xi += 0 #since J_RC goes to zero
    return H_0, Chi_1, Xi_1, Chi_2, Xi_2

def liouvillian_build(H_0, A_1, A_2, gamma_1, gamma_2,  wRC_1, wRC_2, T_1, T_2, num_cpus=0):
    ti = time.time()
    conversion = 0.695
    beta_1 = 0. # We want to calculate beta for each reaction coordinate, but avoid divergences
    RCnb_1 = 0
    if T_1 == 0.0:
        beta_1 = np.infty # This is a hack but hopefully won't have to be used
        #RCnb_1 = 0
        print "Temperature is too low in RC 1, this won't work"
    else:
        beta_1 = 1./(conversion * T_1)
        #RCnb_1 = (1 / (sp.exp( beta_1 * wRC_1)-1))
    beta_2 = 0.
    RCnb_2 = 0
    if T_2 == 0.0:
        beta_2 = np.infty
        #RCnb_2 = 0
        print "Temperature is too low in RC 2, this won't work"
    else:
        beta_2 = 1./(conversion * T_2)
        #RCnb_2 = (1 / (sp.exp( beta_2 * wRC_2)-1))
    # Now this function has to construct the liouvillian so that it can be passed to mesolve
    H_1 = H_0
    H_0, Chi_1, Xi_1,Chi_2, Xi_2  = RCME_operators(H_0, A_1, A_2, gamma_1, gamma_2, beta_1, beta_2, num_cpus=num_cpus)
    '''
    print np.sum(Chi_1.full() == Chi_1_.full()), Chi_1.shape[0]**2
    print np.sum(Chi_2.full() == Chi_2_.full()), Chi_1.shape[0]**2
    print np.sum(Xi_1.full() == Xi_1_.full()), Chi_1.shape[0]**2
    print np.sum(Xi_2.full() == Xi_2_.full()), Chi_1.shape[0]**2
    print abs(Chi_1.full()) == abs(Chi_1_.full())'''
    L = 0
    #print beta_1, beta_2
    for A, Chi in zip([A_1, A_2],[Chi_1, Chi_2]):
        L=L-spre(A*Chi)
        L=L+sprepost(A, Chi)
        L=L+sprepost(Chi, A)
        L=L-spost(Chi*A)
    for A, Xi in zip([A_1, A_2],[Xi_1, Xi_2]):
        L=L+spre(A*Xi)
        L=L+sprepost(A, Xi)
        L=L-sprepost(Xi, A)
        L=L-spost(Xi*A)
    print "Building the RC Liouvillian took ", time.time()-ti, "seconds."
    return L

def construct_thermal(args):
    w_1, w_2, w_xx, V, mu = args['w_1'], args['w_2'], args['w_xx'], args['V'], args['mu']
    alpha_1, alpha_2, wc, T_EM = args['alpha_1'], args['alpha_2'], args['wc'], args['T_EM']
    gamma_1, gamma_2 = 2, 2
    wRC_1, wRC_2 = 2*pi*wc*gamma_1, 2*pi*wc*gamma_2
    kappa_1, kappa_2 = np.sqrt(pi*alpha_1*wRC_1/2.), np.sqrt(pi*alpha_2*wRC_2/2.)
    N_1, N_2, exc = args['N_1'], args['N_2'], args['exc']
    H_0, A_1, A_2, A_EM = dimer_ham_RC(w_1, w_2, w_xx, V, mu, wRC_1, wRC_2, kappa_1, kappa_2, N_1, N_2, exc)
    I = enr_identity([N_1,N_2], exc)
    D = (-H_0/(0.695*T_EM)).expm()
    return D/D.tr()

def RC_mapping_OD(args):

    # we define all of the RC parameters by the underdamped spectral density
    w_1, w_2, w_xx, V = args['w_1'], args['w_2'], args['w_xx'], args['V']
    T_1, T_2, mu = args['T_1'], args['T_2'], args['mu']
    wRC_1, wRC_2, alpha_1, alpha_2, wc = args['w0_1'], args['w0_2'], args['alpha_1'], args['alpha_2'], args['wc']
    N_1, N_2, exc = args['N_1'], args['N_2'], args['exc']
    gamma_1, gamma_2 = 2, 2
    wRC_1 = wRC_2 = 2*pi*wc*gamma_1
    kappa_1, kappa_2 = np.sqrt(pi*alpha_1*wRC_1/2.), np.sqrt(pi*alpha_2*wRC_2/2.)
    print "****************************************************************"
    H_0, A_1, A_2, A_EM = dimer_ham_RC(w_1, w_2, w_xx, V, mu, wRC_1, wRC_2, kappa_1, kappa_2, N_1, N_2, exc)
    L_RC =  liouvillian_build(H_0, A_1, A_2, gamma_1, gamma_2,  wRC_1, wRC_2, T_1, T_2, num_cpus=args['num_cpus'])
    full_size = (4*N_1*N_1)**2
    print "It is {}by{}. The full basis would be {}by{}".format(L_RC.shape[0], L_RC.shape[0], full_size, full_size)
    return L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2, kappa_1, kappa_2


def RC_mapping_UD(args):

    # we define all of the RC parameters by the underdamped spectral density
    w_1, w_2, w_xx, V = args['w_1'], args['w_2'], args['w_xx'], args['V']
    T_1, T_2, mu = args['T_1'], args['T_2'], args['mu']
    wRC_1, wRC_2, alpha_1, alpha_2, wc = args['w0_1'], args['w0_2'], args['alpha_1'], args['alpha_2'], args['wc']
    N_1, N_2, exc = args['N_1'], args['N_2'], args['exc']

    Gamma_1 = (wRC_1**2)/wc # This param doesn't exist for an overdamped spectral density, we set it to this instead
    gamma_1 = Gamma_1 / (2. * np.pi * wRC_1)  # no longer a free parameter that we normally use to fix wRC to the system splitting
    kappa_1 = np.sqrt(np.pi * alpha_1 * wRC_1 / 2.)  # coupling strength between the TLS and RC

    Gamma_2 = (wRC_2**2)/wc
    gamma_2 = Gamma_2 / (2. * np.pi * wRC_2)
    kappa_2 = np.sqrt(np.pi * alpha_2 * wRC_2 / 2.)
    print "****************************************************************"
    #print args
    H_0, A_1, A_2, A_EM = dimer_ham_RC(w_1, w_2, w_xx, V, mu, wRC_1, wRC_2, kappa_1, kappa_2, N_1, N_2, exc)
    L_RC =  liouvillian_build(H_0, A_1, A_2, gamma_1, gamma_2,  wRC_1, wRC_2, T_1, T_2, num_cpus=args['num_cpus'])
    full_size = (4*N_1*N_1)**2
    print "It is {}by{}. The full basis would be {}by{}".format(L_RC.shape[0], L_RC.shape[0], full_size, full_size)
    return L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2, kappa_1, kappa_2
    #H_dim_full = w_1*XO*XO.dag() + w_2*w_1*OX*OX.dag() + w_xx*XX*XX.dag() + V*((SIGMA_m1+SIGMA_m1.dag())*(SIGMA_m2+SIGMA_m2.dag()))





if __name__ == "__main__":
    ev_to_inv_cm = 8065.5
    w_1, w_2 = 1.4*ev_to_inv_cm, 1.*ev_to_inv_cm
    V = 200.
    w_xx = w_1+w_2+V
    T_1, T_2 = 300., 300.
    wRC_1, wRC_2 = 300., 300.
    alpha_1, alpha_2 = 100./np.pi, 100./np.pi
    wc =53.
    N_1, N_2= 5,5
    exc = int(N_1+N_2)
    mu=1

    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    SIGMA_m1 = OX*XX.dag() + OO*XO.dag()
    SIGMA_m2 = XO*XX.dag() + OO*OX.dag()
    SIGMA_x1 = SIGMA_m1+SIGMA_m1.dag()
    SIGMA_x2 = SIGMA_m2+SIGMA_m2.dag()

    L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2, kap_1, kap_2 = RC_mapping_UD(w_1, w_2, w_xx, V, T_1, T_2, wRC_1,
                                            wRC_2, alpha_1, alpha_2, wc, N_1, N_2,
                                            exc, mu=1, num_cpus=1) # test that it works
    H_S, A_1, A_2, A_EM = dimer_ham_RC_full(w_1, w_2, w_xx, V, mu, wRC_1, wRC_2, kap_1, kap_2, N_1, N_2)
    H_S_, A_1_, A_2_, A_EM_ = dimer_ham_RC(w_1, w_2, w_xx, V, mu, wRC_1, wRC_2, kap_1, kap_2, N_1, N_2, exc)

    for i,j in zip([H_S, A_1, A_2, A_EM], [H_S_, A_1_, A_2_, A_EM_]):
        print i.shape, j.shape, i.dims, j.dims
        print np.sum(i.full() == j.full()), i.shape[0]**2
    '''
    I = enr_identity([N_1,N_2], exc)
    try:
        ss = steadystate(to_super(H_0), [L_RC], method='eigen')
        #print "Ground state population should be 1 "
        #assert (ss*tensor(OO*OO.dag(), I)).tr().real == 1 # ptrace doesn't really work
    except ValueError, e:
        print "steady states error:", e
    print "dimer_phonons is finished. It is an {}by{} of type {}.".format(L_RC.shape[0],
                                                            L_RC.shape[0], L_RC.type)'''
