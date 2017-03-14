from qutip import basis, ket, mesolve, qeye, tensor, thermal_dm, destroy, steadystate, spost, spre, sprepost, enr_destroy, enr_identity, steadystate
import qutip.parallel as par
from sympy.functions import coth
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import time

def dimer_ham_RC(w_1, w_2, w_xx, V, mu, Omega_1, Omega_2, kap_1, kap_2, N_1, N_2, exc):
    """
    Input: System splitting, RC freq., system-RC coupling and Hilbert space dimension
    Output: Hamiltonian, sigma_- and sigma_z in the vibronic Hilbert space
    """
    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_1 = OX*XX.dag() + OO*XO.dag()
    sigma_2 = XO*XX.dag() + OO*OX.dag()
    assert sigma_1*OX == sigma_2*XO
    #I_RC_1 = qeye(N_1)
    #I_RC_2 = qeye(N_2)
    I_dim = qeye(4)
    I = enr_identity([N_1,N_2], exc)
    H_dim = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag() + V*(XO*OX.dag() + OX*XO.dag())
    print H_dim
    H_dim = tensor(H_dim, I)
    A_EM = tensor(sigma_1+mu*sigma_2, I)

    atemp = enr_destroy([N_1,N_2], exc)

    A = [tensor(I_dim, aa) for aa in atemp]

    H_I1 = kap_1*tensor(sigma_1.dag()*sigma_1, I)*A[0]
    H_I2 = kap_2*tensor(sigma_2.dag()*sigma_2, I)*A[1]

    H_RC1 = Omega_1*A[0].dag()*A[0]
    H_RC2 = Omega_2*A[1].dag()*A[1]

    H_S = H_dim + H_RC1 + H_RC2 + H_I1 + H_I2

    return H_S, A[0], A[1], A_EM

def operator_func(index_2tuples, eVals=[], eVecs=[], A_1=[], A_2=[], gamma_1=1., gamma_2=1., beta_1=0.4, beta_2=0.4):
    Chi_1, Xi_1, Chi_2, Xi_2 = 0, 0, 0, 0
    for j,k in index_2tuples:
        try:
            e_jk = (eVals[j] - eVals[k]).real # eigenvalue difference, needs to be real for coth and hermiticity
            A_jk_1 = A_1.matrix_element(eVecs[j].dag(), eVecs[k])
            outer_eigen = eVecs[j] * (eVecs[k].dag())
            pref = 0
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
                    Chi_2 += 0.5*np.pi*e_jk*gamma_2 * float(coth(e_jk * beta_2 / 2).evalf())*A_jk_2*outer_eigen # e_jk*gamma is the spectral density
                    Xi_2 += 0.5*np.pi*e_jk*gamma_2 * A_jk_2 * outer_eigen
                else:
                    Chi_2 += np.pi*gamma_2*A_jk_2*outer_eigen/beta_2 # Just return coefficients which are left over
                    #Xi += 0 #since J_RC goes to zero
        except TypeError, e:
            print eVals[j] - eVals[k], j,k, e
    return Chi_1, Xi_1, Chi_2, Xi_2

def RCME_operators_par(H_0, A_1, A_2, gamma_1, gamma_2, beta_1, beta_2):
    dim_ham = H_0.shape[0]
    eVals, eVecs = H_0.eigenstates()
    names = ['eVals', 'eVecs', 'A_1', 'A_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2']
    kwargs = dict()
    for name in names:
        kwargs[name] = eval(name)
    l = dim_ham*range(dim_ham) # Perform two loops in one
    index_2tuples = zip(sorted(l), l)
    Chi_1, Xi_1, Chi_2, Xi_2 = par.parfor(operator_func, [index_2tuples],num_cpus=1, **kwargs)
    return H_0, Chi_1[0], Xi_1[0], Chi_2[0], Xi_2[0]


def RCME_operators(H_0, A_1, A_2, gamma_1, gamma_2, beta_1, beta_2):
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

def liouvillian_build(H_0, A_1, A_2, gamma_1, gamma_2,  wRC_1, wRC_2, T_1, T_2):
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
    H_0, Chi_1, Xi_1,Chi_2, Xi_2  = RCME_operators_par(H_0, A_1, A_2, gamma_1, gamma_2, beta_1, beta_2)
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
    print "Building the RC Liouvillian took ", time.time()-ti, " seconds."
    return L

def RC_mapping_UD(w_1, w_2, w_xx, V, T_1, T_2, wRC_1, wRC_2, alpha_1, alpha_2, wc,  N_1, N_2, exc, mu=1):
    # we define all of the RC parameters by the underdamped spectral density
    Gamma_1 = (wRC_1**2)/wc
    gamma_1 = Gamma_1 / (2. * np.pi * wRC_1)  # no longer a free parameter that we normally use to fix wRC to the system splitting
    kappa_1 = np.sqrt(np.pi * alpha_1 * wRC_1 / 2.)  # coupling strength between the TLS and RC

    Gamma_2 = (wRC_2**2)/wc
    gamma_2 = Gamma_2 / (2. * np.pi * wRC_2)
    kappa_2 = np.sqrt(np.pi * alpha_2 * wRC_2 / 2.)

    print 'Gamma_1 == Gamma_2', Gamma_1 == Gamma_2
    print 'gamma_1 == gamma_2', gamma_1 == gamma_2
    print 'kappa_1 == kappa_2', kappa_1 == kappa_2
    print 'alpha_1 == alpha_2', alpha_1 == alpha_2
    print 'wRC_1 == wRC_2', wRC_1 == wRC_2
    print 'T_1 == T_2', T_1 == T_2
    print 'N_2 == N_1', N_2 == N_1
    print "splitting ={}, coupling SB cutoff={}\n RC1 oscillator frequency={}, RC2 oscillator frequency={} \n  gamma={}, kappa={}, N={}".format(w_1-w_2, wc, wRC_1, wRC_2, gamma_1, kappa_1, N_1)
    H_0, A_1, A_2, A_EM = dimer_ham_RC(w_1, w_2, w_xx, V, mu, wRC_1, wRC_2, kappa_1, kappa_2, N_1, N_2, exc)
    L_RC =  liouvillian_build(H_0, A_1, A_2, gamma_1, gamma_2,  wRC_1, wRC_2, T_1, T_2)
    return L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2
    #H_dim_full = w_1*XO*XO.dag() + w_2*w_1*OX*OX.dag() + w_xx*XX*XX.dag() + V*((sigma_m1+sigma_m1.dag())*(sigma_m2+sigma_m2.dag()))





if __name__ == "__main__":
    ev_to_inv_cm = 8065.5
    w_1, w_2 = 1.4*ev_to_inv_cm, 1.*ev_to_inv_cm
    V = 200.
    w_xx = w_1+w_2+V
    T_1, T_2 = 300., 300.
    wRC_1, wRC_2 = 300., 300.
    alpha_1, alpha_2 = 100./np.pi, 100./np.pi
    wc =53.
    N_1, N_2= 6,6
    exc = 9
    mu=1

    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()

    L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2 = RC_mapping_UD(w_1, w_2, w_xx, V, T_1, T_2, wRC_1, wRC_2, alpha_1, alpha_2, wc,  N_1, N_2, exc, mu=1) # test that it works
    ss = steadystate(H_0, [L_RC])
    print ss.ptrace(0)
    print "dimer_UD_liouv is finished. It is an {}by{} of type {}.".format(L_RC.shape[0], L_RC.shape[0], L_RC.type)
