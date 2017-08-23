import sys
import os
from numpy import pi
import numpy as np

from qutip import Qobj, basis, ket, mesolve, qeye, tensor, thermal_dm, destroy, enr_identity, enr_destroy, enr_thermal_dm
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec

import dimer_phonons as RC
import dimer_optical as EM
from utils import *
import dimer_plotting as vis
import dimer_checking as check
matplotlib.style.use('ggplot')
reload(RC)
reload(EM)
reload(vis)
reload(check)



def get_dimer_info(rho):
    e1e2 = tensor(basis(4,1)*basis(4,2).dag(), I)
    e2e1 = tensor(basis(4,2)*basis(4,1).dag(), I)
    g = (rho*OO).tr()
    e1 = (rho*XO).tr()
    e2 = (rho*OX).tr()

    e1e2 = (rho*e1e2).tr()
    e2e1 = (rho*e2e1).tr()
    xx = (rho*XX).tr()
    return Qobj([[g.real, 0,0,0], [0, e1.real,e1e2.real,0],[0, e2e1.real,e2.real,0],[0, 0,0,xx.real]])#/(g+e1+e2+xx)



if __name__ == "__main__":

    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()
    """
    w_2 = 1.4*ev_to_inv_cm
    V = 92 #0.01*8065.5
    bias = 0.01*ev_to_inv_cm
    w_1 = w_2 + bias
    dipole_1, dipole_2 = 1., 1.
    T_EM = 6000. # Optical bath temperature
    alpha_EM = 0.3*inv_ps_to_inv_cm # Optical S-bath strength (from inv. ps to inv. cm)(larger than a real decay rate because dynamics are more efficient this way)
    mu = w_2*dipole_2/(w_1*dipole_1)

    T_1, T_2 = 300., 300. # Phonon bath temperature

    wc = 1*53. # Ind.-Boson frame phonon cutoff freq
    w0_2, w0_1 = 1000., 1000. # underdamped SD parameter omega_0
    w_xx = w_2 + w_1

    """
    w_2 = 1500
    V = 10 #0.01*8065.5
    bias = 250 #1*V #0.01*ev_to_inv_cm
    w_1 = w_2 + bias
    dipole_1, dipole_2 = 1., 1.
    T_EM = 5700. # Optical bath temperature
    alpha_EM = 0.1 #0.3*inv_ps_to_inv_cm # Optical S-bath strength (from inv. ps to inv. cm)(larger than a real decay rate because dynamics are more efficient this way)
    mu = w_2*dipole_2/(w_1*dipole_1)

    T_1, T_2 = 300., 300. # Phonon bath temperature

    wc = 1*53. # Ind.-Boson frame phonon cutoff freq
    w0_2, w0_1 = 3000., 3000. # underdamped SD parameter omega_0
    w_xx = w_2 + w_1

    alpha_1, alpha_2 = 0.5/pi, 0.5/pi # Ind.-Boson frame coupling
    N_1, N_2 = 3,3 # set Hilbert space sizes
    exc = 3
    num_cpus = 4
    J = J_minimal

    H_dim = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag() + V*(XO*OX.dag() + OX*XO.dag())
    num_energies, num_states = H_dim.eigenstates()
    PARAM_names = ['w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2', 'wc',
                    'w0_1', 'w0_2', 'alpha_1', 'alpha_2', 'N_1', 'N_2', 'exc', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J']
    PARAMS = dict((name, eval(name)) for name in PARAM_names)

    I_dimer = qeye(4)
    I = enr_identity([N_1,N_2], exc)
    atemp = enr_destroy([N_1,N_2], exc)
    n_RC_1 = Occupation(w0_1, T_1)
    n_RC_2 = Occupation(w0_2, T_2)

    phonon_num_1 = atemp[0].dag()*atemp[0]
    phonon_num_2 = atemp[1].dag()*atemp[1]
    x_1 = (atemp[0].dag()+atemp[0])
    x_2 = (atemp[1].dag()+atemp[1])

    #initial_sys = OO*OO.dag()
    #initial_sys = 0.5*(XO+OX)*(XO+OX).dag()
    '''Defining DM states'''
    site_coherence = tensor(OX*XO.dag(), I)
    OO = tensor(OO*OO.dag(), I)
    XO = tensor(XO*XO.dag(), I)
    OX = tensor(OX*OX.dag(), I)
    XX = tensor(XX*XX.dag(), I)

    eVals, eVecs = H_dim.eigenstates()
    eVals, eVecs = zip(*sorted(zip(eVals, eVecs))) # sort them
    dark_old= eVecs[1]
    bright_old= eVecs[2]
    energies, states = check.exciton_states(PARAMS)
    lam_p = 0.5*(w_1+w_2)+0.5*np.sqrt((w_2-w_1)**2+4*(V**2))
    lam_m = 0.5*(w_1+w_2)-0.5*np.sqrt((w_2-w_1)**2+4*(V**2))
    bright_vec = states[1]
    dark_vec = states[0]
    dark = tensor(dark_vec*dark_vec.dag(), I)
    bright = tensor(bright_vec*bright_vec.dag(), I)
    #print  (states[1]*states[1].dag()).tr(), bright_old, states[1]*states[1].dag()
    #print (states[0]*states[0].dag()).tr(), dark_old, states[0]*states[0].dag()
    exciton_coherence = tensor(dark_vec*bright_vec.dag(), I)
    Phonon_1 = tensor(I_dimer, phonon_num_1)
    Phonon_2 = tensor(I_dimer, phonon_num_2)
    disp_1 = tensor(I_dimer, x_1)
    disp_2 = tensor(I_dimer, x_2)

    #rho_0 = tensor(initial_sys, enr_thermal_dm([N_1,N_2], exc, [n_RC_1, n_RC_2]))

    #rho_0 = rho_0/rho_0.tr()
    ops = [OO, XO, OX, XX, site_coherence]
    # Expectation values and time increments needed to calculate the dynamics
    expects = ops + [dark, bright, exciton_coherence]
    expects +=[Phonon_1, Phonon_2, disp_1, disp_2]
    #Now we build all of the mapped operators and RC Liouvillian.

    # electromagnetic bath liouvillians

    #print sys.getsizeof(L_ns)
    opts = qt.Options(num_cpus=num_cpus, store_states=True)
    ncolors = len(plt.rcParams['axes.prop_cycle'])

    thermal_RCs = enr_thermal_dm([N_1,N_2], exc, [n_RC_1, n_RC_2])
    rho_0 = tensor(basis(4,0)*basis(4,0).dag(),thermal_RCs)
    #timelist = np.linspace(0,3,1000)
    L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2, kappa_1, kappa_2 = RC.RC_mapping_OD(PARAMS)
    timelist = (0,1,5000)
    #Gamma_1 = (wRC_1**2)/wc
    #plt.plot(freqs, J_overdamped(freqs, alpha_1, wc), label='OD')
    #plt.plot(freqs, J_underdamped(freqs, alpha_1, Gamma_1, wRC_1), label='UD')
    #plt.plot(freqs, J_OD_to_UD(freqs, 2, wRC_1, kappa_1), label='setting gamma=2')
    plt.legend()
    ss_J, DATA_J = vis.calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, timelist, EM_approx='j', l='flat_')
    ss_P, DATA_P = vis.calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, timelist, EM_approx='p')
    ss_S, DATA_S = vis.calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, timelist, EM_approx='s')
    ss_NS, DATA_NS = vis.calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, timelist, EM_approx='ns')
    plt.plot(DATA_J.expect[7])
    #PARAMS.update({'exc': 5})
    #PARAMS.update({'N_1': 5, 'N_2': 5, 'alpha_1':50/pi, 'alpha_2': 50/pi})
    #PARAMS.update({'N_2': 6, 'N_1': 6})

    PARAMS.update({'J': J_flat})
    #DATA_J = vis.calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, EM_approx='j', l='flat_')
    #DATA_P = vis.calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, EM_approx='p', l='flat_')
    #DATA_S = vis.calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, EM_approx='s', l='flat_')
    #DATA_NS = vis.calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, EM_approx='ns', l='flat_')
    """
    mut_inf_d1 = []
    mut_inf_d2 = []
    mut_inf_12 = []
    for i in DATA_ns.states:
        dimer_MI = dimer_mutual_information(i, PARAMS)
        mut_inf_d1.append(dimer_MI[1])
        mut_inf_d2.append(dimer_MI[2])
        mut_inf_12.append(dimer_MI[0])
    plt.plot(timelist, mut_inf_12, label='b1 & b2')
    plt.plot(timelist, mut_inf_d2, label='dimer & b2')
    plt.plot(timelist, mut_inf_d1, label='dimer & b1')
    plt.legend()
    plt.show()
    #fig = plt.figure()
    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    """

    #ax1.plot(timelist, DATA_ns.expect[7].real, label='real eig. coherence')
    #ax1.plot(timelist, DATA_ns.expect[7].imag, label='imag. eig. coherence')
    #ax2.plot(timelist, DATA_ns.expect[8]-DATA_ns.expect[9], label='Phonon occupation diff.')
    #ax2.plot(timelist, DATA_ns.expect[1], label='site 1', linestyle="--")
    #ax2.plot(timelist, DATA_ns.expect[2], label='site 2', linestyle="--")
    #ax2.plot(timelist, DATA_ns.expect[0]+ DATA_ns.expect[5]+ DATA_ns.expect[6]+ DATA_ns.expect[3], label='nsb')
    #ax1.plot(timelist, DATA_s.expect[7].real, linestyle="--",label='secular')
    #ax1.plot(timelist, DATA_s.expect[7].imag, linestyle="--", label='pbc imag')
    #ax2.plot(timelist, DATA_ns.expect[5], linestyle="-",label='dark', color ='r')
    #ax2.plot(timelist, DATA_ns.expect[6], linestyle="-", label='bright', color ='b')
    #ax2.plot(timelist, DATA_ns.expect[3], linestyle="-", label='exc')
    #ax2.plot(timelist, DATA_s.expect[5], linestyle="-",label='p dark')
    #ax2.plot(timelist, DATA_s.expect[6], linestyle="-", label='p bright')
    #method= 'direct' #'iterative-lgmres'

    """
    plt.figure()
    plt.plot(timelist, DATA_ns.expect[5], linestyle="-",label='dark', color ='r')
    plt.plot(timelist, DATA_ns.expect[6], linestyle="-", label='bright', color ='b')
    plt.ylabel('Population')
    plt.legend()

    #ss_ns = qt.steadystate(H_0, [p*L_RC+L_ns], method= method, use_precond=True)
    #exc_coh = (ss_ns*expects[7]).tr()
    #ax1.axhline(exc_coh.real, linestyle = '--')
    ax1.legend()
    #d = (ss_ns*expects[5]).tr()
    #print d
    #ax2.axhline(d.real, linestyle = '--', label='dark ss', color ='b')
    #d = (ss_ns*expects[6]).tr()
    #print d
    #ax2.axhline(d.real, linestyle = '--', label='bright ss', color ='r')
    ax2.legend()
    plt.show()
    """
    """
    p = 1
    if (alpha_1 == 0 and alpha_2 == 0):
        p = 0

    method= 'direct' #'iterative-lgmres'
    ss_ns = qt.steadystate(H_0, [p*L_RC+L_ns], method= method, use_precond=True)

    #print sum((ss-ss_pred).diag())
    print "Steady state is ", check.get_dimer_info(ss_ns)
    print "Exciton coherence is ", (ss_ns*exciton_coherence).tr()
    print "Dark population is ", (ss_ns*dark).tr()
    print "Bright population is ", (ss_ns*bright).tr()
    #ss_pred = ((-1/T_EM*0.695)*H_0).expm()
    #ss_pred = ss_pred/ss_pred.tr()

    rho_T = Qobj((-1/(T_EM*0.695))*H_0).expm()
    rho_T = check.get_dimer_info(rho_T/rho_T.tr())
    print "Thermal state is :", rho_T
    rho_0 = tensor(rho_T,enr_thermal_dm([N_1,N_2], exc, [n_RC_1, n_RC_2]))

    timelist = np.linspace(0,100,2000)*0.188
    DATA_ns = mesolve(H_0, rho_0, timelist, [p*L_RC+L_ns], expects, options=opts, progress_bar=True)
    ss_dyn = check.ss_from_dynamics(DATA_ns)
    print "SS from dynamics = ", ss_dyn
    print "Exciton coherence is ", (ss_dyn*dark_vec*bright_vec.dag()).tr()
    print "Dark population is ", (ss_dyn*dark_vec*dark_vec.dag()).tr()
    print "Bright population is ", (ss_dyn*bright_vec*bright_vec.dag()).tr()
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    vis.plot_eig_dynamics(DATA_ns, timelist, expects, ax, title='Non-secular driving\n')
    plt.show()
    #print ss_pred.ptrace(0)

    #check.steadystate_comparison(H_0, [L_RC+L_ns], bright)
    """
    """
    L_p = EM.L_phenom(states, energies, I, PARAMS)
    try:
        ss_p = qt.steadystate(H_0, [L_RC+L_p], method= 'iterative-lgmres', use_precond=True)
    except:
        ss_p = qt.steadystate(H_0, [L_RC+L_p], method= 'iterative-lgmres')
    #print sum((ss-ss_pred).diag())
    print "DM is ", check.get_dimer_info(ss_p)

    print "Exciton coherence is ", (ss_p*exciton_coherence).tr()
    print "Dark population is ", (ss_p*dark).tr()
    print "Bright population is ", (ss_p*bright).tr()
    #print "Steady state is ", qt.steadystate(H_0)
    calculate_dynamics()

    #try:
    #     #np.arange(60, 420, 40)/pi
    #PARAMS.update({'w_1':w_2})
    #observable = exciton_coherence
    #check.get_coh_ops(PARAMS, biases, I)
    #
    """
    """
    alpha_ph = np.array([0.1, 1., 10., 100., 500.])/pi
    #alpha_ph=np.array([0])
    biases = np.linspace(0, 0.03, 50)*ev_to_inv_cm
    #biases = np.array([0, 0.01*ev_to_inv_cm])

    PARAMS.update({'N_1':2, 'N_2':2, 'exc': 3})
    PARAMS.update({'V':0.25*92.})
    '''
    for alpha in alpha_ph:
        PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
        check.bias_dependence(biases, PARAMS, ops)
    vis.steadystate_ground_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_coherence_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_dark_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_bright_plot(PARAMS, alpha_ph, biases)'''
    #vis.steadystate_darkbright_plot(PARAMS, alpha_ph, biases)

    PARAMS.update({'V':0.5*92.})
    '''
    for alpha in alpha_ph:
        PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
        check.bias_dependence(biases, PARAMS, I, ops)
    vis.steadystate_ground_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_coherence_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_dark_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_bright_plot(PARAMS, alpha_ph, biases)'''
    #vis.steadystate_darkbright_plot(PARAMS, alpha_ph, biases)
    PARAMS.update({'V':1*92.})
    '''
    for alpha in alpha_ph:
        PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
        check.bias_dependence(biases, PARAMS, I, ops)
    vis.steadystate_ground_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_coherence_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_dark_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_bright_plot(PARAMS, alpha_ph, biases)'''
    vis.steadystate_darkbright_plot(PARAMS, alpha_ph, biases)

    PARAMS.update({'V':2*92.})
    '''
    for alpha in alpha_ph:
        PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
        check.bias_dependence(biases, PARAMS, I, ops)
    vis.steadystate_ground_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_coherence_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_dark_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_bright_plot(PARAMS, alpha_ph, biases)'''
    vis.steadystate_darkbright_plot(PARAMS, alpha_ph, biases)

    PARAMS.update({'N_1':4, 'N_2':4, 'exc': 5})

    I = enr_identity([4,4], 5)
    PARAMS.update({'V':0.25*92.})
    '''
    for alpha in alpha_ph:
        PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
        check.bias_dependence(biases, PARAMS, I, ops)
    vis.steadystate_ground_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_coherence_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_dark_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_bright_plot(PARAMS, alpha_ph, biases)'''
    vis.steadystate_darkbright_plot(PARAMS, alpha_ph, biases)

    PARAMS.update({'V':0.5*92.})
    '''
    for alpha in alpha_ph:
        PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
        check.bias_dependence(biases, PARAMS, I, ops)
    vis.steadystate_ground_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_coherence_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_dark_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_bright_plot(PARAMS, alpha_ph, biases)'''
    vis.steadystate_darkbright_plot(PARAMS, alpha_ph, biases)
    PARAMS.update({'V':1*92.})
    '''
    for alpha in alpha_ph:
        PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
        check.bias_dependence(biases, PARAMS, I, ops)
    vis.steadystate_ground_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_coherence_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_dark_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_bright_plot(PARAMS, alpha_ph, biases)'''
    vis.steadystate_darkbright_plot(PARAMS, alpha_ph, biases)

    PARAMS.update({'V':2*92.})
    '''
    for alpha in alpha_ph:
        PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
        check.bias_dependence(biases, PARAMS, I, ops)
    vis.steadystate_ground_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_coherence_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_dark_plot(PARAMS, alpha_ph, biases)
    vis.steadystate_bright_plot(PARAMS, alpha_ph, biases)'''
    vis.steadystate_darkbright_plot(PARAMS, alpha_ph, biases)
    """
    #del L_ns
    #L_s = EM.L_secular(H_0, A_EM, eps, alpha_EM, T_EM, J, num_cpus=num_cpus)
    #L_naive = EM_lind.electronic_lindblad(w_xx, w_1, eps, V, mu, alpha_EM, T_EM, N_1, N_2, exc)
    # Set up the initial density matrix
     # you need lots of points so that coherences are well defined -> spectra
    #nonsec_check(eps, H, A_em, N) # Plots a scatter graph representation of non-secularity. Could use nrwa instead.
    #fig = plt.figure(figsize=(12, 6))
    #ax1 = fig.add_subplot(111)
    #energies = plot_manifolds(ax1, H_0)


    # Calculate dynamics

    #DATA_s = mesolve(H_0, rho_0, timelist, [L_RC+L_s], expects, progress_bar=True)
    #DATA_naive = mesolve(H_0, rho_0, timelist, [L_RC+L_naive], expects, progress_bar=True)
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #vis.plot_RC_pop(DATA_ns, timelist, ax, title='Non-secular driving\n')

    #    fig = plt.figure(figsize=(12, 6))
    #    ax1 = fig.add_subplot(121)
    #    ax2 = fig.add_subplot(122)
    #    plot_RC_pop(DATA_ns, ax1)
    #    plot_RC_disp(DATA_ns, ax2)


    #SS, nvals = check.SS_convergence_check(eps, T_EM, T_ph, wc, w0, alpha_ph, alpha_EM, start_n=10)
    #plt.plot(nvals, SS)
    #plot_dynamics_spec(DATA_s, timelist)

    #np.savetxt('DATA/Dynamics/dimer_DATA_ns.txt', np.array([1- DATA_ns.expect[0], timelist]), delimiter = ',', newline= '\n')

    #plt.show()x
