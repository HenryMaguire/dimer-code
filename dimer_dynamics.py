import sys
from numpy import pi

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





if __name__ == "__main__":

    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()

    w_1 = 1.1*8065.5
    w_2 = 1.1*8065.5
    V = 92. #0.1*8065.5
    eps = (w_1+w_2)*0.5

    T_EM = 6000. # Optical bath temperature
    alpha_EM = 1.*5.309 # Optical S-bath strength (from inv. ps to inv. cm)(optical)
    mu = 1.

    T_1, T_2 = 300., 300. # Phonon bath temperature

    wc = 53. # Ind.-Boson frame phonon cutoff freq
    w0_2, w0_1 = 300., 300. # underdamped SD parameter omega_0
    w_xx = w_2 + w_1 + V
    alpha_1, alpha_2 = 400/pi, 400/pi # Ind.-Boson frame coupling
    N_1, N_2 = 5,5  # set Hilbert space sizes
    exc = int((N_1+N_2)*0.5)
    num_cpus = 2
    J = J_minimal

    H_dim = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag() + V*(XO*OX.dag() + OX*XO.dag())
    PARAM_names = ['w_1', 'w_2', 'V', 'eps', 'w_xx', 'T_1', 'T_2', 'wc',
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

    initial_sys = OO*OO.dag()
    #initial_sys = 0.5*(XO+OX)*(XO+OX).dag()

    OO = tensor(OO, I)
    XO = tensor(XO, I)
    OX = tensor(OX, I)
    XX = tensor(XX, I)
    eVals, eVecs = H_dim.eigenstates()
    eVals, eVecs = zip(*sorted(zip(eVals, eVecs))) # sort them
    dark = tensor(eVecs[1]*eVecs[1].dag(), I)
    bright = tensor(eVecs[2]*eVecs[2].dag(), I)
    exciton_coherence = tensor(eVecs[1]*eVecs[2].dag(), I)
    Phonon_1 = tensor(I_dimer, phonon_num_1)
    Phonon_2 = tensor(I_dimer, phonon_num_2)
    disp_1 = tensor(I_dimer, x_1)
    disp_2 = tensor(I_dimer, x_2)

    rho_0 = tensor(initial_sys, enr_thermal_dm([N_1,N_2], exc, [n_RC_1, n_RC_2]))
    #rho_0 = rho_0/rho_0.tr()


    site_coherence = OX*XO.dag()
    # Expectation values and time increments needed to calculate the dynamics
    expects = [OO*OO.dag(), XO*XO.dag(), OX*OX.dag(), XX*XX.dag()]
    expects +=[dark, bright, exciton_coherence]
    expects +=[Phonon_1, Phonon_2, disp_1, disp_2]

    timelist = np.linspace(0,10.0,10000)*0.188

    #Now we build all of the mapped operators and RC Liouvillian.



    # electromagnetic bath liouvillians

    #print sys.getsizeof(L_ns)
    opts = qt.Options(num_cpus=num_cpus)
    ncolors = len(plt.rcParams['axes.prop_cycle'])
    #fig = plt.figure(figsize=(12,6))
    alpha_ph = [50/pi]#, 100/pi, 200/pi, 400/pi, 700/pi]

    #L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2, kappa_1, kappa_2 = RC.RC_mapping_UD(w_1, w_2, w_xx,
    #                                    V, T_1, T_2, w0_1, w0_2, alpha_1, alpha_2,
    #                                    wc,  N_1, N_2, exc, mu=mu, num_cpus=num_cpus)
    """
    try:
        biases = np.linspace(0, 1000, 35)
        data_list = []
        global DATA_ns
        observable = Phonon_1
        for alpha in alpha_ph[::]:
            PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
            check.bias_dependence(biases, PARAMS)
            print "WE just finished pi*alpha={}".format(int(alpha*pi))
    except Exception as err:
        print "data not calculated fully because", err
    """
    """
    try:
        alpha_ph = [50/pi, 100/pi, 200/pi, 400/pi, 700/pi]
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        #colors = iter(['C1', 'C2', 'C3', 'C4', 'C5', 'c6', 'c7', 'c8'])
        data_list = []
        for i, color in enumerate(plt.rcParams['axes.prop_cycle'][0:len(alpha_ph)]):
            biases = np.linspace(0, 1000, 35)
            vis.plot_bias_dependence(ax1, exciton_coherence, biases, alpha_ph[i], color['color'], linestyle='-', xy_flip=False, y_label='Steady State Exciton Coherence Population')
            #data_list.append(ssdata_for_alpha)
        print "bias and coupling strength data seems to have been plotted"
    except Exception as err:
        print "data not plotted fully because", err"""

    try:
        alpha_ph = [50/pi, 100/pi, 200/pi, 400/pi, 700/pi]
        fig = plt.figure(1)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        #colors = iter(['C1', 'C2', 'C3', 'C4', 'C5', 'c6', 'c7', 'c8'])
        for i, color in enumerate(plt.rcParams['axes.prop_cycle'][0:len(alpha_ph)]):
            biases = np.linspace(0, 1000, 35)
            col = color['color']
            coh = vis.plot_bias_dependence(ax1, exciton_coherence, biases, alpha_ph[i], col, linestyle='-', x_label='Steady State Exciton Coherence Population')
            p1 = vis.plot_bias_dependence(ax2, Phonon_1, biases, alpha_ph[i], col, linestyle='-', y_label=False)
            p2 = vis.plot_bias_dependence(ax2, Phonon_2, biases, alpha_ph[i], col, x_label='Steady State RC Population', linestyle='-', legend_on=False, y_label=False)
            ax1.set_xlim(-0.09,0)

            ax2.set_xlim(0.3,0.5)
            noneq = abs(p1-p2)
            #ax2.plot(noneq, biases, color=col)
            max_idx = list(noneq).index(np.max(noneq))
            bias_at_max, coh_at_bias = biases[max_idx], coh[max_idx].real
            ax1.plot([coh_at_bias, 0], [bias_at_max,bias_at_max], color=col, linestyle='--')
            ax2.plot([0.3, p1[max_idx].real], [bias_at_max,bias_at_max], color=col, linestyle='--')
            plt.setp(ax2.get_yticklabels(), visible=False)
            #data_list.append(ssdata_for_alpha)
            fig.subplots_adjust(wspace=0.03)
        print "bias and coupling strength data seems to have been plotted"
    except Exception as err:
        print "data not plotted fully because", err
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #observable = exciton_coherence
        #alpha = 200/pi
        #ss_values = check.bias_dependence(biases, PARAMS, observable)

    """
    try:

        L_ns = EM.L_nonsecular(H_0, A_EM, eps, alpha_EM, T_EM, J, num_cpus=num_cpus)
        L_full = L_RC+L_ns

        DATA_ns = mesolve(H_0, rho_0, timelist, [L_full], expects, options=opts,
                                                            progress_bar=True)
        ss_dm = 0
        try:
            ss_dm = qt.steadystate(H_0, [L_full])
        except Exception as err:
            print "Warning: steady state density matrix didn't converge. Probably"
            print "\t due to some problem with excitation restriction. \n"
            print err
        timelist=timelist/0.188 # Convert from cm to picoseconds
        #DATA_ns = load_obj("DATA_N7_exc8")
        #fig = plt.figure(figsize=(12,6))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        title = 'Eigenstate population'
        #title = title + r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0_1, alpha_1, T_EM)
        fig = plt.figure()
        vis.plot_eig_dynamics(DATA_ns, timelist, expects, ax1, ss_dm=ss_dm)
        ax2 = fig.add_subplot(111)
        vis.plot_coherences(DATA_ns, timelist, expects, ax2, ss_dm=ss_dm)
        print 'Plotting worked!'
    except Exception as err:
        print "Could not get non-secular-driving dynamics because ",err
        """
    try:
        """
        L_s = EM.L_secular(H_0, A_EM, eps, alpha_EM, T_EM, J, num_cpus=num_cpus)
        DATA_s = mesolve(H_0, rho_0, timelist, [L_RC+L_ns], expects, options=opts,
                                                            progress_bar=True)
        ax = fig.add_subplot(212)
        vis.plot_dynamics(DATA_s, timelist, ax, title='Non-secular driving\n')"""
        print 'Secular dynamics skipped'
    except e:
        print "Could not get secular-driving dynamics because ",e


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
    '''
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plot_RC_pop(DATA_ns, ax1)
        plot_RC_disp(DATA_ns, ax2)
    '''

    #SS, nvals = check.SS_convergence_check(eps, T_EM, T_ph, wc, w0, alpha_ph, alpha_EM, start_n=10)
    #plt.plot(nvals, SS)
    #plot_dynamics_spec(DATA_s, timelist)

    #np.savetxt('DATA/Dynamics/dimer_DATA_ns.txt', np.array([1- DATA_ns.expect[0], timelist]), delimiter = ',', newline= '\n')

    plt.show()
