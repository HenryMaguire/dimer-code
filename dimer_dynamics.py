import sys
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


def calculate_dynamics():
    assert PARAMS['w_1'] != PARAMS['w_2']
    try:
        timelist = np.linspace(0,20.0,5000)*0.188
        L_ns = EM.L_nonsecular(H_0, A_EM, PARAMS)
        L_full = L_RC+L_ns

        DATA_ns = mesolve(H_0, rho_0, timelist, [L_full], expects, options=opts, progress_bar=True)
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
        plt.savefig("Notes/dynamics.png")
        print (ss_dm*exciton_coherence).tr()
        print 'Plotting worked!'
        return L_full
    except Exception as err:
        print "Could not get non-secular-driving dynamics because ",err

def steadystate_coherence_plot(args, alpha_list, biases):
    coh_ops = load_obj('DATA/zoomed_coherence_ops_N{}_wRC{}_V{}'.format(args['N_1'], int(args['w0_1']), int(args['V'])))
    fig = plt.figure(figsize=(12,6))
    print len(coh_ops)
    ax = fig.add_subplot(111)
    max_coh_for_alpha = []
    bias_at_max_list = []
    for alpha in alpha_list:
        ss_dms = load_obj('DATA/zoomed_bias_dependence_alpha{}_wRC{}_N{}_V{}'.format(int(alpha),int(args['w0_1']), args['N_1'], int(args['V'])))
        assert len(ss_dms) == len(coh_ops)
        coh_list = []
        for i in range(len(ss_dms)):
            ss_obs = (ss_dms[i]*coh_ops[i]).tr()
            coh_list.append(ss_obs)
        ax.plot(biases, np.array(coh_list).real, label=int(alpha))
        #real_pos = abs(np.array(coh_list).real)
        #max_coh = max(real_pos)
        #max_coh_for_alpha.append(-1*max_coh)
        #bias_at_max = biases[list(real_pos).index(max_coh)]
        #bias_at_max_list.append(bias_at_max)
    ax.set_xlabel(r'Bias $cm^{-1}$')
    ax.set_ylabel('Exciton Coherence')
    ax.set_xlim(biases[0], biases[-1])
    plt.savefig('zoomed_bias_dependence_wRC{}_N{}_V{}.pdf'.format(int(alpha),int(args['w0_1']), args['N_1'], int(args['V'])))
    #print max_coh_for_alpha, bias_at_max_list
    #ax.scatter(np.array(alpha_list)*pi, max_coh_for_alpha)
    #ax.scatter(np.array(alpha_list)*pi, bias_at_max_list)

def steadystate_coherence_and_RC_plot():
        try:
            coh_ops = load_obj('coherence_ops_N5') #[exciton_coherence]*35
            #site_ops = [site_coherence]*35
            #alpha_ph = [50/pi, 100/pi, 200/pi, 400/pi, 700/pi]
            fig = plt.figure(figsize=(12,6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharey=ax1)
            #colors = iter(['C1', 'C2', 'C3', 'C4', 'C5', 'c6', 'c7', 'c8'])
            for i, color in enumerate(plt.rcParams['axes.prop_cycle'][0:len(alpha_ph)]):
                biases = np.linspace(0, 1000, 35)
                col = color['color']
                #firstly get the data
                coh = vis.plot_bias_dependence(ax1, coh_ops, biases, alpha_ph[i], col, linestyle='-', linewidth=1.5, x_label=r'Steady State Exciton Coherence')
                p1 = vis.get_bias_dependence(Phonon_1, biases, alpha_ph[i])
                p2 = vis.get_bias_dependence(Phonon_2, biases, alpha_ph[i])
                # then calculate and plot phonon number difference
                phonon_diff = abs(p1-p2)
                label = r'$\pi\alpha=$'+'{}'.format(int(alpha_ph[i]*np.pi))+r'$cm^{-1}$'
                ax2.plot(phonon_diff, biases, color=col, linewidth=1.5, label=label)
                ax2.legend(loc='lower right')
                # add joining lines
                max_idx = list(phonon_diff).index(np.max(phonon_diff))
                bias_at_max, coh_at_bias = biases[max_idx], coh[max_idx].real
                ax1.plot([coh_at_bias, 0], [bias_at_max,bias_at_max], color=col, linestyle='--')
                ax2.plot([0., phonon_diff[max_idx].real], [bias_at_max,bias_at_max], color=col, linestyle='--')
                # pure formatting and aesthetics
                ax1.set_xlim(-0.09,0)
                ax2.set_xlim(0.,0.15)
                plt.setp(ax2.get_yticklabels(), visible=False)
                ax2.set_xlabel(r"$N_{RC_2}-N_{RC_1}$ at Steady state", weight='medium')
                #data_list.append(ssdata_for_alpha)
                fig.subplots_adjust(wspace=0.0)
            print "bias and coupling strength data seems to have been plotted"
        except Exception as err:
            print "data not plotted fully because", err

def get_steady_state_data():
    try:
        ncolors = len(plt.rcParams['axes.prop_cycle'])
        alpha_ph = np.arange(60, 420, 40)/pi
        PARAMS.update({'w_2':w_1})
        biases = np.linspace(100, 500, 25)
        #observable = exciton_coherence
        #check.get_coh_ops(PARAMS, biases, I)

        for alpha in alpha_ph:
            PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
            coh_ops = check.bias_dependence(biases, PARAMS, I)
            print "WE just finished pi*alpha={}".format(int(alpha*pi))
        save_obj(coh_ops, 'DATA/zoomed_coherence_ops_N{}_wRC{}_V{}'.format(int(N_1), int(w0_1), int(V)))
        steadystate_coherence_plot(PARAMS, alpha_ph, biases)
        plt.show()
    except Exception as err:
        print "data not calculated fully because", err

def ket_constructor(site_op, fock_1, fock_2, PARAMS):
    N_1, N_2, exc = PARAMS['N_1'], PARAMS['N_2'], PARAMS['exc']
    sites = qt.enr_fock([N_1,N_2],exc,[fock_1,fock_2])
    state = tensor(site_op, sites*sites.dag())
    return qt.operator_to_vector(state)

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
    w_2 = w_1
    V = 92. #0.1*8065.5
    w_opt = (w_1+w_2)*0.5 # Characteristic freq in optical spec.

    T_EM = 6000. # Optical bath temperature
    alpha_EM = 1.*5.309 # Optical S-bath strength (from inv. ps to inv. cm)(optical)
    mu = 1.

    T_1, T_2 = 300., 300. # Phonon bath temperature

    wc = 53. # Ind.-Boson frame phonon cutoff freq
    w0_2, w0_1 = 500., 500. # underdamped SD parameter omega_0
    w_xx = w_2 + w_1 + V
    alpha_1, alpha_2 = 400/pi, 400/pi # Ind.-Boson frame coupling
    N_1, N_2 = 5, 5 # set Hilbert space sizes
    exc = int((N_1+N_2)*0.5)
    num_cpus = 4
    J = J_minimal

    H_dim = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag() + V*(XO*OX.dag() + OX*XO.dag())
    PARAM_names = ['w_1', 'w_2', 'V', 'w_opt', 'w_xx', 'T_1', 'T_2', 'wc',
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
    dark_old= eVecs[1]*eVecs[1].dag()
    bright_old= eVecs[2]*eVecs[2].dag()
    energies, states = check.exciton_states(PARAMS)
    lam_p = 0.5*(w_1+w_2)+0.5*np.sqrt((w_2-w_1)**2+4*(V**2))
    lam_m = 0.5*(w_1+w_2)-0.5*np.sqrt((w_2-w_1)**2+4*(V**2))
    dark = tensor(states[0]*states[0].dag(), I)
    bright = tensor(states[1]*states[1].dag(), I)
    #print  (states[1]*states[1].dag()).tr(), bright_old, states[1]*states[1].dag()
    #print (states[0]*states[0].dag()).tr(), dark_old, states[0]*states[0].dag()
    exciton_coherence = tensor(states[0]*states[1].dag(), I)
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

    #Now we build all of the mapped operators and RC Liouvillian.



    # electromagnetic bath liouvillians

    #print sys.getsizeof(L_ns)
    opts = qt.Options(num_cpus=num_cpus)


    L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2, kappa_1, kappa_2 = RC.RC_mapping_UD(PARAMS)
    L_ns = EM.L_nonsecular(H_0, A_EM, PARAMS)
    L = L_RC + L_ns
    #print "Steady state is ", qt.steadystate(H_0)
    #calculate_dynamics()

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

    #plt.show()
