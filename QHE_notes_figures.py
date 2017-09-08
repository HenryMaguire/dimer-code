import sys
import os
import traceback
import time

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
import dimer_tests as check
import optical_liouvillian_J as JAKE

matplotlib.style.use('ggplot')

reload(RC)
reload(EM)
reload(vis)
reload(check)



def named_plot_creator(rho_0, L_RC, H_0, SIGMA_1, SIGMA_2, expects, PARAMS, timelist, EM_approx='s', figure_num = '2', l ='', make_new_data=False):
    ss_dm = False
    L=0
    data_dir = "DATA/QHE_notes_fig{}/N{}_exc{}".format(figure_num, PARAMS['N_1'], PARAMS['exc'])
    data_name = data_dir+"/{}_{}data".format(EM_approx, l)
    plot_name = data_dir+"/{}_{}dynamics.pdf".format(EM_approx, l)
    if l == 'flat_':
        PARAMS.update({'mu':1})
        PARAMS.update({'J':J_flat})
    else:
        mu = (PARAMS['w_2']*PARAMS['dipole_2'])/(PARAMS['w_1']*PARAMS['dipole_1'])
<<<<<<< HEAD
        #print "Mu is  {}".format(mu)
=======
>>>>>>> eigen-bug-fixes
        PARAMS.update({'mu':mu})
        PARAMS.update({'J':J_minimal})
    I = qt.enr_identity([PARAMS['N_1'],PARAMS['N_2']], PARAMS['exc'])
    print PARAMS
    A_EM = tensor(SIGMA_1+PARAMS['mu']*SIGMA_2, I)
    opts = qt.Options(num_cpus=PARAMS['num_cpus'], nsteps=6000)
    ''' define names for files, we'll need these in every if statement'''
    if make_new_data:
        if EM_approx=='ns':
            L = EM.L_nonsecular_par(H_0, A_EM, PARAMS)
        elif EM_approx=='s':
<<<<<<< HEAD
=======
            print "Secular {} parameters are {}".format(l, PARAMS)
>>>>>>> eigen-bug-fixes
            L = EM.L_secular_par(H_0, A_EM, PARAMS)
        elif EM_approx=='p':
            print "Phenom {} parameters are {}".format(l, PARAMS)
            L = EM.L_phenom(I, PARAMS)
        elif EM_approx =='j':
            energies, states = check.exciton_states(PARAMS)
            L = JAKE.EM_dissipator(states, PARAMS['w_xx'], PARAMS['w_2'], PARAMS['bias'],
                                                PARAMS['V'], PARAMS['mu'], PARAMS['alpha_EM'], PARAMS['T_EM'], PARAMS['J'],
                                                PARAMS['N_1'], PARAMS['exc'])
        else:
            raise KeyError
        L_full = L_RC+L
        '''
        try:
            ss_dm = qt.steadystate(H_0, [L_full])
        except Exception as err:
            print "Warning: steady state density matrix didn't converge. Probably"
            print "\t due to some problem with excitation restriction. \n"
            print err
        '''
        try:
            DATA = qt.mesolve(H_0, rho_0, timelist, [L_full], e_ops=expects, progress_bar=True, options=opts)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            save_obj(DATA, data_name)

            fig = plt.figure(figsize=(10,8))
            ax1 = fig.add_subplot(211)
            title = 'Eigenstate dynamics'
            #title = title + r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0_1, alpha_1, T_EM)
            vis.plot_eig_dynamics(DATA, timelist, expects, ax1, ss_dm=ss_dm)
            ax2 = fig.add_subplot(212)
            vis.plot_coherences(DATA, timelist, expects, ax2, ss_dm=ss_dm)

            print "plot saved at: {}".format(plot_name)
        except Exception as err:
            DATA = None
            print "could not calculate or plot dynamics because:\n {}".format(err)

    else:
        del L_RC
        try:
            DATA = load_obj(data_name)
            fig = plt.figure(figsize=(12,8))
            ax1 = fig.add_subplot(211)
            title = 'Eigenstate dynamics'
            #title = title + r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0_1, alpha_1, T_EM)
            vis.plot_eig_dynamics(DATA, timelist, expects, ax1, ss_dm=ss_dm)
            ax2 = fig.add_subplot(212)
            vis.plot_coherences(DATA, timelist, expects, ax2, ss_dm=ss_dm)
            print "plot saved at: {}".format(plot_name)
        except Exception as err:
            DATA = None
            print 'Could not plot the data because:\n {}'.format(err)
    #J_DATA = load_obj("DATA/QHE_notes_fig{}/N{}_exc{}/jake_data".format(figure_num, PARAMS['N_1'], PARAMS['exc']))
    J_DATA = load_obj("DATA/QHE_notes_fig{}/N{}_exc{}/jake_data".format(figure_num, 2, 4))
    ax1.plot(timelist, J_DATA.expect[0], linestyle='--')
    ax1.plot(timelist, J_DATA.expect[4], linestyle='--')
    ax1.plot(timelist, J_DATA.expect[5], linestyle='--')
    ax1.plot(timelist, J_DATA.expect[3], linestyle='--')
    ax2.plot(timelist, J_DATA.expect[6].real, linestyle='--')
    ax2.plot(timelist, J_DATA.expect[6].imag, linestyle='--')
    plt.savefig(plot_name)
    plt.close()

    return ss_dm, DATA

def data_maker(w_2, bias, V, T_EM, alpha_EM, alpha_1, alpha_2, N, end_time, figure_num, initial, make_new_data=False):
    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()
    w_1 = w_2 + bias
    dipole_1, dipole_2 = 1., 1.
    mu = w_2*dipole_2/(w_1*dipole_1)

    T_1, T_2 = 300., 300. # Phonon bath temperature

    wc = 1*53.08 # Ind.-Boson frame phonon cutoff freq
    w0_2, w0_1 = 500., 500. # underdamped SD parameter omega_0
    w_xx = w_2 + w_1
    N_1, N_2 = N,N # set Hilbert space sizes
    exc = N_1 + N_2
    if N_1>4:
        exc= N_1
    num_cpus = 2
    J = J_minimal

    PARAM_names = ['w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2', 'wc',
                    'w0_1', 'w0_2', 'alpha_1', 'alpha_2', 'N_1', 'N_2', 'exc', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J', 'dipole_1','dipole_2']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)
    H_dim = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag() + V*(XO*OX.dag() + OX*XO.dag())
    energies, states = check.exciton_states(PARAMS)
    bright_vec = states[1]
    dark_vec = states[0]
    energies_n, states_n = H_dim.eigenstates()


    print energies_n, energies
    dark_vec_n = states_n[1]
    bright_vec_n = states_n[2]

<<<<<<< HEAD
    #bright_vec = states_n[2]
    #dark_vec = states_n[1]
    #print "Dark state overlap with numerical dark:{}".format(dark_vec.dag()*dark_vec_n)
    #print "Dark state overlap with numerical brgith:{}".format(dark_vec.dag()*bright_vec_n)
    #print "brigtht state overlap with numerical brgith:{}".format(bright_vec.dag()*dark_vec_n)
    #print "Brgight state overlap with numerical vector:{}".format(bright_vec.dag()*bright_vec_n)
=======
    print "Dark state overlap with numerical dark:{}".format(dark_vec.dag()*dark_vec_n)
    print "Dark state overlap with numerical brgith:{}".format(dark_vec.dag()*bright_vec_n)
    print "brigtht state overlap with numerical dark:{}".format(bright_vec.dag()*dark_vec_n)
    print "Brgight state overlap with numerical bright:{}".format(bright_vec.dag()*bright_vec_n)
>>>>>>> eigen-bug-fixes
    I_dimer = qeye(4)
    I = enr_identity([N_1,N_2], exc)
    dark = tensor(dark_vec*dark_vec.dag(), I)
    bright = tensor(bright_vec*bright_vec.dag(), I)
    exciton_coherence = tensor(dark_vec*bright_vec.dag(), I)

    '''Defining DM states'''
    site_coherence = tensor(OX*XO.dag(), I)
    OO = tensor(OO*OO.dag(), I)
    XO = tensor(XO*XO.dag(), I)
    OX = tensor(OX*OX.dag(), I)
    XX = tensor(XX*XX.dag(), I)

    #Now we build all of the mapped operators and RC Liouvillian.

    # electromagnetic bath liouvillians

    opts = qt.Options(num_cpus=num_cpus, store_states=True)
    ncolors = len(plt.rcParams['axes.prop_cycle'])
    ''' generate the RC liouvillian'''
    L_RC, H_0, A_1, A_2, SIG_1, SIG_2, wRC_1, wRC_2, kappa_1, kappa_2 = RC.RC_mapping_OD(PARAMS)
    ''' make the RC observable operators '''
    atemp = enr_destroy([N_1,N_2], exc)
    n_RC_1 = Occupation(wRC_1, T_1)
    n_RC_2 = Occupation(wRC_2, T_2)
    phonon_num_1 = atemp[0].dag()*atemp[0]
    phonon_num_2 = atemp[1].dag()*atemp[1]
    x_1 = (atemp[0].dag()+atemp[0])
    x_2 = (atemp[1].dag()+atemp[1])
    Phonon_1 = tensor(I_dimer, phonon_num_1)
    Phonon_2 = tensor(I_dimer, phonon_num_2)
    disp_1 = tensor(I_dimer, x_1)
    disp_2 = tensor(I_dimer, x_2)

    thermal_RCs = enr_thermal_dm([N_1,N_2], exc, [n_RC_1, n_RC_2])
    ''' initial density operator state'''
    rho_0 = tensor(basis(4,initial)*basis(4,initial).dag(),thermal_RCs)

    ops = [OO, XO, OX, XX, site_coherence]
    # Expectation values and time increments needed to calculate the dynamics
    expects = ops + [dark, bright, exciton_coherence]
    expects +=[Phonon_1, Phonon_2, disp_1, disp_2]

    timelist = np.linspace(0,end_time,4000*end_time)

    #DATA_J = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
    #                        timelist, EM_approx='j', figure_num =  figure_num, make_new_data=make_new_data)
    #del DATA_J

<<<<<<< HEAD
    #DATA_P = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
    #                        timelist, EM_approx='p', figure_num =  figure_num,
    #                        make_new_data=make_new_data)
=======
    DATA_P = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
                            timelist, EM_approx='p', figure_num =  figure_num,
                            make_new_data=make_new_data)

>>>>>>> eigen-bug-fixes
    DATA_P = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
                            timelist, l ='flat_', EM_approx='p', figure_num =  figure_num,
                            make_new_data=make_new_data)
    del DATA_P
<<<<<<< HEAD
    #DATA_S = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
    #                        timelist, EM_approx='s', figure_num =  figure_num,
    #                        make_new_data=make_new_data)
=======
    DATA_S = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
                            timelist, l ='flat_', EM_approx='s', figure_num =  figure_num,
                            make_new_data=make_new_data)

>>>>>>> eigen-bug-fixes
    DATA_S = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
                            timelist, EM_approx='s', figure_num =  figure_num,
                            make_new_data=make_new_data)
    del DATA_S
    """
    DATA_NS = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
                            timelist, EM_approx='ns', figure_num =  figure_num,
                            make_new_data=make_new_data)
    save_params(PARAMS, figure_num, '')
    DATA_NS = named_plot_creator(rho_0, L_RC, H_0, SIG_1, SIG_2, expects, PARAMS,
                            timelist, l ='flat_', EM_approx='ns', figure_num =  figure_num,
                            make_new_data=make_new_data)
    del DATA_NS
<<<<<<< HEAD
    """
=======
    #save_params(PARAMS, figure_num, 'flat_')
>>>>>>> eigen-bug-fixes

    return PARAMS

def save_params(PARAMS, fig, l):
    block = ''
    fn = 'DATA/QHE_notes_fig{}/N{}_exc{}/{}PARAMS.txt'.format(fig,
                                            PARAMS['N_1'], PARAMS['exc'], l)
    print fn
    f = 0
    f = open(fn, 'w')
    for key, value in PARAMS.items():
        line = '{}: \t {} \n'.format(key, value)
        block+= line
    f.write(block)
    f.close()
    return None


if __name__ == "__main__":

    # (w_2, bias, V, T_EM, alpha_EM, alpha_1, alpha_2, N, end_time, figure_num)
    try:
        # Plot batch 1: flat spectrum, fully converged, overlay jake's data on the top
        #figure 2
        PARAMS =  data_maker(100., 0., 20, 50, 1., 0., 0., 2, 1, '2a', 1,  make_new_data=True)
        PARAMS = data_maker(100., 10., 20, 50, 1., 0., 0., 2, 10, '2b', 1, make_new_data=True)


        #figure 4
<<<<<<< HEAD
        """
=======
>>>>>>> eigen-bug-fixes
        PARAMS = data_maker(1500., 50., 100, 5700, 0.1, 2., 2., 4, 1, '4ab', 0, make_new_data=True)

        PARAMS = data_maker(1500., 50., 100, 5700, 0.1, 2., 2., 4, 4, '4cd', 0, make_new_data=True)

        #figure 5
        N = 5
        PARAMS = data_maker(1500., 50., 100, 5700, 0.1, 100/pi, 100/pi, N, 1, '5ab-p', 0, make_new_data=True)
        PARAMS = data_maker(1500., 50., 100, 5700, 0.1, 100/pi, 100/pi, N, 4, '5cd-p', 0, make_new_data=True)
        PARAMS = data_maker(1500., 50., 100, 5700, 0.1, 100, 100, N, 1, '5ab', 0, make_new_data=True)
        PARAMS = data_maker(1500., 50., 100, 5700, 0.1, 100, 100, N, 4, '5cd', 0, make_new_data=True)
<<<<<<< HEAD
        """
=======
>>>>>>> eigen-bug-fixes
    except:
        var = traceback.format_exc()
        print var
        f = open('errors.log', 'w')
        f.write(var+'\n')
        f.close()
