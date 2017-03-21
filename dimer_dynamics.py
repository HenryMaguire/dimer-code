import sys
from numpy import pi

from qutip import Qobj, basis, ket, mesolve, qeye, tensor, thermal_dm, destroy, enr_identity, enr_destroy, enr_thermal_dm
import qutip as qt
import matplotlib.pyplot as plt
import matplotlib
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
    N_1, N_2 = 2,2  # set Hilbert space sizes
    exc = int((N_1+N_2)*0.75)
    num_cpus = 4
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

    timelist = np.linspace(0,0.5,6000)

    #Now we build all of the mapped operators and RC Liouvillian.


    L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2, kappa_1, kappa_2 = RC.RC_mapping_UD(w_1, w_2, w_xx,
                                        V, T_1, T_2, w0_1, w0_2, alpha_1, alpha_2,
                                        wc,  N_1, N_2, exc, mu=mu, num_cpus=num_cpus)
    # electromagnetic bath liouvillians

    #print sys.getsizeof(L_ns)
    opts = qt.Options(num_cpus=num_cpus)
    ncolors = len(plt.rcParams['axes.prop_cycle'])
    #fig = plt.figure(figsize=(12,6))
    try:
        """
        biases = np.linspace(0, 1000, 35)
        data_list = []
        global DATA_ns
        alpha_ph = [50/pi, 100/pi, 200/pi, 400/pi, 700/pi]
        observable = Phonon_1
        for alpha in alpha_ph[1::]:
            PARAMS.update({'alpha_1':alpha, 'alpha_2':alpha})
            check.bias_dependence(biases, PARAMS)
            print "WE just finished pi*alpha={}".format(int(alpha*pi))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #colors = iter(['C1', 'C2', 'C3', 'C4', 'C5', 'c6', 'c7', 'c8'])
        for i, color in enumerate(plt.rcParams['axes.prop_cycle'][0:len(alpha_ph)]):
            biases = np.linspace(0, 1000, 35)
            ssdata_for_alpha = vis.plot_bias_dependence(ax, biases, alpha_ph[i], color['color'])
            data_list.append(ssdata_for_alpha)
        print "bias and coupling strength data seems to have been collected"
        """
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #observable = exciton_coherence
        #alpha = 200/pi
        #ss_values = check.bias_dependence(biases, PARAMS, observable)

        L_ns = EM.L_nonsecular(H_0, A_EM, eps, alpha_EM, T_EM, J, num_cpus=num_cpus)
        L_full = L_RC+L_ns

        DATA_ns = mesolve(H_0, rho_0, timelist, [L_full], expects, options=opts,
                                                            progress_bar=True)
        timelist/=0.188 # Convert from cm to picoseconds
        #DATA_ns = load_obj("DATA_N7_exc8")
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        title = 'Eigenstate population'
        #title = title + r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0_1, alpha_1, T_EM)
        split_colors = False
        vis.plot_eig_dynamics(DATA_ns, timelist, ax1, title='', split_colors=split_colors)
        print 'Non-secular eig dynamics calculated and plotted'
        ax2 = fig.add_subplot(122)
        vis.plot_coherences(DATA_ns, timelist, ax2, title='', split_colors=split_colors)
        print 'Non-secular eig coherences plotted'
    except Exception as err:
        print "Could not get non-secular-driving dynamics because ",err

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
