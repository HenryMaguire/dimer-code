"""
Plotting module for the dimer dynamics
"""
import os
import dimer_phonons as RC
import dimer_optical as EM
import optical_liouvillian_J as JAKE
from qutip import steadystate, Qobj
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import time
matplotlib.style.use('ggplot')
reload(JAKE)
reload(RC)
reload(EM)

from utils import *

class dataObject(Qobj):
    """ Allows you to store Hamiltonian, Liouvillian, dynamics,
     steady_states and timelists in an array-like onject.

    In the future:
     -May want to include plotting methods in this
     -Include different dynamics with different initial conditions
      list of all parameter regimes for dynamics
    """
    def __init__(self, L, H):
        self.H = H
        self.L = L
        self.steady_state = 0
        self.dynamics =[]
        self.timelist =[]
        self.params = {}
    def add_H(self, H):
        self.H = H
    def add_dynamics(self, dynamics, timelist):
        self.dynamics, self.timelist = dynamics, timelist
    def add_params(self, PARAMS):
        self.params = PARAMS
    def get_steady_state(self):
        if self.H:
            self.steady_state = steadystate(self.H, [self.L], method='direct')
            return self.steady_state
        else:
            print "add Hamiltonian first with .add_H(H)"
            return None
    def mutual_information(self):
        if self.steady_state:
            S_s = qt.entropy_vn(self.steady_state.ptrace(0))
            S_1 = qt.entropy_vn(self.steady_state.ptrace(1))
            S_2 = qt.entropy_vn(self.steady_state.ptrace(1))
            #S_s1 =
            return
def get_bias_dependence(observable, biases, alpha):
    name = 'DATA/dm_bias_dependence_alpha{}'.format(int(alpha))
    data = load_obj(name)
    return np.array([(ss_dm*observable).tr() for ss_dm in data])

def plot_bias_dependence(ax, observables, biases, alpha, color, x_label='', y_label =True, linestyle='-', linewidth=1.0,legend_on=True):
    p_name = main_dir+'phenom/steadystate_DMs_alpha{}'.format('alpha')
    #ns_name =
    p_data = load_obj(p_name)
    #ns_data = load_obj(ns_name)
    #for ss_dm, obs in zip(data, observables):
    #    print ss_dm
    ss_data = np.array([(ss_dm*obs).tr() for ss_dm, obs in zip(p_data, observables)])
    label = r'$\pi\alpha=$'+'{}'.format(int(alpha*np.pi))+r'$cm^{-1}$'
    if not legend_on:
        label = None
    if y_label:
        ax.set_ylabel(r"Bias ($cm^{-1}$)")
    ax.plot(ss_values.real, biases, label=label, color=color, ls=linestyle, linewidth=linewidth)
    #plt.scatter(biases, np.array(data).imag, marker='^', color=color)
    #ax.set_ylim(-0.1, 0.0001)
    #ax.set_xlim(0, 1000.0001)
    ax.set_xlabel(x_label)
    plt.legend(loc='lower left')
    return ss_values



def plot_dynamics(DATA, timelist, exp_ops, ax, title='', ss_dm = False):
    labels = [r'Ground', r'Site 1', r'Site 2', r'Biexciton']
    colors = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])][0:4]
    info = zip([0,1,2,3], labels, colors) # expval id, etc., etc.
    linewidth = 1.5
    linestyle = '-'
    for i, l, c in info:
        ax.plot(timelist, DATA.expect[i].real, label=l, color=c, linewidth=linewidth, linestyle=linestyle)
        if ss_dm:
            ax.axhline((ss_dm*exp_ops[i]).tr().real, ls='--')
    # plot the population dynamics for all of the sites
    ax.set_ylabel("Site populations")
    ax.set_xlabel("Time (ps)")
    ax.legend(loc='best')
    #ax.title(title)
    #p_file_name = "Notes/Images/Dynamics/Pop_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.savefig(p_file_name)

def plot_eig_dynamics(DATA, timelist, exp_ops, ax, title='', ss_dm = False):
    """ Docstring here please
    """
    labels = [r'Ground', r'Dark.', r'Bright', r'Biexciton']
    colors = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])][0:4]
    info = zip([0,5,6,3], labels, colors) # expval id, etc., etc.
    #ax.title(title)
    linewidth = 1.5
    linestyle = '-'
    for i, l, c in info:
        ax.plot(timelist, DATA.expect[i].real, label=l, color=c, linewidth=linewidth, linestyle=linestyle)
        if ss_dm:
            ax.axhline((ss_dm*exp_ops[i]).tr().real, color=c, ls='--')
    ax.set_ylabel("Eigenstate population")
    ax.set_xlabel("Time (ps)")
    ax.set_ylim(0,1)
    ax.set_xlim(0,timelist[-1])
    ax.legend()
    #p_file_name = "Notes/Images/Dynamics/Pop_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.savefig(p_file_name)
def plot_coherences(DATA, timelist, exp_ops, ax, title='', ss_dm = False):
    labels = [r'Real part', 'Imaginary part']
    colors = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])][0:2]
    coh = DATA.expect[7]
    ss = (ss_dm*exp_ops[7]).tr()
    info = zip([ss.real, ss.imag],[coh.real, coh.imag], labels, colors)

    #ax.title(r"$\alpha_{ph}=$""%i"r"$cm^{-1}$, $T_{EM}=$""%i K" %(alpha_1, T_EM))
    linewidth = 1.5
    linestyle = '-'
    for s, d, l, c in info:
        ax.plot(timelist, d, label=l, color=c,linewidth=linewidth, linestyle=linestyle)
        if ss_dm:
            ax.axhline(s, color=c, ls='--')
    #ax.title(title)
    ax.legend(loc='lower right')
    ax.set_ylabel("Eigenstate Coherence")
    ax.set_xlabel("Time (ps)")
    ax.set_xlim(0,timelist[-1])
    #file_name = "Notes/Images/Dynamics/Coh_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_1), int(T_1), int(T_EM), int(w0_1))
    #plt.savefig(file_name)
    #plt.close()

def plot_dynamics_spec(DAT, t):
    dyn = 1-DAT.expect[1]
    plt.figure()
    ss = dyn[-1]
    gg = dyn-ss
    spec = np.fft.fft(gg)
    freq = np.fft.fftfreq(t.shape[-1])
    plt.plot(freq, abs(spec.real), label=r'$re(S)$')
    plt.plot(freq, abs(spec.imag), linestyle="dotted", label=r'$im(S)$')
    #plt.plot(freq, abs(spec), linestyle='dotted')
    plt.title("Frequency spectrum of vibronic TLS coherence\n TLS-RC coupling=".format(kappa))
    plt.legend()
    plt.ylabel("Magnitude" )
    plt.xlabel(r"Frequency- $\epsilon$ ($cm^{-1}$)")
    p_file_name = "Notes/Images/Spectra/Coh_spec_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.xlim(0, 0.5)
    plt.savefig(p_file_name)
    d_file_name = "DATA/Spectra/Coh_spec_a{:d}_Tph{:d}_TEM{:d}_w0{:d}.txt".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    np.savetxt(d_file_name, np.array([spec, freq]), delimiter = ',', newline= '\n')
    plt.close()


def plot_RC_pop(DATA_ns, timelist, ax, title=''): # No need to plot SS for T=0
    #ax.title(r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0, alpha_ph, T_EM))
    ax.plot(timelist, DATA_ns.expect[7].real, label='Left RC NS', color='g')
    ax.plot(timelist, DATA_ns.expect[8].real, label='Right RC NS', color='r')
    ax.set_ylabel("Reaction-Coordinate population")
    ax.set_xlabel("Time (ps)")
    ax.legend()



def plot_RC_disp(DATA_ns,ax):
    #ax.title(r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0, alpha_ph, T_EM))
    ax.plot(timelist, DATA_ns.expect[9].real, label='Left RC NS', color='g')
    ax.plot(timelist, DATA_ns.expect[10].real, label='Right RC NS', color='r')
    ax.set_ylabel("Reaction-Coordinate displacement")
    ax.set_xlabel("Time (ps)")
    ax.legend()

def plot_manifolds(ax, H):
    eigs = H.eigenenergies()
    p_title = "Vibronic manifolds: " + r"$\alpha_{ph}$="+"{:d}".format(int(alpha_1))+ "$\epsilon$={:d}, $\omega_0=${:d}".format(int(eps), int(w0_1))
    #ax.title(p_title)
    plt.ylabel(r"Energy ($cm^{-1}$)")
    j = 0
    for i in eigs:
        col = 'b'
        """
        if j<H.shape[0]/2:
            col = 'b'
        else:
            col = 'r'
        """
        ax.axhline(i, color = col)
        j+=1
    #p_file_name = "Notes/Images/Spectra/Manifolds_a{:d}_Tph{:d}_Tem{:d}_w0{:d}_eps{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0), int(eps))
    #plt.savefig(p_file_name)
    #print "Plot saved: ", p_file_name
    #plt.show()
    return eigs


def process_data():
    data = {"ground": [[], 0.]}
    return data

def plot_observable(DATA, timelist, ax, things):
    labels = things['labels']
    for key in DATA.keys():
        print DATA[key]
        dynamics, SS = DATA[key][0], DATA[key][1]
        ax.plot(dynamics, timelist, label=labels[key])
        if things['show_steadystates']:
            ax.axhline(SS)

def exciton_states(PARS):
    w_1, w_2, V, bias = PARS['w_1'], PARS['w_2'],PARS['V'], PARS['bias']
    v_p, v_m = 0, 0
    eta = np.sqrt(4*(V**2)+bias**2)
    lam_p = w_2+(bias+eta)*0.5
    lam_m = w_2+(bias-eta)*0.5
    v_m = np.array([0., -(w_1-lam_p)/V, -1, 0.])
    #v_p/= /(1+(V/(w_2-lam_m))**2)
    v_m/= np.sqrt(np.dot(v_m, v_m))
    v_p = np.array([0, V/(w_2-lam_m),1., 0.])

    v_p /= np.sqrt(np.dot(v_p, v_p))
    #print  np.dot(v_p, v_m) < 1E-15
    return [lam_m, lam_p], [qt.Qobj(v_m), qt.Qobj(v_p)]

def calculate_dynamics(rho_0, L_RC, H_0, A_EM, expects, PARAMS, timelist, EM_approx='s', l=''):
    L=0
    if l == 'flat_':
        PARAMS.update({'mu':1})
    else:
        pass
    if EM_approx=='ns':
        L = EM.L_nonsecular(H_0, A_EM, PARAMS)
    elif EM_approx=='s':
        L = EM.L_secular(H_0, A_EM, PARAMS)
    elif EM_approx=='p':
        I = qt.enr_identity([PARAMS['N_1'],PARAMS['N_2']], PARAMS['exc'])
        energies, states = exciton_states(PARAMS)
        L = EM.L_phenom(states, energies, I, PARAMS)
    elif EM_approx =='j':
        energies, states = exciton_states(PARAMS)
        L = JAKE.EM_dissipator(states, PARAMS['w_xx'], PARAMS['w_2'], PARAMS['bias'],
                                            PARAMS['V'], PARAMS['mu'], PARAMS['alpha_EM'], PARAMS['T_EM'], PARAMS['J'],
                                            PARAMS['N_1'], PARAMS['exc'])
    else:
        raise KeyError
    L_full = L_RC+L
    opts = qt.Options(num_cpus=PARAMS['num_cpus'], store_final_state=True, nsteps=4000)
    DATA = qt.mesolve(H_0, rho_0, timelist, [L_full], expects, progress_bar=True, options=opts)
    ss_dm = 0
    try:
        ss_dm = qt.steadystate(H_0, [L_full])
    except Exception as err:
        print "Warning: steady state density matrix didn't converge. Probably"
        print "\t due to some problem with excitation restriction. \n"
        print err

    #timelist=timelist/0.188 # Convert from cm to picoseconds
    #DATA_ns = load_obj("DATA_N7_exc8")
    #fig = plt.figure(figsize=(12,6))
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(211)
    title = 'Eigenstate dynamics'
    #title = title + r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0_1, alpha_1, T_EM)
    plot_eig_dynamics(DATA, timelist, expects, ax1, ss_dm=ss_dm)
    ax2 = fig.add_subplot(212)
    plot_coherences(DATA, timelist, expects, ax2, ss_dm=ss_dm)
    lab='wc'
    if PARAMS['alpha_1']>PARAMS['w_1']/500.:
        lab = 'sc'
    data_dir = "DATA/Dynamics_J{}_N{}_exc{}".format(lab, PARAMS['N_1'], PARAMS['exc'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    plt.savefig(data_dir+"/{}_{}dynamics.pdf".format(EM_approx, l))
    data_name = data_dir+"/{}_{}data".format(EM_approx, l)
    save_obj(DATA, data_name)
    plt.close()
    print 'Plotting finished!'
    return ss_dm, DATA

def steadystate_coherence_plot(args, alpha_list, biases):
    main_dir = "DATA/bias_dependence_wRC{}_N{}_V{}_wc{}/".format(int(args['w0_1']), args['N_1'], int(args['V']), int(args['wc']))
    #energy_differences = 2*np.sqrt(4*float(args['V'])**2 + biases**2)
    #p_dm_dir = main_dir +"phenom/"
    ns_dm_dir = main_dir +"nonsecular/"
    ops_dir = main_dir +"operators/"
    coh_ops = load_obj(ops_dir+'eigcoherence_ops')
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    max_coh_for_alpha = []
    bias_at_max_list = []
    colors = ['m', 'b','r','g', 'k', 'y']
    for k, alpha in enumerate(alpha_list):
        #p_ss_dms = load_obj(p_dm_dir+'steadystate_DMs_alpha{}'.format(int(alpha)))
        ns_ss_dms = load_obj(ns_dm_dir+'steadystate_DMs_pialpha{}'.format(int(pi*alpha)))
        print len(ns_ss_dms), len(coh_ops)
        assert len(ns_ss_dms) == len(coh_ops)
        #p_coh_list = []
        ns_coh_list = []
        for i in range(len(ns_ss_dms)):
            #p_ss_obs = ((p_ss_dms[i]*coh_ops[i]).tr()).real
            ns_ss_obs = abs((ns_ss_dms[i]*coh_ops[i]).tr())
            #p_coh_list.append(p_ss_obs)
            ns_coh_list.append(ns_ss_obs)
        #ax.plot(biases, np.array(p_coh_list).real, linestyle='--', linewidth=1.2, color=colors[k])
        label = r"$\pi\alpha =$"+ "{}".format(int(pi*alpha))
        ax.plot(biases, np.array(ns_coh_list), label=label, color=colors[k])
    ax.set_xlabel(r'Bias $cm^{-1}$')
    ax.set_ylabel('Exciton Coherence')
    ax.legend()
    plt.savefig(main_dir+'bias_dependence.pdf')
    plt.close()

def steadystate_dark_plot(args, alpha_list, biases):
    main_dir = "DATA/bias_dependence_wRC{}_N{}_V{}_wc{}/".format(int(args['w0_1']), args['N_1'], int(args['V']), int(args['wc']))
    #energy_differences = 2*np.sqrt(4*float(args['V'])**2 + biases**2)
    #p_dm_dir = main_dir +"phenom/"
    ns_dm_dir = main_dir +"nonsecular/"
    ops_dir = main_dir +"operators/"
    dark_ops = load_obj(ops_dir+'dark_ops')
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    max_coh_for_alpha = []
    bias_at_max_list = []
    colors = ['m', 'b','r','g', 'k', 'y']
    for k, alpha in enumerate(alpha_list):
        #p_ss_dms = load_obj(p_dm_dir+'steadystate_DMs_alpha{}'.format(int(alpha)))
        ns_ss_dms = load_obj(ns_dm_dir+'steadystate_DMs_pialpha{}'.format(int(pi*alpha)))
        assert len(ns_ss_dms) == len(dark_ops)
        #p_coh_list = []
        ns_coh_list = []
        for i in range(len(ns_ss_dms)):
            #p_ss_obs = (p_ss_dms[i]*dark_ops[i]).tr()
            ns_ss_obs = (ns_ss_dms[i]*dark_ops[i]).tr()
            #p_coh_list.append(p_ss_obs)
            ns_coh_list.append(ns_ss_obs)
        #ax.plot(biases, np.array(p_coh_list).real, linestyle='--', linewidth=1.2, color=colors[k])
        label = r"$\pi\alpha =$"+ "{}".format(int(pi*alpha))
        ax.plot(biases, np.array(ns_coh_list).real, label=label, color=colors[k])
    ax.set_xlabel(r'Bias $cm^{-1}$')
    ax.set_ylabel('Dark Eigenstate Population')
    ax.legend()
    plt.savefig(main_dir+'dark_bias_dependence.pdf')
    plt.close()


def steadystate_bright_plot(args, alpha_list, biases):
    main_dir = "DATA/bias_dependence_wRC{}_N{}_V{}_wc{}/".format(int(args['w0_1']), args['N_1'], int(args['V']), int(args['wc']))
    #energy_differences = 2*np.sqrt(4*float(args['V'])**2 + biases**2)
    #p_dm_dir = main_dir +"phenom/"
    ns_dm_dir = main_dir +"nonsecular/"
    ops_dir = main_dir +"operators/"
    bright_ops = load_obj(ops_dir+'bright_ops')
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    max_coh_for_alpha = []
    bias_at_max_list = []
    colors = ['m', 'b','r','g', 'k', 'y']
    for k, alpha in enumerate(alpha_list):
        #p_ss_dms = load_obj(p_dm_dir+'steadystate_DMs_alpha{}'.format(int(alpha)))
        ns_ss_dms = load_obj(ns_dm_dir+'steadystate_DMs_pialpha{}'.format(int(pi*alpha)))
        assert len(ns_ss_dms) == len(bright_ops)
        #p_coh_list = []
        ns_coh_list = []
        for i in range(len(ns_ss_dms)):
            #p_ss_obs = (p_ss_dms[i]*bright_ops[i]).tr()
            ns_ss_obs = (ns_ss_dms[i]*bright_ops[i]).tr()
            #p_coh_list.append(p_ss_obs)
            ns_coh_list.append(ns_ss_obs)
        label = r"$\pi\alpha =$"+ "{}".format(int(pi*alpha))
        ax.plot(biases, np.array(ns_coh_list).real, label=label, color=colors[k])
    ax.set_xlabel(r'Bias $cm^{-1}$')
    ax.set_ylabel('Bright Eigenstate Population')
    ax.legend()
    #ax.set_xlim(-2000, 2000)
    plt.savefig(main_dir+'bright_bias_dependence.pdf')
    plt.close()

def steadystate_darkbright_plot(args, alpha_list, biases):
    main_dir = "DATA/bias_dependence_wRC{}_N{}_V{}_wc{}/".format(int(args['w0_1']), args['N_1'], int(args['V']), int(args['wc']))
    #energy_differences = 2*np.sqrt(4*float(args['V'])**2 + biases**2)
    #p_dm_dir = main_dir +"phenom/"
    ns_dm_dir = main_dir +"nonsecular/"
    ops_dir = main_dir +"operators/"
    bright_ops = load_obj(ops_dir+'bright_ops')
    dark_ops = load_obj(ops_dir+'dark_ops')
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    max_coh_for_alpha = []
    bias_at_max_list = []
    colors = ['m', 'b','r','g', 'k', 'y']
    OO = qt.basis(4,0)
    I = qt.enr_identity([args['N_1'],args['N_2']], args['exc'])
    OO = qt.tensor(OO*OO.dag(), I)
    for k, alpha in enumerate(alpha_list):
        #p_ss_dms = load_obj(p_dm_dir+'steadystate_DMs_alpha{}'.format(int(alpha)))
        ns_ss_dms = load_obj(ns_dm_dir+'steadystate_DMs_pialpha{}'.format(int(pi*alpha)))
        #assert len(ns_ss_dms) == len(bright_ops)
        dark_list = []
        bright_list = []
        ground_list = []

        for i in range(len(ns_ss_dms)):
            d_obs = (ns_ss_dms[i]*dark_ops[i]).tr()
            b_obs = (ns_ss_dms[i]*bright_ops[i]).tr()
            gr_obs = (ns_ss_dms[i]*OO).tr()
            ground_list.append(gr_obs)
            dark_list.append(d_obs)
            bright_list.append(b_obs)
        #ax.plot(biases, np.array(p_coh_list).real, linestyle='--', linewidth=1.2, color=colors[k])
        print bright_list
        label = r"$\pi\alpha =$"+ "{}".format(int(pi*alpha))
        ax.plot(biases, (np.array(dark_list)/(np.array(ground_list)+np.array(bright_list))).real, label=label, color=colors[k])
    #print energy_differences[int(len(energy_differences)/2)::]
    #print -1*(np.array(bright_list)-np.array(dark_list))[int(len(energy_differences)/2)::]
    ax.set_xlabel(r'Bias $cm^{-1}$')
    ax.set_ylabel('Relative Dark Population ratio (dark/(bright+ground))')
    ax.legend()
    #ax.set_xlim(-2000, 2000)
    plt.savefig(main_dir+'darkbrightdiff_bias_dependence.pdf')
    plt.close()

def steadystate_ground_plot(args, alpha_list, biases):
    main_dir = "DATA/bias_dependence_wRC{}_N{}_V{}_wc{}/".format(int(args['w0_1']), args['N_1'], int(args['V']), int(args['wc']))
    #energy_differences = 2*np.sqrt(4*float(args['V'])**2 + biases**2)
    #p_dm_dir = main_dir +"phenom/"
    ns_dm_dir = main_dir +"nonsecular/"
    ops_dir = main_dir +"operators/"
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    max_coh_for_alpha = []
    bias_at_max_list = []
    OO = qt.basis(4,0)
    I = qt.enr_identity([args['N_1'],args['N_2']], args['exc'])
    OO = qt.tensor(OO*OO.dag(), I)
    colors = ['m', 'b','r','g', 'k', 'y']
    for k, alpha in enumerate(alpha_list):
        #p_ss_dms = load_obj(p_dm_dir+'steadystate_DMs_alpha{}'.format(int(alpha)))
        ns_ss_dms = load_obj(ns_dm_dir+'steadystate_DMs_pialpha{}'.format(int(pi*alpha)))
        ground_list = []
        for i in range(len(ns_ss_dms)):
            gr_obs = (ns_ss_dms[i]*OO).tr()
            ground_list.append(gr_obs)
        #ax.plot(biases, np.array(p_coh_list).real, linestyle='--', linewidth=1.2, color=colors[k])
        label = r"$\pi\alpha =$"+ "{}".format(int(pi*alpha))
        ax.plot(biases, ground_list, label=label, color=colors[k])
    #print energy_differences[int(len(energy_differences)/2)::]
    #print -1*(np.array(bright_list)-np.array(dark_list))[int(len(energy_differences)/2)::]
    ax.set_xlabel(r'Bias $cm^{-1}$')
    ax.set_ylabel('Ground State Population')
    ax.legend()
    #ax.set_xlim(-2000, 2000)
    plt.savefig(main_dir+'ground_bias_dependence.pdf')
    plt.close()


def steadystate_coherence_and_RC_plot():
        try:
            main_dir = "DATA/bias_dependence_wRC{}_N{}_V{}_wc{}/".format(int(args['w0_1']), args['N_1'], int(args['V']), int(args['wc']))
            #energy_differences = 2*np.sqrt(4*float(args['V'])**2 + biases**2)
            #p_dm_dir = main_dir +"phenom/"
            ns_dm_dir = main_dir +"nonsecular/"
            ops_dir = main_dir +"operators/"
            coh_ops = load_obj(ops_dir+'eigcoherence_ops')
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

if __name__ == "__main__":
    """
    qt_data = [range(4),range(4),range(4),range(4)]
    steady_states = [1,1,1,1]
    timelist = [1,2,3,4]
    names = ["ground", "site_1", "site_2", "double"]
    data = dict()
    for i in range(len(names)):
        data[names[i]]=[qt_data[i], steady_states[i]]
    all_plots = ["site_pop", "site_coh", "eigen_pop", "eigen_coh"]
    site_pop = {"labels": {"ground":"Ground", "site_1": "Left Site","site_2": "Right Site",  "double": "Biexciton"}, "show_steadystates":True}

    #all_kwargs = dict((plot_name, ))
    obs = [["site_pop", "site_coh"], ["eigen_pop", "eigen_coh"]]
    n_rows, n_cols = len(obs), len(obs[0])
    subplot_number = 1
    fig = plt.figure()
    for j in range(n_rows):
        for i in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, subplot_number)
            subplot_number+=1
            plot_observable(data, timelist, ax, site_pop)
            plt.legend()
    plt.show()
    """
