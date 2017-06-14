"""
Plotting module for the dimer dynamics
"""

from qutip import steadystate, Qobj
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import time
matplotlib.style.use('ggplot')

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
    labels = [r'Ground', r'Symmetric', r'Anti-symm', r'Biexciton']
    colors = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])][0:4]
    info = zip([0,4,5,3], labels, colors) # expval id, etc., etc.
    #ax.title(title)
    linewidth = 1.5
    linestyle = '-'
    for i, l, c in info:
        ax.plot(timelist, DATA.expect[i].real, label=l, color=c, linewidth=linewidth, linestyle=linestyle)
        if ss_dm:
            ax.axhline((ss_dm*exp_ops[i]).tr().real, color=c, ls='--')
    ax.set_ylabel("Eigenstate population")
    ax.set_xlabel("Time (ps)")
    ax.legend()
    #p_file_name = "Notes/Images/Dynamics/Pop_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.savefig(p_file_name)
def plot_coherences(DATA, timelist, exp_ops, ax, title='', ss_dm = False):
    labels = [r'Real part', 'Imaginary part']
    colors = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])][0:2]
    coh = DATA.expect[6]
    ss = (ss_dm*exp_ops[6]).tr()
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
    ax.set_ylabel("Symm./Anti-symm. Eigenstate Coherence")
    ax.set_xlabel("Time (ps)")
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
