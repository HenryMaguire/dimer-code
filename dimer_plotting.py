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

def plot_bias_dependence(ax, observable, biases, alpha, color):
    name = 'DATA/bias_dependence_alpha{}'.format(int(alpha))
    data = load_obj(name)
    ss_values = np.array([(ss_dm*observable).tr() for ss_dm in data])
    label = r'$\pi\alpha=$'+'{}'.format(int(alpha*np.pi))+r'$cm^{-1}$'
    plt.scatter(biases, ss_values.real, label=label, color=color)
    #plt.scatter(biases, np.array(data).imag, marker='_', color=color)
    ax.set_ylim(-0.1, 0.0001)
    ax.set_xlim(0, 1000.0001)
    ax.set_ylabel("Exciton coherence")
    ax.set_xlabel(r"Bias ($cm^{-1}$)")
    plt.legend(loc='lower left')
    """
    ss_values = []
    for i in range(len(biases)):
        print "bias = {}".format(biases[i])
        ti = time.time()
        s = load_obj(name)
        obj = s[i]
        del s
        ss_dm = obj.get_steady_state()
        ss_values.append((ss_dm*observable).tr())
        print "it took {} seconds to calculate and store the steady_state".format(time.time()-ti)"""
    return ss_values

def plot_dynamics(DATA, timelist, ax, title='', split_colors=False):
    if not split_colors:
        split_colors = iter(plt.rcParams['axes.prop_cycle'])
    # takes a string saying the plot title
    '''
    if T_EM>0.0: # No need to plot SS for T=0
        ss_ns = steadystate(H_0, [L_RC+L_ns]).ptrace(0)
        #ss_v = steadystate(H_0, [L_RC+L_s]).ptrace(0)
        #ss_n = steadystate(H_0, [L_RC+L_naive]).ptrace(0)
        ss_g_ns = ss_ns.matrix_element(G.dag(), G)
        #ss_g_v = ss_v.matrix_element(G.dag(), G)
        #ss_g_n = ss_n.matrix_element(OO.dag(), OO)
        #plt.axhline(1-ss_g_v, color='b', ls='--')
        ax.axhline(1-ss_g_ns, color='g', ls='--')
        #plt.axhline(1-ss_g_n, color='r', ls='--')'''
    #title = title + r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0_1, alpha_1, T_EM)
    #ax.title(title)
    # Want to plot the dynamics for all of the eigenstates
    ax.plot(timelist, DATA.expect[0].real, label=r'|00>', color=p.next()['color'])
    ax.plot(timelist, DATA.expect[1].real, label=r'|XO>', color=p.next()['color'])
    ax.plot(timelist, DATA.expect[2].real, label=r'|OX>', color=p.next()['color'])
    ax.plot(timelist, DATA.expect[3].real, label=r'|XX>', color=p.next()['color'])
    ax.set_ylabel("Site populations")
    ax.set_xlabel("Time (ps)")
    ax.legend(loc='best')
    #p_file_name = "Notes/Images/Dynamics/Pop_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.savefig(p_file_name)

def plot_eig_dynamics(DATA, timelist, ax, title='', split_colors=False):
    """ split_colors is an iterator which contains colours. If you want two separate plots to have complimentary colour schemes but not identical, pass one iterator like below to split_colours in the separate plots.
    """
    if not split_colors:
        split_colors = iter(plt.rcParams['axes.prop_cycle'])
    #ax.title(title)
    linewidth = 1.5
    # Want to plot the dynamics for all of the eigenstates
    ax.plot(timelist, DATA.expect[0].real, label=r'Ground', color=split_colors.next()['color'],linewidth=linewidth)
    ax.plot(timelist, DATA.expect[4].real, label=r'Symmetric', color=split_colors.next()['color'], linewidth=linewidth)
    ax.plot(timelist, DATA.expect[5].real, label=r'Anti-symm', color=split_colors.next()['color'], linewidth=linewidth)
    ax.plot(timelist, DATA.expect[3].real, label=r'Biexciton', color=split_colors.next()['color'], linewidth=linewidth)
    ax.set_ylabel("Eigenstate population")
    ax.set_xlabel("Time (ps)")
    ax.legend()
    #p_file_name = "Notes/Images/Dynamics/Pop_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.savefig(p_file_name)
def plot_coherences(DATA, timelist, ax, title='', split_colors=False):
    if not split_colors:
        split_colors = iter(plt.rcParams['axes.prop_cycle'])
    #ax.title(r"$\alpha_{ph}=$""%i"r"$cm^{-1}$, $T_{EM}=$""%i K" %(alpha_1, T_EM))
    #plt.plot(timelist, DATA_s.expect[1].real, label='Vib. Lindblad', color='b')
    linewidth=1.5
    ax.plot(timelist, DATA.expect[6].real, label='real', color=split_colors.next()['color'],linewidth=linewidth)
    ax.plot(timelist, DATA.expect[6].imag, label='imaginary', color=split_colors.next()['color'],linewidth=linewidth)
    #plt.plot(timelist, DATA_naive.expect[1].real, label='Simple Lindblad', color='r',alpha=0.4)
    ax.legend()
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
    #ss_ns = steadystate(H_0, [L_RC])
    #ss_v = steadystate(H_0, [L_RC+L_s])
    #ss_n = steadystate(H_0, [L_RC+L_naive])
    #ss_ns_l = (ss_ns*Phonon_1).tr()
    #ss_ns_r = (ss_ns*Phonon_2).tr()
    #ax.axhline(ss_ns_l, color='g', ls='--')
    #ax.axhline(ss_ns_r, color='r', ls='--')
    #ax.title(r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0, alpha_ph, T_EM))
    #ax.plot(timelist, 1-DATA_nrwa.expect[0], label='nrwa', color='y')
    ax.plot(timelist, DATA_ns.expect[7].real, label='Left RC NS', color='g')
    ax.plot(timelist, DATA_ns.expect[8].real, label='Right RC NS', color='r')
    #ax.plot(timelist, DATA_s.expect[3].real, label='Vib. Lindblad', color='b')
    #ax.plot(timelist, DATA_naive.expect[7].real, label='Left RC Naive', color='g')
    #ax.plot(timelist, DATA_naive.expect[8].real, label='Right RC Naive', color='g')
    ax.set_ylabel("Reaction-Coordinate population")
    ax.set_xlabel("Time (ps)")
    ax.legend()



def plot_RC_disp(DATA_ns,ax):

    #ax.title(r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0, alpha_ph, T_EM))
    #ax.plot(timelist, 1-DATA_nrwa.expect[0], label='nrwa', color='y')
    ax.plot(timelist, DATA_ns.expect[9].real, label='Left RC NS', color='g')
    ax.plot(timelist, DATA_ns.expect[10].real, label='Right RC NS', color='r')
    #ax.plot(timelist, DATA_s.expect[3].real, label='Vib. Lindblad', color='b')
    #ax.plot(timelist, DATA_naive.expect[9].real, label='Left RC Naive', color='g')
    #ax.plot(timelist, DATA_naive.expect[10].real, label='Right RC Naive', color='g')
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
