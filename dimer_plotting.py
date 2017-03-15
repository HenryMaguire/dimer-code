"""
Plotting module for the dimer dynamics
"""

from qutip import steadystate
import qutip as qt
import matplotlib.pyplot as plt
import numpy as np




def plot_dynamics(DATA, title=''):
    # takes a string saying the plot title
    plt.figure()
    if T_EM>0.0: # No need to plot SS for T=0
        ss_ns = steadystate(H_0, [L_RC+L_ns]).ptrace(0)
        #ss_v = steadystate(H_0, [L_RC+L_s]).ptrace(0)
        #ss_n = steadystate(H_0, [L_RC+L_naive]).ptrace(0)
        ss_g_ns = ss_ns.matrix_element(G.dag(), G)
        #ss_g_v = ss_v.matrix_element(G.dag(), G)
        #ss_g_n = ss_n.matrix_element(OO.dag(), OO)
        #plt.axhline(1-ss_g_v, color='b', ls='--')
        plt.axhline(1-ss_g_ns, color='g', ls='--')
        #plt.axhline(1-ss_g_n, color='r', ls='--')
    title = title + r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0_1, alpha_1, T_EM)
    plt.title(title)
    # Want to plot the dynamics for all of the eigenstates
    plt.plot(timelist, DATA.expect[0].real, label=r'|00>', color='y')
    plt.plot(timelist, DATA.expect[1].real, label=r'|XO>', color='g')
    plt.plot(timelist, DATA.expect[2].real, label=r'|OX>', color='b')
    plt.plot(timelist, DATA.expect[3].real, label=r'|XX>', color='r')
    plt.ylabel("Site populations")
    plt.xlabel("Time (ps)")
    plt.legend()
    #p_file_name = "Notes/Images/Dynamics/Pop_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.savefig(p_file_name)

def plot_eig_dynamics(DATA, title=''):
    # takes a string saying the plot title
    plt.figure()
    if T_EM>0.0: # No need to plot SS for T=0
        ss_ns = steadystate(H_0, [L_RC+L_ns]).ptrace(0)
        ss_v = steadystate(H_0, [L_RC+L_s]).ptrace(0)
        ss_n = steadystate(H_0, [L_RC+L_naive]).ptrace(0)
        #ss_g_ns = ss_ns.matrix_element(G.dag(), G)
        #ss_g_v = ss_v.matrix_element(G.dag(), G)
        #ss_g_n = ss_n.matrix_element(OO.dag(), OO)
        #plt.axhline(1-ss_g_v, color='b', ls='--')
        #plt.axhline(1-ss_g_ns, color='g', ls='--')
        #plt.axhline(1-ss_g_n, color='r', ls='--')
    title = title + r"$\omega_0=$""%i"r"$cm^{-1}$, $\alpha_{ph}=$""%f"r"$cm^{-1}$, $T_{EM}=$""%i K" %(w0_1, alpha_1, T_EM)
    plt.title(title)
    # Want to plot the dynamics for all of the eigenstates
    plt.plot(timelist, DATA.expect[0].real, label=r'|00>', color='y')
    plt.plot(timelist, DATA.expect[4].real, label=r'|+>', color='g')
    plt.plot(timelist, DATA.expect[5].real, label=r'|->', color='b')
    plt.plot(timelist, DATA.expect[3].real, label=r'|XX>', color='r')
    plt.ylabel("Eigenstate population")
    plt.xlabel("Time (ps)")
    plt.legend()
    #p_file_name = "Notes/Images/Dynamics/Pop_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.savefig(p_file_name)
def plot_coherences(DATA, title=''):

    plt.figure()
    plt.title(r"$\alpha_{ph}=$""%i"r"$cm^{-1}$, $T_{EM}=$""%i K" %(alpha_1, T_EM))
    #plt.plot(timelist, DATA_s.expect[1].real, label='Vib. Lindblad', color='b')
    plt.plot(timelist, DATA_ns.expect[6].real, label='real', color='g')
    plt.plot(timelist, DATA_ns.expect[6].imag, label='imaginary', color='r')
    #plt.plot(timelist, DATA_naive.expect[1].real, label='Simple Lindblad', color='r',alpha=0.4)
    plt.legend()
    plt.ylabel("Eigenstate Coherence")
    plt.xlabel("Time (cm)")
    file_name = "Notes/Images/Dynamics/Coh_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_1), int(T_1), int(T_EM), int(w0_1))
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


def plot_RC_pop(DATA_ns, ax): # No need to plot SS for T=0
    ss_ns = steadystate(H_0, [L_RC])
    #ss_v = steadystate(H_0, [L_RC+L_s])
    #ss_n = steadystate(H_0, [L_RC+L_naive])
    ss_ns_l = (ss_ns*Phonon_1).tr()
    ss_ns_r = (ss_ns*Phonon_2).tr()
    ax.axhline(ss_ns_l, color='g', ls='--')
    ax.axhline(ss_ns_r, color='r', ls='--')
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
