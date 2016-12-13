"""
Plotting module for the dimer dynamics
"""

from qutip import basis, ket, mesolve, qeye, tensor, thermal_dm, destroy, steadystate
import matplotlib.pyplot as plt
import numpy as np

import dimer_UD_liouv as RC
import dimer_driving_liouv as EM
import electronic_lindblad as EM_lind

reload(RC)
reload(EM)
reload(EM_lind)



def plot_dynamics(DATA, title=''):
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
    plt.plot(timelist, DATA.expect[-1].real, label=r'|+>', color='g')
    plt.plot(timelist, DATA.expect[-2].real, label=r'|->', color='b')
    plt.plot(timelist, DATA.expect[3].real, label=r'|XX>', color='r')
    plt.ylabel("Eigenstate population")
    plt.xlabel("Time (cm)")
    plt.legend()
    #p_file_name = "Notes/Images/Dynamics/Pop_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    #plt.savefig(p_file_name)
    plt.show()
def plot_coherences(DATA, title=''):

    plt.figure()
    plt.title(r"$\alpha_{ph}=$""%i"r"$cm^{-1}$, $T_{EM}=$""%i K" %(alpha_ph, T_EM))
    #plt.plot(timelist, DATA_s.expect[1].real, label='Vib. Lindblad', color='b')
    #plt.plot(timelist, DATA_ns.expect[1].real, label='Non-secular', color='g',alpha=0.7)
    plt.plot(timelist, DATA_naive.expect[1].real, label='Simple Lindblad', color='r',alpha=0.4)
    plt.legend()
    plt.ylabel("Coherence")
    plt.xlabel("Time (cm)")
    plt.xlim(0,2)
    file_name = "Notes/Images/Dynamics/Coh_a{:d}_Tph{:d}_Tem{:d}_w0{:d}.pdf".format(int(alpha_ph), int(T_ph), int(T_EM), int(w0))
    plt.savefig(file_name)
    plt.close()

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

if __name__ == "__main__":

    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()

    w_1 = 11000
    w_2 = 10000
    V = 0
    eps = w_1-w_2

    T_EM = 6000. # Optical bath temperature
    alpha_EM = 0.3 # System-bath strength (optical)
    mu = 0

    T_2 = T_1 = 300. # Phonon bath temperature

    wc = 53. # Ind.-Boson frame phonon cutoff freq
    w0_2  = w0_1 = 300. # underdamped SD parameter omega_0
    w_xx = w0_2 + w0_1 + V
    alpha_2 = alpha_1 = 100. # Ind.-Boson frame coupling
    N_1 = 6 # set Hilbert space sizes

    N_2 = 6
    exc = N_1 + N_2
    #Now we build all of the mapped operators and RC Liouvillian.
    L_RC, H_0, A_1, A_2, A_EM, wRC_1, wRC_2 = RC.RC_mapping_UD(w_1, w_2, w_xx, V, T_1, T_2, w0_1, w0_2, alpha_1, alpha_2, wc,  N_1, mu=mu)
    # electromagnetic bath liouvillians
    L_ns = EM.L_nonsecular(H_0, A_EM, alpha_EM, T_EM)
    L_s = EM.L_vib_lindblad(H_0, A_EM, alpha_EM, T_EM)
    L_naive = EM_lind.electronic_lindblad(w_xx, w_1, eps, V, mu, alpha_EM, T_EM, N_1, N_2, exc)
    # Set up the initial density matrix

    I_RC_1 = qeye(N_1)
    I_RC_2 = qeye(N_2)
    n_RC_1 = EM.Occupation(wRC_1, T_1)
    n_RC_2 = EM.Occupation(wRC_2, T_2)

    initial_sys = OO*OO.dag()
    rho_0 = tensor(initial_sys, thermal_dm(N_1, n_RC_1), thermal_dm(N_2, n_RC_2))
    OO = tensor(OO, I_RC_1, I_RC_2)
    XO = tensor(XO, I_RC_1, I_RC_2)
    OX = tensor(OX, I_RC_1, I_RC_2)
    XX = tensor(XX, I_RC_1, I_RC_2)

    # Expectation values and time increments needed to calculate the dynamics
    expects = [OO*OO.dag(), XX*XX.dag(), (XO+OX)*(XO+OX).dag(), (XO-OX)*(XO-OX).dag(), OX*XO.dag(), XO*XO.dag(), OX*OX.dag()]
    timelist = np.linspace(0,10,30000) # you need lots of points so that coherences are well defined -> spectra
    #nonsec_check(eps, H, A_em, N) # Plots a scatter graph representation of non-secularity. Could use nrwa instead.

    # Calculate dynamics
    DATA_ns = mesolve(H_0, rho_0, timelist, [L_RC+L_ns], expects, progress_bar=True)
    #DATA_s = mesolve(H_0, rho_0, timelist, [L_RC+L_s], expects, progress_bar=True)
    #DATA_naive = mesolve(H_0, rho_0, timelist, [L_RC+L_naive], expects, progress_bar=True)

    plot_dynamics(DATA_ns, title='Non-secular driving\n')
    #SS, nvals = check.SS_convergence_check(eps, T_EM, T_ph, wc, w0, alpha_ph, alpha_EM, start_n=10)
    #plt.plot(nvals, SS)
    #plot_dynamics_spec(DATA_s, timelist)

    #np.savetxt('DATA/Dynamics/dimer_DATA_ns.txt', np.array([1- DATA_ns.expect[0], timelist]), delimiter = ',', newline= '\n')
