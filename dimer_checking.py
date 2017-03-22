import time


from qutip import basis, ket, mesolve, qeye, tensor, thermal_dm, destroy, steadystate
import matplotlib.pyplot as plt
import numpy as np
import dimer_phonons as RC
import dimer_optical as EM
#import electronic_lindblad as EM_naive
from dimer_plotting import dataObject
from utils import *
reload(RC)
reload(EM)

def bias_dependence(biases, args):
    name = 'DATA/dm_bias_dependence_alpha{}'.format(int(args['alpha_1']))
    ss_list = []
    for eps in biases:
        w_1 = args['w_1']
        w_2 = w_1-eps
        w_xx = w_1 + w_2 + args['V']
        av_w = (w_1+w_2)*0.5
        L_RC, H, A_1, A_2, A_EM, wRC_1, wRC_2, kappa_1, kappa_2 = RC.RC_mapping_UD(
                                        w_1, w_2, w_xx, args['V'], args['T_1'],
                                        args['T_2'], args['w0_1'], args['w0_2'], args['alpha_1'],
                                        args['alpha_2'], args['wc'], args["N_1"], args['N_2'],
                                        args['exc'],mu=args['mu'], num_cpus=args['num_cpus'])
        #print L_RC.shape
        L_ns = EM.L_nonsecular(H, A_EM, av_w, args['alpha_EM'], args['T_EM'], args['J'],
                                        num_cpus=args['num_cpus'])
        ti = time.time()
        # rather than saving all the massive objects to a list, just calculate steady_states and return them
        ss_list.append(steadystate(H, [L_RC+L_ns]))
        print "Calculating the steady state took {} seconds".format(time.time()-ti)
        print "so far {} steady states".format(len(ss_list))
    print "file saving at {}".format(name)
    save_obj(ss_list, name)

def SS_convergence_check(sigma, w_1, w_2, w_xx, V, T_1, T_2, w0_1, w0_2, alpha_1, alpha_2, wc,  alpha_EM, T_EM, mu=0, expect_op='bright', time_units='cm', start_n=2, end_n=5, method='direct'):

    """
    TODO: rewrite this entire method to take in the dimer parameters
          Plot all of the state populations
    """
    # Only for Hamiltonians of rotating wave form
    XO = basis(4,1)
    OX = basis(4,2)
    ss_list_s,ss_list_ns,ss_list_naive  = [],[],[] # steady states
    r_vector = XO+OX # r_vector is the ket vector on the right in the .matrix_element operation. Default is E.
    l_vector = (XO+OX).dag() # Default is bright state
    N_values = range(start_n,end_n)
    eps = w_1-w_2
    if expect_op == 'coherence':
        l_vector = (G-E).dag()
    else:
        pass
    for n in N_values:
        L_RC, H, A_1, A_2, A_EM, wRC_1, wRC_2 = RC.RC_mapping_UD(w_1, w_2, w_xx, V, T_1, T_2, w0_1, w0_2, alpha_1, alpha_2, wc,  n, N_2=n, mu=mu, time_units='cm')
        #L_s = EM.L_vib_lindblad(H, A_EM, alpha_EM, T_EM)
        L_ns = EM.L_nonsecular(H, A_EM, alpha_EM, T_EM)
        L_naive = EM_naive.electronic_lindblad(w_xx, w_1, eps, V, mu, alpha_EM, T_EM, n, n, 2*n)
        #ss_s = steadystate(H, [L_RC+L_s], method=method).ptrace(0)
        ss_ns = steadystate(H, [L_RC+L_ns], method=method).ptrace(0)
        ss_naive = steadystate(H, [L_RC+L_naive], method=method).ptrace(0)
        #ss_list_s.append(ss_s.matrix_element(l_vector, r_vector))
        ss_list_ns.append(ss_ns.matrix_element(l_vector, r_vector))
        ss_list_naive.append(ss_naive.matrix_element(l_vector, r_vector))
        print "N=", n, "\n -----------------------------"
    plt.figure()
    #plt.ylim(0,0.4)
    #plt.plot(N_values, ss_list_s, label='secular')
    plt.plot(N_values, ss_list_ns, label='non-secular')
    plt.plot(N_values, ss_list_naive, label='naive')
    plt.legend()
    plt.ylabel("Excited state population")
    plt.xlabel("RC Hilbert space dimension")
    p_file_name = "Notes/Images/Checks/SuperPop_convergence_a{:d}_Tem{:d}_w0{:d}_eps{:d}_{}.pdf".format(int(alpha_1), int(T_EM), int(w0_1), int(eps), method)
    plt.savefig(p_file_name)
    return ss_list_s,ss_list_ns,ss_list_naive, p_file_name

if __name__ == "__main__":
    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    sigma_m1 = OX*XX.dag() + OO*XO.dag()
    sigma_m2 = XO*XX.dag() + OO*OX.dag()
    sigma_x1 = sigma_m1+sigma_m1.dag()
    sigma_x2 = sigma_m2+sigma_m2.dag()

    w_1 = 6000
    w_2 = 5000
    V = 1000
    eps = w_1-w_2

    T_EM = 6000. # Optical bath temperature
    alpha_EM = 0.3 # System-bath strength (optical)
    mu = 0

    T_2 = T_1 = 300. # Phonon bath temperature

    wc = 53. # Ind.-Boson frame phonon cutoff freq
    w0_2  = w0_1 = 300. # underdamped SD parameter omega_0
    w_xx = w0_2 + w0_1 + V
    alpha_2 = alpha_1 = 400. # Ind.-Boson frame coupling
    #N_1 = 6 # set Hilbert space sizes
    #N_2 = 6
    #Now we build all the operators
    """
    L_RC, H, A_EM, A_nrwa, wRC, kappa= RC.RC_function_UD(sigma, eps, T_ph, wc, w0, alpha_ph, N)
    L_s = EM.L_vib_lindblad(H, A_EM, alpha_EM, T_EM)
    L_ns = EM.L_nonsecular(H, A_EM, alpha_EM, T_EM)
    L_naive = EM.L_EM_lindblad(eps, A_EM, alpha_EM, T_EM)
    ss_naive = steadystate(H, [L_RC+L_naive]).ptrace(0)
    #TD, rates  = nonsec_check_A(H, A_EM, alpha_EM, T_EM, N)
    #plt.figure()
    #plt.scatter(TD, rates)
    #plt.show()
    """
    plt.figure()
    ss_list_s,ss_list_ns,ss_list_naive, p_file_name = SS_convergence_check(sigma_m1+(1-mu)*sigma_m1, w_1, w_2, w_xx, V, T_1, T_2, w0_1, w0_2, alpha_1, alpha_2, wc,  alpha_EM, T_EM, start_n = 2, end_n=5)
    #eps_values = range(1000, 2000, 50)+range(2000, 4000, 500)+range(4000, 14000, 1000)
    #N_values = [30]*len(range(1000, 2000, 50)) + [20]*len(range(2000, 4000, 500)) + [12]*len(range(4000, 14000, 1000))
    #solver_method = 'power'
    #ss_list_s,ss_list_ns,ss_list_naive, p_file_name = plot_SS_divergences(sigma, eps, T_EM, T_ph, wc, w0, alpha_ph, alpha_EM, N_values, eps_values, method=solver_method)
    print "Plot saved: ",p_file_name
    plt.savefig(p_file_name)
    plt.close()
