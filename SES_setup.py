import os
import time

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt
import copy 

import phonons as RC
import optical as opt

from phonons import RC_mapping
from optical import L_non_rwa, L_phenom_SES
from qutip import basis, qeye, enr_identity, enr_destroy, tensor, enr_thermal_dm, steadystate
from utils import *
import imp


OO = basis(3,0)
XO = basis(3,1)
OX = basis(3,2)


site_coherence = OX*XO.dag()

OO_proj = OO*OO.dag()
XO_proj = XO*XO.dag()
OX_proj = OX*OX.dag()

sigma_m1 =  OO*XO.dag()
sigma_m2 =  OO*OX.dag()
sigma_x1 = sigma_m1+sigma_m1.dag()
sigma_x2 = sigma_m2+sigma_m2.dag()

sigma_x = 0.5*(site_coherence+site_coherence.dag())
sigma_y = 0.5*(1j*(site_coherence-site_coherence.dag()))

I_sys = qeye(3)

imp.reload(RC)
imp.reload(opt)

labels = [ 'OO', 'XO', 'OX', 'site_coherence', 
            'bright', 'dark', 'eig_x', 'eig_y', 'eig_z', 'eig_x_equiv', 'sigma_x', 'sigma_y', 'sigma_z',
             'RC1_position1', 'RC2_position', 
             'RC1_number', 'RC2_number']

tex_labels = [ '$\\rho_0$', '$\\rho_1$', '$\\rho_2$', '$\\rho_12$', 
            '$|+ \\rangle$', '$|- \\rangle$', '$\\tilde{\\sigma}_x$', '$\\tilde{\\sigma}_y^{\prime}$','$\\tilde{\\sigma}_z^{\prime}$', 'eig_x_equiv',
            '$\\sigma_x$', '$\\sigma_y$','$\\sigma_z$',
             r'$\hat{x_1}$', r'$\hat{x_2}$', 
             r'$\hat{N_1}$', r'$\hat{N_2}$']


def make_expectation_labels():
    # makes a dict: keys are names of observables, values are latex friendly labels
    assert(len(tex_labels) == len(labels))
    return dict((key_val[0], key_val[1]) for key_val in zip(labels, tex_labels))


def make_expectation_operators(PARAMS, H=None, weak_coupling=False, shift=True):
    # makes a dict: keys are names of observables values are operators
    I_sys=qeye(PARAMS['sys_dim'])
    I = enr_identity([PARAMS['N_1'], PARAMS['N_2']], PARAMS['exc'])
    N_1, N_2, exc = PARAMS['N_1'], PARAMS['N_2'], PARAMS['exc']
    energies, states = exciton_states(PARAMS, shift=shift)
    bright_vec = states[1]
    dark_vec = states[0]
    sigma_z = XO_proj - OX_proj
    eta = np.sqrt(PARAMS['bias']**2 + 4*PARAMS['V']**2)
    eig_x_equiv = (2*PARAMS['V']/eta)*sigma_z - (PARAMS['bias']/eta)*sigma_x

    # electronic operators
     # site populations site coherences, eig pops, eig cohs
    subspace_ops = [OO_proj, XO_proj, OX_proj, site_coherence,
                   bright_vec*bright_vec.dag(), dark_vec*dark_vec.dag(),
                   dark_vec*bright_vec.dag()+dark_vec.dag()*bright_vec,
                   1j*(dark_vec*bright_vec.dag()-dark_vec.dag()*bright_vec),
                   bright_vec*bright_vec.dag()-dark_vec.dag()*dark_vec, eig_x_equiv,
                    sigma_x, sigma_y, sigma_z]
    # put operators into full RC tensor product basis
    fullspace_ops = [tensor(op, I) for op in subspace_ops]
    # RC operators
    # RC positions, RC number state1, RC number state1, RC upper N fock, RC ground fock

    if weak_coupling:
        return dict((key_val[0], key_val[1]) for key_val in zip(labels[0:len(subspace_ops)], subspace_ops))
    else:
        a_enr_ops = enr_destroy([N_1, N_2], exc)
        position1 = a_enr_ops[0].dag() + a_enr_ops[0]
        position2 = a_enr_ops[1].dag() + a_enr_ops[1]
        number1   = a_enr_ops[0].dag()*a_enr_ops[0]
        number2   = a_enr_ops[1].dag()*a_enr_ops[1]

        subspace_ops = [position1, position2, number1, number2]
        fullspace_ops += [tensor(I_sys, op) for op in subspace_ops]

        return dict((key_val[0], key_val[1]) for key_val in zip(labels, fullspace_ops))

def separate_states(H, PARAMS, trunc=0.8):
    # truncation removes the really dodgy states for which the parity is unclear 
    # (this might not be numerical error, but it probs is)
    ops = make_expectation_operators(PARAMS)
    energies, states = H.eigenstates()
    energies, states = sort_eigs(energies, states)
    energies, states = energies[0:int(len(states)*trunc)], states[0:int(len(states)*trunc)]
    parities = [(state*state.dag()*ops['sigma_x']).tr() for state in states]
    phonon_occ_dict = {'dark': [], 'bright': [], 'ground': []}
    states_dict = {'dark': [], 'bright': [], 'ground': []}
    energies_dict = {'dark': [], 'bright': [], 'ground': []} # for checking
    for i, parity in enumerate(parities):
        occ_1 = (states[i].dag()*ops['RC1_number']*states[i]).tr().real
        occ_2 = (states[i].dag()*ops['RC2_number']*states[i]).tr().real
        if abs(parity)<1e-10:
            states_dict['ground'].append(states[i])
            energies_dict['ground'].append(energies[i])
            phonon_occ_dict['ground'].append((occ_1, occ_2))
        elif parity>1e-10:
            states_dict['bright'].append(states[i])
            energies_dict['bright'].append(energies[i])
            phonon_occ_dict['bright'].append((occ_1, occ_2))
        elif parity<-1e-10:
            states_dict['dark'].append(states[i])
            energies_dict['dark'].append(energies[i])
            phonon_occ_dict['dark'].append((occ_1, occ_2))
        else:
            raise ValueError("Parity is {} ".format(parity))
    if len(states_dict['ground'])  == len(states):
        print("This will not work for V=0. Ground contains all states.")
    #print(len(states_dict['dark']), len(states_dict['bright']), len(states_dict['ground']))
    #assert (len(states_dict['dark']) == len(states_dict['bright']))
    dark_bright_check(states_dict, ops)
    return energies_dict, states_dict, phonon_occ_dict

def get_H_and_L_local(PARAMS, silent=False, threshold=0.):
    V = PARAMS['V']
    PARAMS['H_sub'] = PARAMS['w_1']*XO_proj + PARAMS['w_2']*OX_proj 
    PARAMS.update({'V': 0.})
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent, shift=True, site_basis=True, parity_flip=PARAMS['parity_flip'])
    L_add = copy.deepcopy(L)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2
    PARAMS['H_sub'] += V*(site_coherence + site_coherence.dag())
    PARAMS.update({'V': V})
    H, A_1, A_2 = RC.H_mapping_RC(PARAMS['H_sub'], PARAMS['coupling_ops'], PARAMS['w_1'], PARAMS['w_2'],
                                PARAMS['kappa_1'], PARAMS['kappa_2'], PARAMS['N_1'], PARAMS['N_2'], PARAMS['exc'], shift=True)
    
    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_BMME(H[1], tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        L_add += opt.L_BMME(tensor(PARAMS['H_sub'],I), tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        #L_add += opt.L_phenom_SES(PARAMS)
    else:
        print("Not including optical dissipator")
    spar0 = sparse_percentage(L)
    if threshold:
        L_add.tidyup(threshold)
        L.tidyup(threshold)
    if not silent:
        print(("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L))))
    PARAMS.update({'V': V})
    
    return H, L, L_add


def get_H_and_L_add_and_sec(PARAMS,silent=False, threshold=0., shift_in_additive=False):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent, shift=True, site_basis=True, parity_flip=PARAMS['parity_flip'])
    L_add = copy.deepcopy(L)
    L_sec = copy.deepcopy(L)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2
    if shift_in_additive:
        H_add = tensor(H[0],I)
    else:
        H_add = tensor(PARAMS['H_sub'], I)

    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_non_rwa(H[1], tensor(sigma,I), PARAMS, silent=silent) #opt.L_BMME(H[1], tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        L_add += opt.L_non_rwa(H_add, tensor(sigma,I), PARAMS, silent=silent) #opt.L_BMME(tensor(H_unshifted,I), tensor(sigma,I), PARAMS, 
        L_sec += opt.L_secular(H[1], tensor(sigma,I), PARAMS) #opt.L_BMME(tensor(H_unshifted,I), tensor(sigma,I), PARAMS, 
        #ME_type='nonsecular', site_basis=True, silent=silent)
        #L_secular(H_vib, A, args, silent=False)
        #L_add += opt.L_phenom_SES(PARAMS)
    else:
        print("Not including optical dissipator")

    spar0 = sparse_percentage(L)
    if threshold:
        L_add.tidyup(threshold)
        L.tidyup(threshold)
    if not silent:
        print(("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L))))
    return H, {'nonadd':L , 'add': L_add, 'sec': L_sec}, PARAMS

def get_H_and_L_full(PARAMS,silent=False, threshold=0.):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent, shift=True, site_basis=True)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2


    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_non_rwa(H[1], tensor(sigma,I), PARAMS, silent=silent) #opt.L_BMME(H[1], tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)

    else:
        print("Not including optical dissipator")

    spar0 = sparse_percentage(L)
    if threshold:
        L.tidyup(threshold)
    if not silent:
        print(("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L))))
    return H, {'nonadd':L }, PARAMS

def get_H_and_L(PARAMS,silent=False, threshold=0., shift_in_additive=False):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent, shift=True, site_basis=True, parity_flip=PARAMS['parity_flip'])
    L_add = copy.deepcopy(L)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2
    if shift_in_additive:
        H_add = tensor(H[0],I)
    else:
        H_add = tensor(PARAMS['H_sub'], I)

    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_non_rwa(H[1], tensor(sigma,I), PARAMS, silent=silent) #opt.L_BMME(H[1], tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        L_add += opt.L_non_rwa(H_add, tensor(sigma,I), PARAMS, silent=silent) #opt.L_BMME(tensor(H_unshifted,I), tensor(sigma,I), PARAMS, 
        #ME_type='nonsecular', site_basis=True, silent=silent)
        #L_add += opt.L_phenom_SES(PARAMS)
    else:
        print("Not including optical dissipator")

    spar0 = sparse_percentage(L)
    if threshold:
        L_add.tidyup(threshold)
        L.tidyup(threshold)
    if not silent:
        print(("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L))))
    return H, {'nonadd':L , 'add': L_add}, PARAMS


def get_H_and_L_RWA(PARAMS, silent=False, threshold=0.):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent, shift=True, site_basis=True, parity_flip=PARAMS['parity_flip'])
    L_RWA = copy.deepcopy(L)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']
    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2
    H_unshifted = PARAMS['w_1']*XO_proj + PARAMS['w_2']*OX_proj + PARAMS['V']*(site_coherence+site_coherence.dag())
    if abs(PARAMS['alpha_EM'])>0:
        L_RWA += opt.L_BMME(H[1], tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        L+= opt.L_non_rwa(H[1], tensor(sigma,I), PARAMS, silent=silent)
    else:
        print("Not including optical dissipator")
    spar0 = sparse_percentage(L)
    if threshold:
        L_RWA.tidyup(threshold)
        L.tidyup(threshold)
    if not silent:
        print(("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L))))
    return H, L, L_RWA, PARAMS

def get_H_and_L_wc(H, PARAMS, silent=True, secular_phonon=False, 
                                shift=True, tol=1e-6):
    import optical as opt
    import weak_phonons as wp
    imp.reload(wp)
    imp.reload(opt)
    ti = time.time()
    L_s = wp.weak_phonon(H, PARAMS, secular=secular_phonon, tol=tol)
    L_ns = copy.deepcopy(L_s)
    mu = PARAMS['mu']
    sigma = sigma_m1 + mu*sigma_m2
    if abs(PARAMS['alpha_EM'])>0:
        L_ns += opt.L_non_rwa(H, sigma, PARAMS, silent=silent)
        #opt.L_BMME(H, sigma, PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        L_s +=  wp.L_sec_wc_SES(PARAMS, silent=silent)
        #L += opt.L_BMME(H, sigma, PARAMS, ME_type='secular', site_basis=True, silent=silent)
    H += 0.5*pi*(PARAMS['alpha_1']*sigma_m1.dag()*sigma_m1 + PARAMS['alpha_2']*sigma_m2.dag()*sigma_m2)
    if not silent:
        print("WC non-secular and secular dissipators calculated in {} seconds".format(time.time() - ti))
    return H, L_ns, L_s

def get_H_and_L_additive(PARAMS,silent=False, threshold=0.):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent, shift=True, site_basis=True)
    H_unshifted = PARAMS['w_1']*XO_proj + PARAMS['w_2']*OX_proj + PARAMS['V']*(site_coherence+site_coherence.dag())
    L_add = copy.deepcopy(L)
    L_add_shift = copy.deepcopy(L)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2
    
    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_non_rwa(H[1], tensor(sigma,I), PARAMS, silent=silent)
        L_add += opt.L_non_rwa(tensor(H_unshifted,I), tensor(sigma,I), PARAMS, silent=silent)
        L_add_shift += opt.L_non_rwa(tensor(H[0],I), tensor(sigma,I), PARAMS, silent=silent)
    else:
        print("Not including optical dissipator")
    spar0 = sparse_percentage(L)
    if threshold:
        L.tidyup(threshold)
    if not silent:
        print(("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L))))

    return H, {'nonadd':L, 'add-shift':L_add_shift , 'add': L_add}, PARAMS

def get_H(PARAMS):
    H, phonon_operators, PARAMS = RC.mapped_H(PARAMS)
    sigma = sigma_m1 + PARAMS['mu']*sigma_m2 
    I = enr_identity([PARAMS['N_1'], PARAMS['N_2']], PARAMS['exc'])
    return H, phonon_operators, tensor(sigma+sigma.dag(), I), PARAMS

def PARAMS_setup(bias=100., w_2=2000., V = 100., alpha=100.,
                                 T_EM=0., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, w_0=200, Gamma=50., N=3,
                                 silent=False, exc_diff=0, sys_dim=3, alpha_bias=0., parity_flip=False):
    # alpha_1 = alpha+alpha_bias
    # Sets up the parameter dict
    N_1 = N_2 = N
    exc = N+exc_diff
    gap = sqrt(bias**2 +4*(V**2))
    phonon_energy = T_ph*0.695


    w_1 = w_2 + bias
    dipole_1, dipole_2 = 1., 1.
    mu = (w_2*dipole_2)/(w_1*dipole_1)
    T_1, T_2 = T_ph, T_ph # Phonon bath temperature

    Gamma_1 = Gamma_2 = Gamma
    w0_2, w0_1 = w_0, w_0 # underdamped SD parameter omega_0
    if not silent:
        plot_UD_SD(Gamma_1, alpha, w_0, eps=w_2)
    w_xx = w_2 + w_1
    if not silent:
        print(("Gap is {}. Phonon thermal energy is {}. Phonon SD peak is {}. N={}.".format(gap, phonon_energy, 
        SD_peak_position(Gamma, 1, w_0), N)))

    J = J_minimal
    H_sub = w_1*XO_proj + w_2*OX_proj + V*(site_coherence+site_coherence.dag())
    coupling_ops = [sigma_m1.dag()*sigma_m1, sigma_m2.dag()*sigma_m2] # system-RC operators
    PARAM_names = ['H_sub', 'coupling_ops', 'w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2',
                   'w0_1', 'w0_2', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J',
                   'dipole_1','dipole_2', 'Gamma_1', 'Gamma_2', 'parity_flip']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)

    PARAMS.update({'alpha_1': alpha+alpha_bias, 'alpha_2': alpha})
    PARAMS.update({'N_1': N_1, 'N_2': N_2, 'exc': exc})
    PARAMS.update({'sys_dim' : sys_dim})
    return PARAMS

def PARAMS_update_bias(PARAMS_init=None, bias_value=10.):
    # Sets up the parameter dict

    bias = bias_value
    w_2 = PARAMS_init['w_2']
    w_1 = w_2 + bias
    dipole_1, dipole_2 = PARAMS_init['dipole_1'], PARAMS_init['dipole_2']
    mu = (w_2*dipole_2)/(w_1*dipole_1)

    w_xx = w_2 + w_1

    H_sub = w_1*XO_proj + w_2*OX_proj + PARAMS_init['V']*(site_coherence+site_coherence.dag())
    PARAM_names = ['H_sub', 'w_1', 'w_2', 'bias', 'w_xx', 'mu']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS_init.update(dict((name, eval(name, scope)) for name in PARAM_names))
    return PARAMS_init


def displace(offset, a):
    return (offset*(a.dag()) - offset.conjugate()*a).expm()

def undisplaced_initial(init_sys, PARAMS):
    n1 = Occupation(PARAMS['w0_1'], PARAMS['T_1'])
    n2 = Occupation(PARAMS['w0_2'], PARAMS['T_2'])
    return tensor(init_sys, qt.enr_thermal_dm([PARAMS['N_1'], PARAMS['N_2']], PARAMS['exc'], 
                                              [n1, n2]))
def position_ops(PARAMS):
    atemp = enr_destroy([PARAMS['N_1'], PARAMS['N_2']], PARAMS['exc'])
    return [tensor(I_sys, (a + a.dag())/sqrt(2)) for a in atemp] # Should have a 0.5 in this

def displaced_initial(init_sys, PARAMS, silent=False, return_error=False):
    # Works for 
    offset_1 = sqrt(pi*PARAMS['alpha_1']/(2*PARAMS['w0_1']))
    offset_2 = sqrt(pi*PARAMS['alpha_2']/(2*PARAMS['w0_2']))
    atemp = enr_destroy([PARAMS['N_1'], PARAMS['N_2']], PARAMS['exc'])
    x = position_ops(PARAMS)
    
    r0 = undisplaced_initial(init_sys, PARAMS)
    disp = copy.deepcopy(r0)
    for offset, a_ in zip([offset_1, offset_2], atemp):
        d = tensor(I_sys, displace(offset, a_))
        disp =  d * disp * d.dag()
    error = 100*(abs((disp*x[0]).tr()- offset_1)/offset_1 + abs((disp*x[1]).tr()- offset_2)/offset_2)
    if not silent:
        print ("Error in displacement: {:0.8f}%".format(error))
        print ("Ratio of kBT to Omega: {:0.4f}".format(0.695*PARAMS['T_1']/PARAMS['w0_1']))
        if ((PARAMS['T_1'] != PARAMS['T_2']) or (PARAMS['w0_1'] != PARAMS['w0_2'])):
           print("Ratio of kBT to Omega (2): {:0.4f}".format(0.695*PARAMS['T_2']/PARAMS['w0_2']))
    if return_error:   
        return disp, error
    else:
        return disp

def get_converged_N(PARAMS, err_threshold=1e-2, min_N=4, max_N=10, silent=True, exc_diff_N=False):
    err = 0
    for N in range(min_N,max_N+1):
        if exc_diff_N:
            exc_diff = N # when using partial traces we can't use ENR
        else:
            exc_diff = 0
        PARAMS.update({'N_1':N, 'N_2':N, 'exc':N+exc_diff})
        disp, err = displaced_initial(OO_proj, PARAMS, silent=True, return_error=True)
        if err<err_threshold:
            return PARAMS
    print("Error could only converge to {}".format(err))
    return PARAMS

print("SES_setup loaded globally")