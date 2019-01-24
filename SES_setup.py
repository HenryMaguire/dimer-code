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

I_sys = qeye(3)

reload(RC)
reload(opt)

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


def make_expectation_operators(PARAMS, H=None, site_basis=True):
    # makes a dict: keys are names of observables values are operators
    I_sys=qeye(PARAMS['sys_dim'])
    I = enr_identity([PARAMS['N_1'], PARAMS['N_2']], PARAMS['exc'])
    N_1, N_2, exc = PARAMS['N_1'], PARAMS['N_2'], PARAMS['exc']
    energies, states = exciton_states(PARAMS, shift=False)
    bright_vec = states[1]
    dark_vec = states[0]
    sigma_x = site_coherence+site_coherence.dag()
    sigma_y = 1j*(site_coherence-site_coherence.dag())
    sigma_z = XO_proj - OX_proj
    eta = np.sqrt(PARAMS['bias']**2 + 4*PARAMS['V']**2)
    eig_x_equiv = (2*PARAMS['V']/eta)*sigma_z - (0.5*PARAMS['bias']/eta)*sigma_x

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

    
    a_enr_ops = enr_destroy([N_1, N_2], exc)
    position1 = a_enr_ops[0].dag() + a_enr_ops[0]
    position2 = a_enr_ops[1].dag() + a_enr_ops[1]
    number1   = a_enr_ops[0].dag()*a_enr_ops[0]
    number2   = a_enr_ops[1].dag()*a_enr_ops[1]

    subspace_ops = [position1, position2, number1, number2]
    fullspace_ops += [tensor(I_sys, op) for op in subspace_ops]

    return dict((key_val[0], key_val[1]) for key_val in zip(labels, fullspace_ops))

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
    PARAMS['H_sub'] += V*(site_coherence+site_coherence.dag())
    PARAMS.update({'V': V})
    H, A_1, A_2 = RC.H_mapping_RC(PARAMS['H_sub'], PARAMS['coupling_ops'], PARAMS['w_1'],
                PARAMS['w_2'], PARAMS['kappa_1'], PARAMS['kappa_2'], PARAMS['N_1'], PARAMS['N_2'], PARAMS['exc'],
                shift=True)
    
    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_BMME(H[1], tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        L_add += opt.L_BMME(tensor(PARAMS['H_sub'],I), tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        #L_add += opt.L_phenom_SES(PARAMS)
    else:
        print "Not including optical dissipator"
    spar0 = sparse_percentage(L)
    if threshold:
        L_add.tidyup(threshold)
        L.tidyup(threshold)
    if not silent:
        print("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L)))
    PARAMS.update({'V': V})
    
    return H, L, L_add

def get_H_and_L(PARAMS,silent=False, threshold=0.):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent, shift=True, site_basis=True, parity_flip=PARAMS['parity_flip'])
    L_add = copy.deepcopy(L)
    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2
<<<<<<< HEAD
=======
    H_unshifted = PARAMS['w_1']*XO_proj + PARAMS['w_2']*OX_proj + PARAMS['V']*(site_coherence+site_coherence.dag())
>>>>>>> 2898e71b728bcfe6cc73311cbbb833758ca49d9b
    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_BMME(H[1], tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)
        L_add += opt.L_BMME(tensor(PARAMS["H_sub"],I), tensor(sigma,I), PARAMS, ME_type='nonsecular',                                 site_basis=True, silent=silent)
        #L_add += opt.L_phenom_SES(PARAMS)
    else:
        print "Not including optical dissipator"
    spar0 = sparse_percentage(L)
    if threshold:
        L_add.tidyup(threshold)
        L.tidyup(threshold)
    if not silent:
        print("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L)))

    return H, L, L_add

def get_H_and_L_additive(PARAMS,silent=False, threshold=0.):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent, shift=True, site_basis=True)

    N_1 = PARAMS['N_1']
    N_2 = PARAMS['N_2']
    exc = PARAMS['exc']
    mu = PARAMS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2
    
    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_BMME(H[1], tensor(sigma,I), PARAMS, ME_type='nonsecular', site_basis=True, silent=silent)

    else:
        print "Not including optical dissipator"
    spar0 = sparse_percentage(L)
    if threshold:
        L.tidyup(threshold)
    if not silent:
        print("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L)))

    return H, L

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
        print("Gap is {}. Phonon thermal energy is {}. Phonon SD peak is {}. N={}.".format(gap, phonon_energy, 
        SD_peak_position(Gamma, 1, w_0), N))

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
    dipole_1, dipole_2 = 1., 1.
    mu = (w_2*dipole_2)/(w_1*dipole_1)

    w_xx = w_2 + w_1

    H_sub = w_1*XO_proj + w_2*OX_proj + PARAMS_init['V']*(site_coherence+site_coherence.dag())
    PARAM_names = ['H_sub', 'w_1', 'w_2', 'bias', 'w_xx', 'mu']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS_init.update(dict((name, eval(name, scope)) for name in PARAM_names))
    return PARAMS_init

print("SES_setup loaded globally")