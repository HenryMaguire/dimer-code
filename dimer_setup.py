import os
import time

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt

import copy 
import phonons as RC
import optical as opt

import optical as opt
from qutip import basis, qeye, enr_identity, enr_destroy, tensor, enr_thermal_dm, steadystate
from utils import *
reload(RC)
reload(opt)

OO = basis(4,0)
XO = basis(4,1)
OX = basis(4,2)
XX = basis(4,3)

site_coherence = OX*XO.dag()

OO_proj = OO*OO.dag()
XO_proj = XO*XO.dag()
OX_proj = OX*OX.dag()
XX_proj = XX*XX.dag()

sigma_m1 = OX*XX.dag() + OO*XO.dag()
sigma_m2 = XO*XX.dag() + OO*OX.dag()
sigma_x1 = sigma_m1+sigma_m1.dag()
sigma_x2 = sigma_m2+sigma_m2.dag()

I_dimer = qeye(4)

reload(RC)
reload(opt)

labels = [ 'OO', 'XO', 'OX', 'site_coherence', 
            'bright', 'dark', 'eig_x', 'eig_y', 'eig_z', 'eig_x_equiv', 'sigma_x', 'sigma_y', 'sigma_z',
             'RC1_position1', 'RC2_position', 
             'RC1_number', 'RC2_number', 'XX']

tex_labels = [ '$\\rho_0$', '$\\rho_1$', '$\\rho_2$', '$\\rho_12$', 
            '$|+ \\rangle$', '$|- \\rangle$', '$\\tilde{\\sigma}_x$', '$\\tilde{\\sigma}_y^{\prime}$','$\\tilde{\\sigma}_z^{\prime}$', 'eig_x_equiv',
            '$\\sigma_x$', '$\\sigma_y$','$\\sigma_z$',
             r'$\hat{x_1}$', r'$\hat{x_2}$', 
             r'$\hat{N_1}$', r'$\hat{N_2}$', '$\\rho_3$']


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
                    sigma_x, sigma_y, sigma_z, XX_proj]
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

def get_H_and_L_add(PARAMS,silent=False, threshold=0., site_basis=True, rwa=True):
    # this makes the RWA and compares additive to non-additive
    comment = 'Completed additive and non-additive liouvillians'
    if rwa:
        PARAMS = to_RWA(PARAMS)
        comment+= ' in RWA'
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS,silent=silent, shift=True, site_basis=site_basis)
    L_add = copy.deepcopy(L)
    I = enr_identity([PARAMS['N_1'], PARAMS['N_2']], PARAMS['exc'])
    sigma = sigma_m1 + PARAMS['mu']*sigma_m2

    if abs(PARAMS['alpha_EM'])>0:
        if rwa:
            # always assumes nonsecular
            optical_liouv = opt.L_BMME
        else:
            optical_liouv = opt.L_non_rwa

        L += optical_liouv(H[1], tensor(sigma,I), 
                            PARAMS, silent=silent)
        L_add += optical_liouv(tensor(PARAMS['H_sub'],I), tensor(sigma,I), 
                                PARAMS, silent=silent)
    else:
        print "Not including optical dissipator"
    spar0 = sparse_percentage(L)
    if threshold:
        L.tidyup(threshold)
    if not silent:
        if abs(threshold)>0:
            print("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L)))
        print comment
    output_names = ['H', 'L', 'L_add', 'PARAMS']
    scope = locals() # Lets eval below use local variables, not global
    output_dict = dict((name, eval(name, scope)) for name in output_names)

    return output_names

def get_H_and_L_RWA(PARAMS,silent=False, threshold=0.):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARAMS, silent=silent)
    L_RWA, H_RWA, A_1, A_2, PARAMS_RWA = RC.RC_mapping(to_RWA(PARAMS),
                                                        silent=silent)
    assert(PARAMS_RWA['RWA'])
    
    I = enr_identity([PARAMS['N_1'], PARAMS['N_2']], PARAMS['exc'])
    sigma = sigma_m1 + PARAMS['mu']*sigma_m2
    if abs(PARAMS['alpha_EM'])>0:
        L += opt.L_non_rwa(H[1], tensor(sigma,I), PARAMS, silent=silent)
        # rwa form uses non-sec master equation
        
        L_RWA += opt.L_BMME(H_RWA[1], tensor(sigma,I), PARAMS_RWA, 
                             ME_type='nonsecular', silent=silent)
    else:
        print "Not including optical dissipator"
    #L*= -1
    #L_RWA*= -1 # Hacky minus sign
    spar0 = sparse_percentage(L)
    if threshold:
        L.tidyup(threshold)
        L_RWA.tidyup(threshold)
    if not silent:
        if abs(threshold)>0:
            print("Chopping reduced the sparsity from {:0.3f}% to {:0.3f}%".format(spar0, sparse_percentage(L)))
        print("Completed non-additive liouvillians in RWA and non-RWA form")
    # returns non-rwa
    output_names = ['H', 'L', 'PARAMS', 'H_RWA', 'PARAMS_RWA', 'L_RWA']
    scope = locals() # Lets eval below use local variables, not global
    output_dict = dict((name, eval(name, scope)) for name in output_names)
    return output_dict


def get_L(PARS,silent=False, threshold=0., site_basis=True):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARS, silent=silent)

    N_1 = PARS['N_1']
    N_2 = PARS['N_2']
    exc = PARS['exc']
    mu = PARS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2

    if abs(PARS['alpha_EM'])>0:
        if PARS['num_cpus']>1:
            L+= opt.L_non_rwa_par(H[1], tensor(sigma,I), PARS, 
                                        silent=silent, site_basis=site_basis)
        else:
            L += opt.L_non_rwa(H[1], tensor(sigma,I), PARS, silent=silent, 
                                                            site_basis=site_basis)
    else:
        print "Not including optical dissipator"
    L = -1*qt.liouvillian(H[1], c_ops=[L])
    if threshold:
        L.tidyup(threshold)
    return L


def PARAMS_setup(bias=100., w_2=2000., V = 100., alpha=0.1,
                                 T_EM=0., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, w_0=200, Gamma=50., N=3,
                                 silent=False, exc_diff=0, w_xx=0.):
    N_1 = N_2 = N
    exc = N+exc_diff
    gap = sqrt(bias**2 +4*(V**2))
    phonon_energy = T_ph*0.695

    w_1 = w_2 + bias
    dipole_1, dipole_2 = 1., 1.
    mu = (w_2*dipole_2)/(w_1*dipole_1)
    sigma = sigma_m1 + mu*sigma_m2
    T_1, T_2 = T_ph, T_ph # Phonon bath temperature

    Gamma_1 = Gamma_2 = Gamma
    w0_2, w0_1 = w_0, w_0 # underdamped SD parameter omega_0
    if not silent:
        plot_UD_SD(Gamma_1, alpha, w_0, eps=w_2)
    #w_xx = w_2 + w_1
    exciton_coupling = sigma_x1*sigma_x2
    H_sub = w_1*sigma_m1.dag()*sigma_m1 + w_2*sigma_m2.dag()*sigma_m2 
    H_sub += V*exciton_coupling + w_xx*XX_proj
    coupling_ops = [sigma_m1.dag()*sigma_m1, sigma_m2.dag()*sigma_m2] # system-RC operators
    if not silent:
        print "Gap is {}. Phonon thermal energy is {}. Phonon SD peak is {}. N={}.".format(gap, phonon_energy, SD_peak_position(Gamma, 1, w_0),N)

    J = J_minimal

    PARAM_names = ['H_sub', 'coupling_ops', 'w_1', 'w_2', 
                    'V', 'bias', 'w_xx', 'T_1', 'T_2', 'w0_1', 
                    'w0_2', 'T_EM', 'alpha_EM','mu', 'num_cpus', 
                    'J','dipole_1','dipole_2', 'Gamma_1', 'Gamma_2']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)

    PARAMS.update({'alpha_1': alpha, 'alpha_2': alpha})
    PARAMS.update({'N_1': N_1, 'N_2': N_2, 'exc': exc})
    PARAMS.update({'sys_dim' : 4})
    PARAMS.update({'RWA' : False})
    return PARAMS

def to_RWA(PARAMS):
    H_sub = PARAMS['w_1']*sigma_m1.dag()*sigma_m1 + PARAMS['w_2']*sigma_m2.dag()*sigma_m2 
    H_sub += PARAMS['V']*(site_coherence+site_coherence.dag()) + PARAMS['w_xx']*XX_proj # exciton coupling is now sigma_x*sigma_x
    try:
        PARAMS['H_sub'] = H_sub
    except:
        raise ValueError("Must have used PARAMS_setup previously")
    PARAMS.update({'RWA' : True})
    return PARAMS