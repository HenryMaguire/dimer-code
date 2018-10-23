import os
import time

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt


import phonons as RC
import optical as opt

from phonons import RC_mapping
from optical import L_non_rwa, L_phenom
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
            'bright', 'dark', 'eig_coherence',
             'RC1_position1', 'RC2_position', 
             'RC1_number', 'RC2_number', 'sigma_x', 'sigma_y']

def make_expectation_operators(PARS):
    # makes a dict: keys are names of observables values are operators
    I_sys=qeye(PARS['sys_dim'])
    I = enr_identity([PARS['N_1'], PARS['N_2']], PARS['exc'])
    energies, states = exciton_states(PARS, shift=False)
    bright_vec = states[1]
    dark_vec = states[0]
    # electronic operators
     # site populations site coherences, eig pops, eig cohs
    subspace_ops = [OO_proj, XO_proj, OX_proj, site_coherence,
                   bright_vec*bright_vec.dag(), dark_vec*dark_vec.dag(),
                   dark_vec*bright_vec.dag(),
                    site_coherence+site_coherence.dag(),
                    1j*(site_coherence-site_coherence.dag())]
    # put operators into full RC tensor product basis
    fullspace_ops = [tensor(op, I) for op in subspace_ops]
    # RC operators
    # RC positions, RC number state1, RC number state1, RC upper N fock, RC ground fock

    N_1, N_2, exc = PARS['N_1'], PARS['N_2'], PARS['exc']
    a_enr_ops = enr_destroy([N_1, N_2], exc)
    position1 = a_enr_ops[0].dag() + a_enr_ops[0]
    position2 = a_enr_ops[1].dag() + a_enr_ops[1]
    number1   = a_enr_ops[0].dag()*a_enr_ops[0]
    number2   = a_enr_ops[1].dag()*a_enr_ops[1]

    subspace_ops = [position1, position2, number1, number2]
    fullspace_ops += [tensor(I_sys, op) for op in subspace_ops]

    return dict((key_val[0], key_val[1]) for key_val in zip(labels, fullspace_ops))

def get_H_and_L(PARS,silent=False, threshold=0.):
    L_RC, H, A_1, A_2, PARAMS = RC.RC_mapping(PARS, silent=silent, shift=True)

    N_1 = PARS['N_1']
    N_2 = PARS['N_2']
    exc = PARS['exc']
    mu = PARS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2

    L = L_RC
    del L_RC # free up some memory
    if abs(PARS['alpha_EM'])>0:
        if PARS['num_cpus']>1:
            L_EM_full = opt.L_non_rwa_par(H[1], tensor(sigma,I), PARS, silent=silent)
        else:
            L_EM_full = opt.L_non_rwa(H[1], tensor(sigma,I), PARS, silent=silent)
        L+=L_EM_full
        del L_EM_full
    else:
        print "Not including optical dissipator"
    if threshold: # Chop off tiny values if you like
        L.tidyup(threshold)
    return H, L

def PARAMS_setup(bias=100., w_2=2000., V = 100., pialpha_prop=0.1,
                                 T_EM=0., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, w_0=200, Gamma=50., N=3,
                                 silent=False, exc_diff=0, sys_dim=3):
    # Sets up the parameter dict
    N_1 = N_2 = N
    exc = (2*N)-exc_diff
    gap = sqrt(bias**2 +4*(V**2))
    phonon_energy = T_ph*0.695

    alpha = w_2*pialpha_prop/pi

    w_1 = w_2 + bias
    dipole_1, dipole_2 = 1., 1.
    mu = (w_2*dipole_2)/(w_1*dipole_1)
    sigma = sigma_m1 + mu*sigma_m2
    T_1, T_2 = T_ph, T_ph # Phonon bath temperature

    Gamma_1 = Gamma_2 = Gamma
    w0_2, w0_1 = w_0, w_0 # underdamped SD parameter omega_0
    if not silent:
        plot_UD_SD(Gamma_1, w_2*pialpha_prop/pi, w_0, eps=w_2)
    w_xx = w_2 + w_1
    if not silent:
        print("Gap is {}. Phonon thermal energy is {}. Phonon SD peak is {}. N={}.".format(gap, phonon_energy, 
        SD_peak_position(Gamma, 1, w_0), N))

    J = J_minimal
    H_sub = w_1*XO_proj + w_2*OX_proj + V*(site_coherence+site_coherence.dag())
    coupling_ops = [XO_proj, OX_proj]
    PARAM_names = ['H_sub', 'coupling_ops', 'w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2',
                   'w0_1', 'w0_2', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J',
                   'dipole_1','dipole_2', 'Gamma_1', 'Gamma_2']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)

    PARAMS.update({'alpha_1': alpha, 'alpha_2': alpha})
    PARAMS.update({'N_1': N_1, 'N_2': N_2, 'exc': exc})
    PARAMS.update({'sys_dim' : sys_dim})
    return PARAMS
