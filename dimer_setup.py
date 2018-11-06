import os
import time

import numpy as np
from numpy import pi, sqrt
import matplotlib.pyplot as plt


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


"""
fock_N1      = qt.enr_fock([N,N],N, (N/2,N/2))
fock_N2      = qt.enr_fock([N,N],N, (0,N))
fock_ground = qt.enr_fock([N,N],N, (N,0))
fock_ground2 = qt.enr_fock([N,N],N, (0,N))
"""

labels = [ 'OO', 'XO', 'OX', 'XX', 'site_coherence', 'bright', 'dark', 'eig_coherence',
             'RC1_position1', 'RC2_position', 'RC1_number', 'RC2_number', 'sigma_x', 'sigma_y']

def make_expectation_operators(H, PARS, site_basis=True):
    # makes a dict: keys are names of observables values are operators
    I = enr_identity([PARS['N_1'], PARS['N_2']], PARS['exc'])
    I_dimer = qeye(PARS['sys_dim'])
    energies, states = exciton_states(PARS, shift=False)
    bright_vec = states[1]
    dark_vec = states[0]
    # electronic operators
     # site populations site coherences, eig pops, eig cohs
    subspace_ops = [OO_proj, XO_proj, OX_proj, XX_proj,site_coherence,
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
    fullspace_ops += [tensor(I_dimer, op) for op in subspace_ops]

    if not site_basis:
        print "THIS ONE IS EIG BASIS"
        eVals, eVecs = H[1].eigenstates()
        eVecs = np.transpose(np.array([v.dag().full()[0] for v in eVecs])) # get into columns of evecs
        eVecs_inv = sp.linalg.inv(eVecs) # has a very low overhead
        for j, op in enumerate(fullspace_ops):
            fullspace_ops[j] = to_eigenbasis(op, eVals, eVecs, eVecs_inv)

    return dict((key_val[0], key_val[1]) for key_val in zip(labels, fullspace_ops))

def get_H_and_L(PARS,silent=False, threshold=0., site_basis=True):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARS,silent=silent, shift=True, site_basis=site_basis)

    N_1 = PARS['N_1']
    N_2 = PARS['N_2']
    exc = PARS['exc']
    mu = PARS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2

    if abs(PARS['alpha_EM'])>0:
        if PARS['num_cpus']>1:
            #L += opt.L_non_rwa_par(H[1], tensor(sigma,I), PARS, silent=silent, site_basis=site_basis)
            L += opt.L_nonsecular_par(H[1], tensor(sigma,I), PARS, silent=silent, site_basis=site_basis)
        else:
            #L += opt.L_non_rwa(H[1], tensor(sigma,I), PARS, silent=silent, site_basis=site_basis)
            L += opt.L_nonsecular(H[1], tensor(sigma,I), PARS, silent=silent, site_basis=site_basis)

    else:
        print "Not including optical dissipator"
    if threshold:
        L.tidyup(threshold)
    return H, L


def get_L(PARS,silent=False, threshold=0., site_basis=True):
    L, H, A_1, A_2, PARAMS = RC.RC_mapping(PARS, silent=silent, shift=True,
                                            site_basis=site_basis)

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


def PARAMS_setup(bias=100., w_2=2000., V = 100., pialpha_prop=0.1,
                                 T_EM=0., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, w_0=200, Gamma=50., N=3,
                                 silent=False, exc_diff=0):
    N_1 = N_2 = N
    exc = 2*N-exc_diff
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
    H_sub = w_1*sigma_m1.dag()*sigma_m1 + w_2*sigma_m2.dag()*sigma_m2 + V*(site_coherence+site_coherence.dag())
    coupling_ops = [sigma_m1.dag()*sigma_m1, sigma_m2.dag()*sigma_m2] # system-RC operators
    if not silent:
        print "Gap is {}. Phonon thermal energy is {}. Phonon SD peak is {}. N={}.".format(gap, phonon_energy,
                                                                                 SD_peak_position(Gamma, 1, w_0),
                                                                                           N)

    J = J_minimal

    PARAM_names = ['H_sub', 'coupling_ops', 'w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2',
                   'w0_1', 'w0_2', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J',
                   'dipole_1','dipole_2', 'Gamma_1', 'Gamma_2']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)

    PARAMS.update({'alpha_1': alpha, 'alpha_2': alpha})
    PARAMS.update({'N_1': N_1, 'N_2': N_2, 'exc': exc})
    PARAMS.update({'sys_dim' : 4})
    return PARAMS
