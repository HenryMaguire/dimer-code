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

def permutations_with_replacement(e):
    # needed for parameter sweep later
    for i in e:
        for j in e:
            for k in e:
                yield (i,j,k)

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

I_ses = qeye(3)

reload(RC)
reload(opt)
# Change to the directory which contains the current script
"""dirFile = os.path.dirname(os.path.join('/Users/henrymaguire/Work',
                          'coherence_analysis.ipynb'))
# Load style file
plt.style.use(os.path.join(dirFile, 'plot_style.mplstyle'))
# Make some style choices for plotting
colourWheel =['#329932',
            '#ff6961',
            'b',
            '#6a3d9a',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#d6604d',
            '#f4a582',
            '#fddbc7',
            '#f7f7f7',
            '#d1e5f0',
            '#92c5de',
            '#4393c3',
            '#2166ac',
            '#053061']



plt.style.use('ggplot')
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = "0.15"
plt.rcParams["axes.linewidth"]  = 1.5
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 13
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'medium'

colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
colors+=colors"""




def plot_UD_SD(Gamma, alpha, w_0, eps=2000., ax=None):
    Omega = np.linspace(0,eps,10000)
    J_w = np.array([J_underdamped(w, alpha, Gamma, w_0) for w in Omega])
    show_im = ax
    if ax is None:
        f, ax = plt.subplots(1,1)
    ax.plot(Omega, J_w)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$J(\omega)$")
    if show_im is None:
        plt.show()

def plot_UD_SD_PARAMS(PARAMS, ax=None):
    eps = PARAMS['w_2']
    alpha = PARAMS['alpha_2']
    w_0 = PARAMS['w0_2']
    Omega = np.linspace(0,eps,10000)
    J_w = np.array([J_underdamped(w, alpha, Gamma, w_0) for w in Omega])
    show_im = ax
    if ax is None:
        f, ax = plt.subplots(1,1)
    ax.plot(Omega, J_w)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$J(\omega)$")
    if show_im is None:
        plt.show()

def SD_peak_position(Gamma, alpha, w_0):
    Omega = np.linspace(0,w_0*50,10000)
    J_w = np.array([J_underdamped(w, alpha, Gamma, w_0) for w in Omega])
    return Omega[np.argmax(J_w)]


def print_PARAMS(PARAMS):
    keys = ['y_values', 'x_values',
            'y_axis_parameters', 'x_axis_parameters']
    try:
        keys+= list(PARAMS['y_axis_parameters'])
        keys+= list(PARAMS['x_axis_parameters'])
    except KeyError:
        pass
    not_useful = np.concatenate((keys, ['J', 'num_cpus']))

    param_strings = []
    for key in PARAMS.keys():
        try:
            if key not in not_useful:
                param_strings.append("{}={:0.2f}".format(key, PARAMS[key]))
        except KeyError:
            pass
    print(", ".join(param_strings))


# conversions between alphas and the ratios in terms of w_2
def alpha_to_pialpha_prop(alpha, w_2):
    return pi*alpha/w_2

def pialpha_prop_to_alpha(pialpha_prop, w_2):
    return pialpha_prop*w_2/pi

assert 0.1 == alpha_to_pialpha_prop(100/pi, 1000.)
assert 100 == pialpha_prop_to_alpha(0.1, 1000.*pi)

def make_initial_state(init_dimer_str, eops_dict, PARS):
    I_dimer = qeye(4)
    # Should also displace these states
    n1 = Occupation(PARS['w0_1'], PARS['T_1'])
    n2 = Occupation(PARS['w0_2'], PARS['T_2'])
    therm = tensor(I_dimer, qt.enr_thermal_dm([PARS['N_1'], PARS['N_2']], PARS['exc'], n1))
    return eops_dict[init_dimer_str]*therm

"""
fock_N1      = qt.enr_fock([N,N],N, (N/2,N/2))
fock_N2      = qt.enr_fock([N,N],N, (0,N))
fock_ground = qt.enr_fock([N,N],N, (N,0))
fock_ground2 = qt.enr_fock([N,N],N, (0,N))
"""

labels = [ 'OO', 'XO', 'OX', 'XX', 'site_coherence', 'bright', 'dark', 'eig_coherence',
             'RC1_position1', 'RC2_position', 'RC1_number', 'RC2_number', 'sigma_x', 'sigma_y']

def make_expectation_operators(PARS):
    # makes a dict: keys are names of observables values are operators
    I = enr_identity([PARS['N_1'], PARS['N_2']], PARS['exc'])
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
    fullspace_ops += [tensor(I_ses, op) for op in subspace_ops]


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
    del L_RC
    if abs(PARS['alpha_EM'])>0:
        if PARS['num_cpus']>1:
            L_EM_full = opt.L_non_rwa_par(H[1], tensor(sigma,I), PARS, silent=silent)
        else:
            L_EM_full = opt.L_non_rwa(H[1], tensor(sigma,I), PARS, silent=silent)
        L+=L_EM_full
        del L_EM_full
    else:
        print "Not including optical dissipator"
    if threshold:
        L.tidyup(threshold)
    return H, L

def PARAMS_setup(bias=100., w_2=2000., V = 100., pialpha_prop=0.1,
                                 T_EM=0., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, w_0=200, Gamma=50., N=3,
                                 silent=False, exc_diff=0):
    N_1 = N_2 = N
    exc = N+exc_diff
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
        print "Gap is {}. Phonon thermal energy is {}. Phonon SD peak is {}. N={}.".format(gap, phonon_energy,
                                                                                 SD_peak_position(Gamma, 1, w_0),
                                                                                           N)

    J = J_minimal
    H_sub = w_1*XO_proj + w_2*OX_proj + V*(site_coherence+site_coherence.dag())
    coupling_ops = [XO_proj, OX_proj]
    PARAM_names = ['H_sub', 'coupling_ops', 'w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2',
                   'w0_1', 'w0_2', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J',
                   'dipole_1','dipole_2', 'Gamma_1', 'Gamma_2']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)

    PARAMS.update({'alpha_1': alpha, 'alpha_2': alpha})
    PARAMS.update({'N_1': N_1, 'N_2': N_2})
    PARAMS.update({'exc': exc})
    return PARAMS
