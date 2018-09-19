import os
import time

import numpy as np
from numpy import pi, sqrt
import matplotlib
matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt


import phonons as RC
import optical as opt

from phonons import RC_mapping
from optical import L_non_rwa, L_phenom
from qutip import basis, qeye, enr_identity, enr_destroy, tensor, enr_thermal_dm, steadystate
from utils import *
import matplotlib.pyplot as plt
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


def PARAMS_setup(bias=100., w_2=2000., V = 100., pialpha_prop=0.1,
                                 T_EM=0., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, w_0=200, Gamma=50., N=3,
                                 silent=False):
    N_1 = N_2 = N
    exc = N
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

    w_xx = w_2 + w_1

    J = J_minimal

    PARAM_names = ['w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2',
                   'w0_1', 'w0_2', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J',
                   'dipole_1','dipole_2', 'Gamma_1', 'Gamma_2']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)

    PARAMS.update({'alpha_1': alpha, 'alpha_2': alpha})
    PARAMS.update({'N_1': N_1, 'N_2': N_2})
    PARAMS.update({'exc': exc})
    return PARAMS
import pdb

def get_H_and_L(PARS,silent=False):
    L_RC, H, A_1, A_2, SIG_1, SIG_2, PARAMS = RC.RC_mapping(PARS,
                                                                silent=silent,
                                                                shift=True)

    N_1 = PARS['N_1']
    N_2 = PARS['N_2']
    exc = PARS['exc']
    mu = PARS['mu']

    I = enr_identity([N_1,N_2], exc)
    sigma = sigma_m1 + mu*sigma_m2

    L = L_RC
    if abs(PARS['alpha_EM'])>0:
        L_EM_full = opt.L_non_rwa(H[1], tensor(sigma,I), PARS, silent=silent)
        L+=L_EM_full
    else:
        print "Not including optical dissipator"
    return H, L

def heat_map_calculator(PARAMS,
                        x_axis_parameters=['w0_1', 'w0_2'],
                        y_axis_parameters=['Gamma_1', 'Gamma_2'],
                        x_values=[100., 200., 300.],
                        y_values=[70., 200., 600.],
                        dir_name='heatmap_oG'):
    info_array = np.zeros(( len(y_values), len(x_values)), dtype=dict)
    ss_array = np.zeros(( len(y_values), len(x_values)), dtype=qt.Qobj)
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            # scan over each combination of parameters and update the PARAMS dict
            # for each point on the grid
            for param_labels in zip(x_axis_parameters, y_axis_parameters):
                PARAMS.update({param_labels[0] : x, param_labels[1] : y})
            ti = time.time()
            H, L = get_H_and_L(PARAMS,silent=True)
            tf = time.time()
            ss = steadystate(H[1], [L], method='iterative-gmres',
                           use_precond=True, fill_factor=47,
                           drop_tol=1e-3, use_rcm=True, return_info=True,
                           tol=1e-9, maxiter=1100)
            print "Build time: {:0.3f} \t | \t Solution time: {:0.3f}".format(tf-ti, ss[1]['solution_time'])
            info_array[i][j] = ss[1]
            ss_array[i][j] = ss[0]
    j = 1
    info_array[0][0].update({'x_axis_parameters': x_axis_parameters,
                             'y_axis_parameters': y_axis_parameters,
                          'x_values': x_values, 'y_values': y_values})
    saved = False
    while not saved:
        directory = 'DATA/'+dir_name+'_'+str(j)
        if not os.path.exists(directory):
            os.makedirs(directory)
            save_obj(ss_array, directory+'/ss_array')
            save_obj(PARAMS, directory+'/PARAMS')
            save_obj(info_array, directory+'/info_array')
            saved = True
            print "Files saved at {}".format(directory)
        else:
            j+=1
    return ss_array, info_array

def steadystate_observable(ss_array, PARAMS, observable, size):
    obs_op = make_expectation_operators(PARAMS)[observable]

    ss_array.resize(1, size[0]*size[1])
    ss_array = ss_array[0]
    f = lambda x : (qt.Qobj(x)*obs_op).tr()
    data_array = map(f, ss_array)
    data_array = np.array(data_array).reshape(size).real
    return data_array


def get_heatmap(comb, observable, dir_name="dynamics"):
    #f, axes = plt.subplots(1,2, figsize=(20,4))

    labels = ['OO', 'XO', 'OX', 'XX', 'site_coherence',
                      'bright', 'dark', 'eig_coherence',
                      'RC1_position1', 'RC2_position',
                      'RC1_number', 'RC2_number', 'sigma_x', 'sigma_y']
    assert observable in labels
    idx = labels.index(observable)

    _dir_name = dir_name+"_"+''.join(comb)+"_"
    data_loaded = False
    #try:
    directory = 'DATA/'+_dir_name+str(1)
    print "trying ", directory
    PARAMS = load_obj(directory+'/PARAMS')
    omega_values=x_values=np.linspace(100., 400,5)
    Gamma_values=y_values=[70., 600.]
    ss_array, info_array = heat_map_calculator(PARAMS,
                                               x_values=x_values,
                                               y_values=y_values)
    data_array = steadystate_observable(ss_array, PARAMS, observable,
                                        (len(y_values), len(x_values)))

    plt.imshow(data_array)
    plt.show()

    colors = [c['color'] for c in plt.rcParams['axes.prop_cycle']]
    return ss_array, info_array

N = 2
# bias vs alpha
pars = PARAMS_setup(bias=100., w_2=2000., V = 100., pialpha_prop=0.1,
                                 T_EM=2000., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, N=N, Gamma=200., w_0=300,
                                 silent=False)
x_values=2000+np.linspace(30., 500,4)
y_values=np.array([10., 50., 150.])
ss_array, info_array= heat_map_calculator(pars,
                        x_axis_parameters=['w_1'],
                        y_axis_parameters=['alpha_1', 'alpha_2'],
                        x_values=x_values,
                        y_values=y_values,
                        dir_name='heatmap_epsAlpha')
plt.imshow(steadystate_observable(ss_array, pars, 'sigma_x',
                                  (len(y_values), len(x_values))))

pialpha_prop = 100./2000.
# Gamma vs w_0
pars = PARAMS_setup(bias=100., w_2=2000., V = 100., pialpha_prop=pialpha_prop,
                                 T_EM=2000., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, N=N,
                                 silent=False)
x_values=np.linspace(100., 400,5)
y_values=[70., 1000.]

ss_array, info_array = heat_map_calculator(pars,
                        x_axis_parameters=['w0_1', 'w0_2'],
                        y_axis_parameters=['Gamma_1', 'Gamma_2'],
                        x_values=x_values,
                        y_values=y_values,
                        dir_name='heatmap_oG')
plt.imshow(steadystate_observable(ss_array, pars, 'sigma_x',
                                  (len(y_values), len(x_values))))
# T_ph vs T_EM
pars = PARAMS_setup(bias=100., w_2=2000., V = 100., pialpha_prop=pialpha_prop,
                                 T_EM=0., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, N=N, Gamma=200., w_0=300,
                                 silent=False)

x_values=np.linspace(30., 500,4)
y_values=[300., 1000, 3000.]
ss_array, info_array= heat_map_calculator(pars,
                        x_axis_parameters=['T_1', 'T_2'],
                        y_axis_parameters=['T_EM'],
                        x_values=x_values,
                        y_values=y_values,
                        dir_name='heatmap_TphTEM')
plt.imshow(steadystate_observable(ss_array, pars, 'sigma_x',
                                  (len(y_values), len(x_values))))

plt.show()
