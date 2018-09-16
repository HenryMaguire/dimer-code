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
import tests as check

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



def dynamics(bias=100., w_2=2000., V = 100., pialpha_prop=0.1,
                                 T_EM=0., T_ph =300.,
                                 alpha_EM=1., shift=True,
                                 num_cpus=1, w_0=200, Gamma=50.,
                                dir_name="dynamics", N=3,
                                tf=0.5, incs=2500, init_dimer='XO', silent=False):
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
    if not silent:
        plot_UD_SD(Gamma_1, w_2*pialpha_prop/pi, w_0, eps=w_2)
    w_xx = w_2 + w_1
    if not silent:
        print "Gap is {}. Phonon thermal energy is {}. Phonon SD peak is {}. N={}.".format(gap, phonon_energy,
                                                                                 SD_peak_position(Gamma, 1, w_0),
                                                                                           N)

    J = J_minimal

    PARAM_names = ['w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2',
                   'w0_1', 'w0_2', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J',
                   'dipole_1','dipole_2', 'Gamma_1', 'Gamma_2']
    scope = locals() # Lets eval below use local variables, not global
    PARAMS = dict((name, eval(name, scope)) for name in PARAM_names)

    if N>7:
        silent = False

    PARAMS.update({'alpha_1': alpha, 'alpha_2': alpha})
    PARAMS.update({'N_1': N_1, 'N_2': N_2})
    PARAMS.update({'exc': exc})

    I = enr_identity([N_1,N_2], exc)
    ''' generate the RC liouvillian '''

    # Create the phonon liouvillian

    L_RC, H, A_1, A_2, SIG_1, SIG_2, PARAMS = RC.RC_mapping(PARAMS,
                                                            silent=silent,
                                                            shift=True)
    L = L_RC
    if abs(alpha_EM)>0:
        # electromagnetic bath liouvillians
        L_EM_full = opt.L_non_rwa(H[1], tensor(sigma,I), PARAMS, silent=silent)
        L+=L_EM_full
    timelist = np.linspace(0, tf, incs)
    # get all the useful operators we might need
    eops_dict =  make_expectation_operators(PARAMS)
    # unpack them all in asensible order
    labels = ['OO', 'XO', 'OX', 'XX', 'site_coherence',
              'bright', 'dark', 'eig_coherence',
              'RC1_position1' , 'RC2_position',
              'RC1_number'    , 'RC2_number']
    eops  = [eops_dict[l] for l in labels]
    rho_0 = make_initial_state(init_dimer, eops_dict, PARAMS)
    try:
        pb = not silent
        if not pb:
            pb = None
        data  = qt.mesolve(H[1], rho_0, timelist, c_ops=L, e_ops=eops,
                                                progress_bar=pb)

        # Save the data
        j = 1
        saved = False
        while not saved:
            directory = 'DATA/'+dir_name+'_'+str(j)
            if not os.path.exists(directory):
                os.makedirs(directory)
                save_obj(data, directory+'/dynamics')
                save_obj(PARAMS, directory+'/PARAMS')
                save_obj(timelist, directory+'/timelist')
                saved = True
                print "Files saved at {}".format(directory)
            else:
                j+=1
        return timelist, data
    except Exception as err:
        print ("omega_0 = {}, Gamma = {} and alpha = {} didn't work because").format(w_0, Gamma, alpha)
        print (err)
        return timelist, None


from itertools import combinations_with_replacement

def permutations_with_replacement(e):
    # iterator to make all permutations of parameters
    for i in e:
        for j in e:
            for k in e:
                yield (k,j,i)

sizes = ['s', 'm', 'l']
combs = []

init_dimer='OO'
w_2 = 2000.

params = dict({'Gamma': dict({'s' : 30, 'm' : 200, 'l' : 1500}),
             'w_0'  : dict({'s' : 50, 'm' : 200, 'l' : 1500}),
                'alpha': dict({'s' : 10, 'm' : 50 , 'l' : 200})})
# have already calculated the below parameters
combs = [i for i in combinations_with_replacement(sizes, 3)]

N_values = range(3,10)
for perm in permutations_with_replacement(sizes):
    if perm not in combs:
        # Gamma, omega_0, alpha
        Gamma = params['Gamma'][perm[0]]
        w_0 = params['w_0'][perm[1]]
        alpha = params['alpha'][perm[2]]
        pialpha_prop = (pi*alpha)/w_2
        label_str = ''.join(perm)
        for N in N_values:
            _, _ = dynamics(N=N, w_0=w_0, Gamma=Gamma, pialpha_prop=pialpha_prop,
                            alpha_EM=1., T_EM=2000.,  silent = True,
                            init_dimer=init_dimer, dir_name='dynamics_{}'.format(label_str))
