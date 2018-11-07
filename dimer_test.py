from dimer_setup import *
from utils import *

from scipy.sparse import csc_matrix
import time
from heatmap_setup import calculate_steadystate
import numpy as np


w_2 = 1.4*ev_to_inv_cm
bias = 0.000001*ev_to_inv_cm #0.0000001*ev_to_inv_cm
V = 0.00000001*ev_to_inv_cm #0.00001*ev_to_inv_cm
alpha = 0./pi
alpha_EM = 5.309e-3 # inv_ps_to_inv_cm *10^-3
N =4
pap = alpha_to_pialpha_prop(alpha, w_2)
wc = 100.
w_0 = 200.
site_basis = True
Gamma = (w_0**2)/wc
PARAMS = PARAMS_setup(bias=bias, w_2=w_2, 
                      V = V, pialpha_prop=pap,
                      T_EM=6000., T_ph =300., alpha_EM=alpha_EM, shift=True,
                      num_cpus=4, N=N, Gamma=Gamma, w_0=w_0,
                      silent=True, exc_diff=0)

H, L = get_H_and_L(PARAMS, silent=False, threshold=1e-12, site_basis=site_basis)


ti = time.time()
ssL, info = calculate_steadystate(H, L, method="power", persistent=True)
exps = make_expectation_operators(H, PARAMS, site_basis=site_basis)
exps_site = make_expectation_operators(H, PARAMS, site_basis=True)
ss_therm = thermal_state(PARAMS["T_EM"], H[1])
for key, op in exps.items():
    if 'RC' not in key:
        print "{} : {} \t | \t {} \t | \t {}".format(key, (ssL*op).tr().real, (ss_therm*exps_site[key]).tr().real, abs((ssL*op).tr() -(ss_therm*exps_site[key]).tr()))
print( time.time()-ti)
print sum([(ssL*exps[key]).tr().real for key in ['OO', 'OX', 'XO', 'XX']])