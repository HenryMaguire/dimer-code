from dimer_setup import *
from utils import *

from scipy.sparse import csc_matrix
import time
from heatmap_setup import calculate_steadystate
import numpy as np


w_2 = 1.4*ev_to_inv_cm
bias = 0.01*ev_to_inv_cm #0.0000001*ev_to_inv_cm
V = 0.01*ev_to_inv_cm #0.00001*ev_to_inv_cm
alpha = 100./pi
alpha_EM = 5.309e-3 # inv_ps_to_inv_cm *10^-3
N =4
wc = 100.
w_0 = 200.
site_basis = True
Gamma = (w_0**2)/wc
PARAMS = PARAMS_setup(bias=bias, w_2=w_2, 
                      V = V, alpha=alpha,
                      T_EM=6000., T_ph =300., alpha_EM=alpha_EM, shift=True,
                      num_cpus=4, N=N, Gamma=Gamma, w_0=w_0,
                      silent=True, exc_diff=0)

H, L = get_H_and_L(PARAMS, silent=False, threshold=1e-7, site_basis=True)
#print H[1].eigenenergies()[::-1]
L_total = qt.liouvillian(H[1], [L])
print(nonzero_elements(L_total), total_elements(L_total), sparse_percentage(L_total))

ssL, info = calculate_steadystate(H, L, method="power", persistent=True)
exps = make_expectation_operators(H, PARAMS, site_basis=site_basis)

ss_therm = thermal_state(PARAMS["T_EM"], H[1])
for key, op in exps.items():
    if 'RC' not in key:
        res, therm = (ssL*op).tr().real, (ss_therm*op).tr()
        print "{} : {} \t | \t {} \t | \t {}".format(key, res, therm, abs(res - therm))

#print sum([(ssL*exps[key]).tr().real for key in ['OO', 'OX', 'XO', 'XX']])