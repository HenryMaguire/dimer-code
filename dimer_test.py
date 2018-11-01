from dimer_setup import *

import matplotlib.pyplot as plt
from heatmap_setup import calculate_steadystate

w_2 = 1.4*ev_to_inv_cm
bias = 0.1*ev_to_inv_cm
V = 0.01*ev_to_inv_cm
alpha = 50/pi
alpha_EM = 5.309e-3
N = 5
pap = alpha_to_pialpha_prop(alpha, w_2)
wc = 100.
w_0 = 200.
Gamma = (w_0**2)/wc
PARAMS = PARAMS_setup(bias=bias, w_2=w_2, 
                      V = 0.1*ev_to_inv_cm, pialpha_prop=pap,
                      T_EM=6000., T_ph =300., alpha_EM=alpha_EM, shift=True,
                      num_cpus=4, N=N, Gamma=Gamma, w_0=w_0,
                      silent=True, exc_diff=0)

H, L = get_H_and_L(PARAMS, silent=False, threshold=1e-7, site_basis=True)

print("L has {} non-zero element in the site basis".format(nonzero_elements(L)))
ti = time.time()
ss1 = steadystate(H[1], [L], method="power")
print("SS took {:0.3f}s in site basis".format(time.time()- ti))



H, L = get_H_and_L(PARAMS, silent=False, threshold=1e-7, site_basis=False)
print("L has {} non-zero element in the eigenbasis".format(nonzero_elements(L)))
ss2 = steadystate(H[1], [L], method="power")
print("SS took {:0.3f}s in eigenbasis".format(time.time()- ti))


site_ops = make_expectation_operators(H, PARAMS, site_basis=True)
print ((ss1*site_ops['OO']).tr())
eig_ops = make_expectation_operators(H, PARAMS, site_basis=False)
print ((ss2*eig_ops['OO']).tr())