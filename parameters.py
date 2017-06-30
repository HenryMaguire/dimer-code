from utils import *

w_2 = 1.4*ev_to_inv_cm
bias = 0.0*ev_to_inv_cm
w_1 = w_2 + bias
V = 0.25*92. #0.1*8065.5
dipole_1, dipole_2 = 1., 1.
T_EM = 6000. # Optical bath temperature
alpha_EM = 0.9*inv_ps_to_inv_cm # Optical S-bath strength (from inv. ps to inv. cm)(larger than a real decay rate because dynamics are more efficient this way)
mu = w_2*dipole_2/w_1*dipole_1

T_1, T_2 = 300., 300. # Phonon bath temperature

wc = 1*53. # Ind.-Boson frame phonon cutoff freq
w0_2, w0_1 = 400., 400. # underdamped SD parameter omega_0
w_xx = w_2 + w_1
alpha_1, alpha_2 = 0, 0 # Ind.-Boson frame coupling
N_1, N_2 = 6,6 # set Hilbert space sizes
exc = int((N_1+N_2)*0.6)
num_cpus = 4
J = J_minimal

H_dim = w_1*XO*XO.dag() + w_2*OX*OX.dag() + w_xx*XX*XX.dag() + V*(XO*OX.dag() + OX*XO.dag())
PARAM_names = ['w_1', 'w_2', 'V', 'bias', 'w_xx', 'T_1', 'T_2', 'wc',
                'w0_1', 'w0_2', 'alpha_1', 'alpha_2', 'N_1', 'N_2', 'exc', 'T_EM', 'alpha_EM','mu', 'num_cpus', 'J']
PARAMS = dict((name, eval(name)) for name in PARAM_names)


params = []

params.append()
