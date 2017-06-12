import numpy as np
import scipy as sp
from qutip import spre, spost, sprepost
import qutip as qt
import pickle

ev_to_inv_cm = 8065.5
inc_ps_to_inv_cm = 5.309
def load_obj(name ):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def Occupation(omega, T, time_units='cm'):
    conversion = 0.695
    if time_units == 'ev':
        conversion == 8.617E-5
    if time_units == 'ps':
        conversion == 0.131
    else:
        pass
    n =0.
    beta = 0.
    if T ==0.: # First calculate beta
        n = 0.
        beta = np.infty
    else:
        # no occupation yet, make sure it converges
        beta = 1. / (conversion*T)
        if sp.exp(omega*beta)-1 ==0.:
            n = 0.
        else:
            n = float(1./(sp.exp(omega*beta)-1))
    return n


def J_multipolar(omega, Gamma, omega_0):
    return Gamma*(omega**3)/(2*np.pi*(omega_0**3))

def J_minimal(omega, Gamma, omega_0):
    return Gamma*omega/(2*np.pi*omega_0)

def J_flat(omega, Gamma, omega_0):
    return Gamma

def rate_up(w, n, gamma, J, w_0):
    rate = 0.5 * pi * n * J(w, gamma, w_0)
    return rate

def rate_down(w, n, gamma, J, w_0):
    rate = 0.5 * pi * (n + 1. ) * J(w, gamma, w_0)
    return rate

def lin_construct(O):
    Od = O.dag()
    L = 2. * spre(O) * spost(Od) - spre(Od * O) - spost(Od * O)
    return L
