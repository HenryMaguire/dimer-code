import numpy as np
from numpy import pi
import scipy as sp
from qutip import spre, spost, sprepost
import qutip as qt
import pickle

#new ptrace for ENR states.   rho is the state, sel is the same as the normal ptrace
#(list of which subsystems you want to keep),
#dims and excitations are the same as the ones you send to the other enr functions
def ENR_ptrace(rho,sel,dims,excitations):
    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)

    if (sel < 0).any() or (sel >= len(rho.dims[0])).any():
        raise TypeError("Invalid selection index in ptrace.")

    ############
    #dimensions
    ####################
    #enr stuff for the original state
    nstates, state2idx, idx2state = enr_state_dictionaries(dims, excitations)



    ################
    #number of states in selection
    ######################
    #
    drho=rho.dims[0]
    dims_short= np.asarray(drho).take(sel)
    nstates2, state2idx2, idx2state2 = enr_state_dictionaries(dims_short.tolist(), excitations)


    # this is a list of the dimensions of the system one has traced out
    rest = np.setdiff1d(np.arange(len(drho)), sel)

    #rest_short= np.asarray(drho).take(rest)
    #nstates3, state2idx3, idx2state3 = enr_state_dictionaries(rest_short.tolist(), excitations)

    #construct matrix to return the new Density matrix
    rhout = np.zeros((nstates2,nstates2),dtype=np.complex64)

    for ind,state in idx2state.items():
        for ind2,state2 in idx2state.items():
            #if the parts of the states of the systems(s) being traced out are diagonal, add this to the new DM
            if  np.all(np.asarray(state).take(rest) == np.asarray(state2).take(rest)):

                rhout[state2idx2[tuple(np.asarray(state).take(sel))],
                      state2idx2[tuple(np.asarray(state2).take(sel))]] += rho.data[state2idx[state],state2idx[state2]]


    dims_kept0 = np.asarray(rho.dims[0]).take(sel)
    dims_kept1 = np.asarray(rho.dims[0]).take(sel)
    rho1_dims = [dims_kept0.tolist(), dims_kept1.tolist()]
    rho1_shape = [nstates2, nstates2]

    return Qobj(rhout,rho1_dims,rho1_shape)

def dimer_mutual_information(rho, args):
    N, exc = args['N_1'], args['exc']
    sel = [1,2]
    vn12 = qt.entropy_vn(ENR_ptrace(rho,sel,[4,N,N],exc))
    vn1 = qt.entropy_vn(ENR_ptrace(rho,1,[4,N,N],exc))
    vn2 = qt.entropy_vn(ENR_ptrace(rho,2,[4,N,N],exc))
    return vn1+ vn2 -vn12

ev_to_inv_cm = 8065.5
inv_ps_to_inv_cm = 5.309
def load_obj(name ):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def beta_f(T):
    conversion = 0.695
    beta = 0
    if T ==0.: # First calculate beta
        beta = np.infty
    else:
        # no occupation yet, make sure it converges
        beta = 1. / (conversion*T)
    return beta

def Occupation(omega, T):
    conversion = 0.695
    """
    if time_units == 'ev':
        conversion == 8.617E-5
    if time_units == 'ps':
        conversion == 0.131
    else:
        pass
    """
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
    return Gamma/2*np.pi
def J_overdamped(omega, alpha, Gamma, omega_0):
    return alpha*Gamma*pow(omega_0,2)*omega/(pow(pow(omega_0,2)-pow(omega,2),2)+(Gamma**2 *omega**2))

def rate_up(w, T, gamma, J, w_0):
    n = Occupation(w, T)
    rate = 0.5 * pi * n * J(w, gamma, w_0)
    return rate

def rate_down(w, T, gamma, J, w_0):
    n = Occupation(w, T)
    rate = 0.5 * pi * (n + 1. ) * J(w, gamma, w_0)
    return rate

def lin_construct(O):
    Od = O.dag()
    L = 2. * spre(O) * spost(Od) - spre(Od * O) - spost(Od * O)
    return L
