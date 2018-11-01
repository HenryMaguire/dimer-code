import numpy as np
from numpy import pi
import scipy as sp
from qutip import spre, spost, sprepost, tensor, basis, qeye
import qutip as qt
import pickle
import sympy


from scipy.linalg import eig, inv

def sorted_eig(_H):
    if type(_H)==qt.Qobj:
        _H = _H.full()
    eigenValues, eigenVectors = eig(_H)
    idx = eigenValues.argsort()[::-1]   
    return eigenValues[idx], eigenVectors[:,idx]

def to_eigenbasis(op, evals, evecs, evecs_inv):
    if type(op)==qt.Qobj:
        dims = op.dims
        op = op.full()
    else:
        raise ValueError("operator must be a Qobj")
    A = np.matmul(evecs_inv, op)
    return qt.Qobj(np.matmul(A, evecs), dims = dims)

def to_site_basis(op, evals, evecs, evecs_inv):
    if type(op)==qt.Qobj:
        dims = op.dims
        op = op.full()
    else:
        raise ValueError("operator must be a Qobj")
    A = np.matmul(evecs, op)
    return qt.Qobj(np.matmul(A, evecs_inv), dims = dims)


def Coth(x):
    return (np.exp(2*x)+1)/(np.exp(2*x)-1)

def exciton_states(PARS, shift=False):
    w_1, w_2, V, eps = PARS['w_1'], PARS['w_2'],PARS['V'], PARS['bias']
    if shift:
        try:
            w_1 += PARS['shift1']
            w_2 += PARS['shift2']
        except KeyError as e:
            print "No RC mapping performed yet."
            raise KeyError
    eps = (w_1-w_2)
    eta = np.sqrt(eps**2 + 4*V**2)
    lam_m = ((w_2+eps)+w_2-eta)*0.5
    lam_p = ((w_2+eps)+w_2+eta)*0.5
    v_p = qt.Qobj(np.array([0., np.sqrt(eta+eps), np.sqrt(eta-eps), 0.]))/np.sqrt(2*eta)
    v_m = qt.Qobj(np.array([0., np.sqrt(eta-eps), -np.sqrt(eta+eps), 0.]))/np.sqrt(2*eta)
    if PARS['sys_dim'] == 3:
        v_p.eliminate_states([3])
        v_m.eliminate_states([3])
    return [lam_m, lam_p], [v_m, v_p]

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

    #dimensions
    #enr stuff for the original state
    nstates, state2idx, idx2state = qt.enr_state_dictionaries(dims, excitations)

    #number of states in selection
    drho=rho.dims[0]
    dims_short= np.asarray(drho).take(sel)
    nstates2, state2idx2, idx2state2 = qt.enr_state_dictionaries(dims_short.tolist(), excitations)

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

    return qt.Qobj(rhout,rho1_dims,rho1_shape)

def dimer_mutual_information(rho, args):
    N, exc = args['N_1'], args['exc']
    vn12 = qt.entropy_vn(ENR_ptrace(rho,[1,2],[4,N,N],exc))
    vn1 = qt.entropy_vn(ENR_ptrace(rho,1,[4,N,N],exc))
    vn2 = qt.entropy_vn(ENR_ptrace(rho,2,[4,N,N],exc))
    vnd = qt.entropy_vn(ENR_ptrace(rho,0,[4,N,N],exc))
    vnd1 = qt.entropy_vn(ENR_ptrace(rho,[0,1],[4,N,N],exc))
    vnd2 = qt.entropy_vn(ENR_ptrace(rho,[0,2],[4,N,N],exc))
    return [vn1+ vn2 -vn12, vnd+ vn1 -vnd1, vnd+ vn2 -vnd2]

ev_to_inv_cm = 8065.5
inv_ps_to_inv_cm = 5.309
def load_obj(name ):
    with open(name + '.pickle', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def beta_f(T, conversion=0.695):
    conversion = 0.695
    beta = 0
    if T ==0.: # First calculate beta
        beta = np.infty
    else:
        # no occupation yet, make sure it converges
        beta = 1. / (conversion*T)
    return beta

def Occupation(omega, T, conversion=0.695):
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


def get_dimer_info(rho, I):
    # must be a density matrix
    e1e2 = tensor(basis(4,1)*basis(4,2).dag(), I)
    e2e1 = tensor(basis(4,2)*basis(4,1).dag(), I)
    OO = basis(4,0)
    XO = basis(4,1)
    OX = basis(4,2)
    XX = basis(4,3)
    OO = tensor(OO*OO.dag(), I)
    XO = tensor(XO*XO.dag(), I)
    OX = tensor(OX*OX.dag(), I)
    XX = tensor(XX*XX.dag(), I)

    g = (rho*OO).tr()
    e1 = (rho*XO).tr()
    e2 = (rho*OX).tr()

    e1e2 = (rho*e1e2).tr()
    e2e1 = (rho*e2e1).tr()
    print g
    print e1, e1e2
    xx = (rho*XX).tr()
    return qt.Qobj([[g.real, 0,0,0], [0, e1.real,e1e2.real,0],[0, e2e1.real,e2.real,0],[0, 0,0,xx.real]])#/(g+e1+e2+xx)




def J_multipolar(omega, Gamma, omega_0):
    return Gamma*(omega**3)/(2*np.pi*(omega_0**3))

def J_minimal(omega, Gamma, omega_0):
    return Gamma*omega/(omega_0*2*np.pi)
def J_minimal_hard(omega, Gamma, omega_0, cutoff):
    if omega >cutoff:
        return 0.
    else:
        return Gamma*omega/(omega_0) #2*np.pi*
def J_flat(omega, Gamma, omega_0):
    return Gamma#/(2*np.pi)

def J_underdamped(omega, alpha, Gamma, omega_0):
    return alpha*Gamma*pow(omega_0,2)*omega/(pow(pow(omega_0,2)-pow(omega,2),2)+(Gamma**2 *omega**2))

def J_overdamped(omega, alpha, wc):
    return alpha*wc*float(omega)/(omega**2 +wc**2)



def J_OD_to_UD(omega, gamma, Omega, kappa):
    # kappa is  referred to as lambda
    # in J. Chem. Phys. 144, 044110 (2016)
    n = 4*gamma*omega*(Omega**2)*(kappa**2)
    d1= (Omega**2-omega**2)**2
    d2 = (2*np.pi*gamma*Omega*omega)**2
    return n/ (d1 + d2)

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

def coth(x):
    return float(sympy.coth(x))






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
            'y_axis_parameters', 'x_axis_parameters', 
            'coupling_ops', 'H_sub']
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
        except ValueError as err:
            raise Exception("Cannot print parameters for "+key + " because "+str(err))
    print(", ".join(param_strings))


# conversions between alphas and the ratios in terms of w_2
def alpha_to_pialpha_prop(alpha, w_2):
    return pi*alpha/w_2

def pialpha_prop_to_alpha(pialpha_prop, w_2):
    return pialpha_prop*w_2/pi

assert 0.1 == alpha_to_pialpha_prop(100/pi, 1000.)
assert 100 == pialpha_prop_to_alpha(0.1, 1000.*pi)

def make_initial_state(init_sys_str, eops_dict, PARS):
    I_sys = qeye(PARS['sys_dim'])
    # Should also displace these states
    n1 = Occupation(PARS['w0_1'], PARS['T_1'])
    n2 = Occupation(PARS['w0_2'], PARS['T_2'])
    therm = tensor(I_sys, qt.enr_thermal_dm([PARS['N_1'], PARS['N_2']], PARS['exc'], n1))
    return eops_dict[init_sys_str]*therm

def permutations_with_replacement(e):
    # needed for parameter sweep later
    for i in e:
        for j in e:
            for k in e:
                yield (i,j,k)

# Sparse matrix utilities

from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt
import numpy as np

def get_csc(_X):
    return (csc_matrix(_X.data)).data

def total_elements(qobj):
    return qobj.shape[0]**2

def nonzero_elements(qobj):
    return len(csc_matrix(qobj.data).nonzero()[0])

def sparse_percentage(qobj):
    return 100*(nonzero_elements(qobj)/float(total_elements(qobj)))

def visualise_dense(qobj):
    plt.imshow(qobj.full().real)
    plt.show()

def visualise_sparse(qobj):
    plt.spy(qobj.data)
    plt.show()

import copy
def chop(X, threshold=1e-7):
    _X = copy.deepcopy(X)
    _X.tidyup(threshold)
    return _X
