
# make qutip available in the rest of the notebook
from qutip import *

#import the functions from numpy, a numerical array handler
from numpy import matrix
from numpy import linalg
import numpy as np

import scipy.sparse as sp


def _enr_state_enumerate(dims, excitations, state=None, idx=0):


    if state is None:
        state = [0] * len(dims)

    if idx == len(dims):
        if sum(state) <= excitations:
            yield tuple(state)
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in _enr_state_enumerate(dims, excitations, state, idx + 1):
                yield s

def _enr_state_dictionaries(dims, excitations):

    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in _enr_state_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1

    return nstates, state2idx, idx2state

def enr_unit(dims, excitations):
    nstates, state2idx, idx2state = _enr_state_dictionaries(dims, excitations)

    u_ops=  sp.lil_matrix((nstates, nstates), dtype=np.complex)


    for n1, state1 in idx2state.items():
        for n2, state2 in idx2state.items():


            if (state1 == state2):
                u_ops[n1, n2] = 1

    return Qobj(u_ops, dims=[dims, dims])


def enr_destroy(dims, excitations):
    nstates, state2idx, idx2state = _enr_state_dictionaries(dims, excitations)

    a_ops = [sp.lil_matrix((nstates, nstates), dtype=np.complex) for _ in range(len(dims))]

    for n1, state1 in idx2state.items():
        for n2, state2 in idx2state.items():
            for idx, a in enumerate(a_ops):
                s1 = [s for idx1, s in enumerate(state1) if idx != idx1]
                s2 = [s for idx2, s in enumerate(state2) if idx != idx2]
                if (state1[idx] == state2[idx] - 1) and (s1 == s2):
                    a[n1, n2] = np.sqrt(state2[idx])

    return [Qobj(a, dims=[dims, dims]) for a in a_ops]

def enr_fock(dims, excitations, state):
    nstates, state2idx, idx2state = _enr_state_dictionaries(dims, excitations)

    data = sp.lil_matrix((nstates, 1), dtype=np.complex)

    try:
        data[state2idx[tuple(state)], 0] = 1
    except:
        raise ValueError("The state tuple %s is not in the restricted state space" % str(tuple(state)))

    return Qobj(data, dims=[dims, 1])




def enr_therm(dims, excitations, nnn):
    nstates, state2idx, idx2state = _enr_state_dictionaries(dims, excitations)
    a_ops = sp.lil_matrix((nstates, nstates), dtype=np.complex)
    i = np.array(range(excitations+1))
    beta = np.log(1.0 / nnn + 1.0)
    diags = np.exp(-beta * i)
    diags = diags / np.sum(diags)
            # populates diagonal terms using truncated operator expression
   # rm = sp.spdiags(diags, 0, excitations, excitations, format='csr')

    for n1, state1 in idx2state.items():
        for n2, state2 in idx2state.items():
            #for idx in range(len(dims)):#, a in enumerate(a_ops):
                #this checks that the non-chosen mode states are diagonal
            #    s1 = [s for idx1, s in enumerate(state1) if idx != idx1]
            #    s2 = [s for idx2, s in enumerate(state2) if idx != idx2]
                #this checks whether chosen mode state is diagonal,
            if (state1 == state2 ):
                #print (state2[0])
                a_ops[n1, n2]=diags[state2[0]]
                for idx in range(1,len(dims)):
                    a_ops[n1, n2] =  a_ops[n1, n2]*diags[state2[idx]]

    return Qobj(a_ops, dims=[dims, dims])




#  these functions are modified to work with enr states.   the latest qutip contains these fixes already, but just in case here they are.

def _drop_projected_dims(dims):
    """
    Eliminate subsystems that has been collapsed to only one state due to
    a projection.
    """
    return [d for d in dims if d != 1]


def sprepost(A, B):

    import scipy.sparse as sp
    """Superoperator formed from pre-multiplication by operator A and post-
    multiplication of operator B.
    Parameters
    ----------
    A : Qobj
        Quantum operator for pre-multiplication.
    B : Qobj
        Quantum operator for post-multiplication.
    Returns
    --------
    super : Qobj
        Superoperator formed from input quantum objects.
    """

    dims = [[_drop_projected_dims(A.dims[0]), _drop_projected_dims(B.dims[1])],
            [_drop_projected_dims(A.dims[1]), _drop_projected_dims(B.dims[0])]]
    data = sp.kron(B.data.T, A.data, format='csr')
    return Qobj(data, dims=dims, superrep='super')




def spost(A):
    """Superoperator formed from post-multiplication by operator A

    Parameters
    ----------
    A : qobj
        Quantum operator for post multiplication.

    Returns
    -------
    super : qobj
        Superoperator formed from input qauntum object.
    """
    if not isinstance(A, Qobj):
        raise TypeError('Input is not a quantum object')

    if not A.isoper:
        raise TypeError('Input is not a quantum operator')

    S = Qobj(isherm=A.isherm, superrep='super')
    S.dims = [[A.dims[0], A.dims[1]], [A.dims[0], A.dims[1]]]
    S.data = sp.kron(A.data.T, sp.identity(np.prod(A.shape[0])), format='csr')
    return S


def spre(A):
    """Superoperator formed from pre-multiplication by operator A.

    Parameters
    ----------
    A : qobj
        Quantum operator for pre-multiplication.

    Returns
    --------
    super :qobj
        Superoperator formed from input quantum object.
    """
    if not isinstance(A, Qobj):
        raise TypeError('Input is not a quantum object')

    if not A.isoper:
        raise TypeError('Input is not a quantum operator')

    S = Qobj(isherm=A.isherm, superrep='super')
    S.dims = [[A.dims[0], A.dims[1]], [A.dims[0], A.dims[1]]]
    S.data = sp.kron(sp.identity(np.prod(A.shape[1])), A.data, format='csr')
    return S
