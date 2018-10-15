from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt
import numpy as np


def total_elements(qobj):
    return qobj.shape[0]**2

def nonzero_elements(qobj):
    return len(csc_matrix(qobj.data).nonzero()[0])

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
