# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module to compute the normalized anisotropy and its non-zero invariants (II, III or eta, xi) from the Reynolds stresses tensor
@contact: b.fontgarcia@soton.ac.uk
"""

# Imports
import numpy as np

# Functions
def anisotropy_tensor(r):
    """
    Return the normalized anisotropy tensor of the Reynolds stresses (np.ndarray with shape (3,3,N,M)).
    Args:
        r: Reynolds stresses. np.ndarray with shape (3,3,N,M) where NxM is the field size of the components (2D field)
    """
    N = r[0,0].shape[0]
    M = r[0,0].shape[1]
    zeros = np.zeros((N, M))

    # Calc anisotropy tensor
    k = 0.5 * (r[0,0] + r[1,1] + r[2,2])  # TKE
    k_d_matrix = np.array([[k, zeros, zeros], [zeros, k, zeros], [zeros, zeros, k]])  # TKE*kronecker_delta matrix
    a = r - (2 / 3) * k_d_matrix  # Anisotropy tensor
    b = a / (2 * k)  # Normalized anisotropy tensor

    return b


def invariants(b):
    """
    Return the invariants of the normalized Reynolds stresses anisotropy tensor (np.ndarray with shape (N,M)).
    Args:
        b: normalized Reynolds stresses anisotropy tensor. np.ndarray with shape (3,3,N,M) where NxM is the field size
            of the components (2D field)
    """
    # Calc invariants
    # b11 = b[0, 0]
    # b12 = b[0, 1]
    # b13 = b[0, 2]
    # b21 = b[1, 0]
    # b22 = b[1, 1]
    # b23 = b[1, 2]
    # b31 = b[2, 0]
    # b32 = b[2, 1]
    # b33 = b[2, 2]
    # I = b11+b22+b33 = 0 # Definition of tr(b)
    # II = (b11*b33+b22*b33+b11*b22-b13*b31-b23*b32-b12*b21) # Definition of -0.5*tr(b**2)!
    # III = (b11*b22*b33+b21*b32*b13+b31*b12*b23-b13*b31*b22-b23*b32*b11-b12*b21*b33) # Definition of det(b)!

    I = np.trace(b) # tr(b) = 0
    II = -0.5 * np.trace(np.einsum('ijkl,jmkl->imkl', b, b)) # -0.5*tr(b**2)
    III = np.linalg.det(b.T).T # det(b)
    eta = np.sqrt(-1 / 3 * II)
    xi = np.cbrt(1 / 2 * III)

    return eta, xi