# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module containing functions to calculate derivatives, averages, decompositions, vorticity.
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np
import warnings

# Functions
def avg_z(u):
    """
    Return the span-wise spatial average of a three-dimensional field.
    If the passed u is spanwise periodic use trapz()/(n-1). Otherwise mean().
    :param u: The field to be spanwise-averaged.
    :return: Return the span-wise spatial average of a three-dimensional field.
    """
    if not len(u.shape)==3:
        warnings.warn("Field not 3D. Returning same array.")
        return u
    else:
        if np.array_equal(u[..., 0], u[..., -1]): # Periodic on last axis
            return np.trapz(u, axis=2)/(u.shape[2]-1)
        else:
            return u.mean(axis=2)


def make_periodic(u, **kwargs): # Methods t = True (add layer), t = False (substitue last layer with 1st layer)
    add = kwargs.get('add', True)
    if add:
        u_temp = np.zeros((u.shape[0], u.shape[1], u.shape[2] + 1))
        u_temp[..., :-1], u_temp[..., -1] = u, u[..., 0]
        u = u_temp
    else:
        u[..., -1] = u[..., 0]
    return u


def decomp_z(u):
    """
    :param u: field to be decomposed in z direction.
    :return: the average and the fluctuating parts of a three-dimensional field spatially averaged
    in the span-wise direction.
    """
    if not len(u.shape)==3:
        raise ValueError("Fields must be three-dimensional")
    else:
        u_avg = avg_z(u)
        u_f = u-u_avg[:, :, None]
        return u_avg, u_f


def ddx(u, x=None):
    """
    :param u: n-dimensional field.
    :return: the first-order derivative in the i direction of (n>=1 dimensional) field.
    """
    if x is not None:
        return np.gradient(u, x, axis=0, edge_order=2)
    else:
        return np.gradient(u, axis=0, edge_order=2)


def ddy(u, y=None):
    """
    :param u: n-dimensional field.
    :return: the first-order derivative in the j direction of (n>=2 dimensional) field.
    """
    if y is not None:
        return np.gradient(u, y, axis=1, edge_order=2)
    else:
        return np.gradient(u, axis=1, edge_order=2)


def ddz(u, z=None):
    """
    :param u: n-dimensional field.
    :return: the first-order derivative in the k direction of (n>=3 dimensional) field.
    """
    if np.array_equal(u[..., 0], u[..., -1]):  # Periodic on last axis
        u_temp = np.zeros((u.shape[0], u.shape[1], u.shape[2]+2))
        u_temp[:, :, 1:-1], u_temp[:, :, 0], u_temp[:, :, -1] = u, u[:, :, -1], u[:, :, 0]
        del u
        if z is not None:
            dudz = np.gradient(u_temp, z, axis=2, edge_order=2)
        else:
            dudz = np.gradient(u_temp, axis=2, edge_order=2)
        del u_temp
        return dudz[:, :, 1:-1]
    else:
        if z is not None:
            return np.gradient(u, z, axis=2, edge_order=2)
        else:
            return np.gradient(u, axis=2, edge_order=2)


def vortZ(u, v, **kwargs):
    """
    :param u: x component of the velocity vector field.
    :param v: y component of the velocity vector field.
    :return: The z-vorticity of a two-dimensional velocity vector field
    """
    # if not (len(u.shape)==2 and len(v.shape)==2):
    #     raise ValueError("Fields must be two-dimensional")
    # else:
    x = kwargs.get('x', None)
    y = kwargs.get('y', None)
    return ddx(v, x)-ddy(u, y)


def vort(u, v, w, **kwargs):
    """
    :param u: x component of the velocity vector field.
    :param v: y component of the velocity vector field.
    :param w: z component of the velocity vector field.

    :return: the three components of the vorticity of a three-dimensional velocity vector field.
    """
    if not (len(u.shape)==3 and len(v.shape)==3 and len(w.shape)==3):
        raise ValueError("Fields must be three-dimensional")
    else:
        x = kwargs.get('x', None)
        y = kwargs.get('y', None)
        z = kwargs.get('z', None)
        return ddy(w, y) - ddz(v, z), ddz(u, z) - ddx(w, x), ddx(v, x) - ddy(u, y)

def grad(u):
    """
    Return the gradient of a n-dimensinal array (scalar field)
    :param u: input scalar array
    :return: vector field array, gradient of u
    """
    return np.array(np.gradient(u, edge_order=2))

def div(*args):
    dd = [ddx, ddy, ddz]
    res = np.zeros(args[0].shape)
    for i, a in enumerate(args):
        res += dd[i](a)
    return res

def J(u, v, w):
    """
    Calculate the velocity gradient tensor
    :param u: Horizontal velocity component
    :param v: Vertical velocity component
    :param w: Spanwise velocity component
    :return: np.array (tensor) of the velocity gradient
    """
    a11, a12, a13 = ddx(u), ddy(u), ddz(u)
    a21, a22, a23 = ddx(v), ddy(v), ddz(v)
    a31, a32, a33 = ddx(w), ddy(w), ddz(w)

    return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

def S(u, v, w):
    J = _J(u, v, w)
    return 0.5*(J + np.transpose(J, (1,0,2,3,4)))

def R(u, v, w):
    J = _J(u, v, w)
    return 0.5*(J - np.transpose(J, (1,0,2,3,4)))

def Q(u, v, w):
    J = _J(u, v, w)
    S = 0.5*(J + np.transpose(J, (1,0,2,3,4)))
    R = 0.5*(J - np.transpose(J, (1,0,2,3,4)))
    S_mag = np.linalg.norm(S, ord='fro', axis=(0,1))
    S_mag2 = np.sqrt(np.trace(np.dot(S, S.T)))
    print(np.sum(S_mag))
    print(np.sum(S_mag2))
    R_mag = np.linalg.norm(R, ord='fro', axis=(0,1))

    print(R_mag.shape)

    Q = 0.5*(R_mag**2-S_mag**2)
    # Q = np.clip(Q, 0, None)
    return Q

def map_cyl(grid2D, r, eps):
    x, y = grid2D[0], grid2D[1]
    r_grid = np.sqrt(x**2+y**2)

    indices = np.where((r_grid>=r*(1+0.5*eps)) & (r_grid<=r*(1+1.5*eps)))
    r_grid = np.zeros(x.shape)
    r_grid[indices] = 1

    return r_grid, indices

def map_normal(grid2D, r, r_max, angle):
    x, y = grid2D[0], grid2D[1]
    r_grid = np.sqrt(x**2+y**2)
    atan_grid = np.arctan2(y, x)*180/np.pi

    angle_eps = 0.5

    # indices = np.where((r_grid<=r_max) & (np.abs(atan_grid-angle)<angle_eps))
    indices = np.where((r_grid<=r_max) & close(atan_grid, angle, angle_eps))
    result = np.zeros(x.shape)
    result[indices] = 1

    return result, indices


def separation_points(q, alphas, eps=0.0005):
    """
    :param q: 2D scalar field (quantity) we analyse
    :param l: list of tuples. Each tuples is: ((i,j), alpha))
    :return: upper and lower separation points
    """
    alpha_u, alpha_l = None, None
    # find upper separation point
    for (idx, alpha) in sorted(alphas, key=lambda tup: tup[1])[80:]:
        if -eps<=q[idx] and q[idx]<=eps:
            alpha_u = alpha
            break
    # find lower separation point
    for (idx, alpha) in sorted(alphas, key=lambda tup: tup[1], reverse=True)[80:]:
        if -eps<=q[idx] and q[idx]<=eps:
            alpha_l = alpha
            break
    return alpha_u, alpha_l-360


def separation_points2(q, alphas, eps=0.1):
    """
    :param q: 2D scalar field (quantity) we analyse
    :param l: list of tuples. Each tuples is: ((i,j), alpha))
    :return: upper and lower separation points
    """
    alpha_u, alpha_l = None, None
    l_upper = sorted(alphas, key=lambda tup: tup[1])[100:]
    l_lower = sorted(alphas, key=lambda tup: tup[1], reverse=True)[100:]

    # find upper separation point
    for i, (idx, alpha) in enumerate(l_upper):
        if q[idx]<eps:
            alpha_u = alpha
            break

    # find lower separation point
    for i, (idx, alpha) in enumerate(l_lower):
        if q[idx]<eps:
            alpha_l = alpha
            break
    return alpha_u, alpha_l-360


def corr(a,b):
    am = np.mean(a)
    bm = np.mean(b)
    ac = a - am  # a centered
    bc = b - bm  # b centered

    cov_ab = np.mean(ac * bc)
    cov_aa = np.mean(ac ** 2)
    cov_bb = np.mean(bc ** 2)

    return cov_ab / np.sqrt(cov_aa * cov_bb)


def close(a, b, eps):
    return np.abs(a - b) < eps