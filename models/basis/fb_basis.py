'''
This file is a part of the official implementation of
1) "DISCO: accurate Discrete Scale Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, BMVC 2021
    arxiv: https://arxiv.org/abs/2106.02733

2) "How to Transform Kernels for Scale-Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, ICCV VIPriors 2021
    pdf: https://openaccess.thecvf.com/content/ICCV2021W/VIPriors/papers/Sosnovik_How_To_Transform_Kernels_for_Scale-Convolutions_ICCVW_2021_paper.pdf 

---------------------------------------------------------------------------

The source of this file is a part of the implementation of 
"Scaling-Translation-Equivariant Networks with Decomposed Convolutional Filters"
Wei Zhu, Qiang Qiu, Robert Calderbank, Guillermo Sapiro, Xiuyuan Cheng, 2019
arxiv: https://arxiv.org/abs/1909.11193

---------------------------------------------------------------------------

MIT License. Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
'''

import os

import numpy as np
from scipy import special
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _Basis, normalize_basis_by_min_scale


BESSEL_PATH = os.path.join(os.path.dirname(__file__), 'bessel.mat')


class FourierBesselBasis(_Basis):

    def __init__(self, size, scales, effective_size):
        super().__init__(size=size, scales=scales, effective_size=effective_size)
        basis = steerable(size, scales, effective_size)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)


# UTIL FUNCTIONS
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)


def fb_basis_grid(size, scale, num_funcs=None):
    maxK = num_funcs or size**2
    L = size // 2 + 1
    R = size // 2 + 0.5
    truncate_freq_factor = 2.7

    xx, yy = np.meshgrid(range(-L, L + 1), range(-L, L + 1))
    xx = xx / R / scale
    yy = yy / R / scale

    ugrid = np.concatenate([yy.reshape(-1, 1), xx.reshape(-1, 1)], 1)
    tgrid, rgrid = cart2pol(ugrid[:, 0], ugrid[:, 1])

    num_grid_points = ugrid.shape[0]

    bessel = scipy.io.loadmat(BESSEL_PATH)['bessel']
    mask1 = bessel[:, 0] <= maxK
    mask2 = bessel[:, 3] <= np.pi * R * truncate_freq_factor
    B = bessel[mask1 & mask2]

    idxB = np.argsort(B[:, 2])
    mu_ns = B[idxB, 2]**2
    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]

    num_kq_all = len(ang_freqs)
    max_ang_freqs = max(ang_freqs)

    Phi_ns = np.zeros((num_grid_points, num_kq_all), np.float32)

    Psi = []
    for i in range(B.shape[0]):
        ki = ang_freqs[i]
        qi = rad_freqs[i]
        rkqi = R_ns[i]
        r0grid = rgrid * R_ns[i]
        F = special.jv(ki, r0grid)
        Phi = 1. / np.abs(special.jv(ki + 1, R_ns[i])) * F
        Phi[rgrid >= 1] = 0
        Phi_ns[:, i] = Phi

        if ki == 0:
            Psi.append(Phi)
        else:
            Psi.append(Phi * np.cos(ki * tgrid) * np.sqrt(2))
            Psi.append(Phi * np.sin(ki * tgrid) * np.sqrt(2))

    Psi = np.array(Psi)
    Psi = Psi[:maxK]
    p = Psi.reshape(-1, 2 * L + 1, 2 * L + 1)
    psi = p[:, 1:-1, 1:-1]
    return psi


def steerable(size, scales, effective_size):
    max_scale = max(scales)
    num_functions = effective_size**2
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        pad_size = (size - size_before_pad) // 2
        print('SIZE: {}'.format(size_before_pad))
        basis = fb_basis_grid(size, scale, num_funcs=num_functions)
        basis = torch.Tensor(np.stack(basis))
        basis = basis[None, :num_functions, :, :] / scale / scale
        basis = F.pad(basis, [-pad_size] * 4)
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)

    B = torch.stack(basis_tensors, 1)
    return B
