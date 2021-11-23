'''
This file is a part of the official implementation of
1) "DISCO: accurate Discrete Scale Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, BMVC 2021
    arxiv: https://arxiv.org/abs/2106.02733

2) "How to Transform Kernels for Scale-Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, ICCV VIPriors 2021
    pdf: https://openaccess.thecvf.com/content/ICCV2021W/VIPriors/papers/Sosnovik_How_To_Transform_Kernels_for_Scale-Convolutions_ICCVW_2021_paper.pdf

---------------------------------------------------------------------------

It contains a re-implementation of the basis presented in 
1) "Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks"
    by Rohan Ghosh, Anupam K. Gupta, 
    ICML 2019 Workshop on Theoretical Physics for Deep Learning
    arxiv: https://arxiv.org/abs/1906.03861

2) "Scale Equivariant CNNs with Scale Steerable Filters"
    by Naderi, Hanieh, Leili Goli, and Shohreh Kasaei, MVIP 2020
    paper: https://ieeexplore.ieee.org/document/9116889 

---------------------------------------------------------------------------

MIT License. Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _Basis, normalize_basis_by_min_scale


class LRHarmonicsBasis(_Basis):

    def __init__(self, size, scales, effective_size, basis_max_order, basis_num_rotations, basis_sigma):
        super().__init__(size=size, scales=scales, effective_size=effective_size)
        basis = steerable(size, scales, effective_size, max_order=basis_max_order,
                          num_rotations=basis_num_rotations, sigma=basis_sigma)

        basis = normalize_basis_by_min_scale(basis)
        assert len(basis) == self.num_funcs
        self.register_buffer('basis', basis)


def basis_func(filter_size, phi0, sigma, order, scale):
    Y, X = np.meshgrid(np.arange(-(filter_size // 2), filter_size // 2 + 1),
                       -np.arange(-(filter_size // 2), filter_size // 2 + 1))

    angle = np.arctan2(Y, X)
    radius = np.sqrt(X**2 + Y**2)
    radius[filter_size // 2, filter_size // 2] = 1
    radius /= scale
    log_r = np.log(radius)

    angle = np.abs(np.stack([
        angle - phi0 - np.pi,
        angle - phi0,
        angle - phi0 + np.pi,
        angle - phi0 + 2 * np.pi
    ], 0)).min(0)

    angular_part = np.exp(-angle**2 / 2 / sigma**2)
    arg = 2 * np.pi * order * log_r / log_r.max() - np.pi / 4  # Why? check the original code
    filter_real = angular_part * np.cos(arg) / radius
    filter_imag = angular_part * np.sin(arg) / radius

    return filter_real, filter_imag


def onescale_grid(size, scale, max_order, num_rotations, sigma=0.2):
    basis = []
    for order in range(max_order):
        order = 2 ** (order - 1)
        for angle in np.linspace(0, np.pi, num_rotations):
            f1, f2 = basis_func(size, angle, sigma=sigma, order=order, scale=scale)
            basis.append(f1)
            basis.append(f2)
    return basis


def steerable(size, scales, effective_size, max_order, num_rotations, sigma):
    max_scale = max(scales)
    num_functions = effective_size**2
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        print('SIZE: {}'.format(size_before_pad))
        basis = onescale_grid(size_before_pad, scale, max_order, num_rotations, sigma)
        basis = torch.Tensor(np.stack(basis))
        basis = basis[None, :num_functions, :, :] / scale / scale
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)

    B = torch.stack(basis_tensors, 1)
    return B
