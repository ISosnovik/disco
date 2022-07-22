'''
This file is a part of the official implementation of
1) "DISCO: accurate Discrete Scale Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, BMVC 2021
    arxiv: https://arxiv.org/abs/2106.02733

2) "How to Transform Kernels for Scale-Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, ICCV VIPriors 2021
    pdf: https://openaccess.thecvf.com/content/ICCV2021W/VIPriors/papers/Sosnovik_How_To_Transform_Kernels_for_Scale-Convolutions_ICCVW_2021_paper.pdf

---------------------------------------------------------------------------

The source of this file is a part of the official implementation of 
"Scale-Equivariant Steerable Networks"
by Ivan Sosnovik, Michał Szmaja, and Arnold Smeulders, ICLR 2020
arxiv: https://arxiv.org/abs/1910.11093
code: https://github.com/ISosnovik/sesn

---------------------------------------------------------------------------

MIT License. Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
'''


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _Basis, normalize_basis_by_min_scale


class HermiteBasisA(_Basis):

    def __init__(self, size, scales, effective_size):
        super().__init__(size=size, scales=scales, effective_size=effective_size)
        basis = steerable_A(size, self.scales, effective_size)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)


class HermiteBasisB(_Basis):

    def __init__(self, size, scales, effective_size, basis_mult, basis_max_order):
        super().__init__(size=size, scales=scales, effective_size=effective_size)
        basis = steerable_B(size, self.scales, effective_size,
                            mult=basis_mult, max_order=basis_max_order)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)


############################################
############     UTILS      ################
############################################
def hermite_poly(X, n):
    """Hermite polynomial of order n calculated at X
    Args:
        n: int >= 0
        X: np.array
    Output:
        Y: array of shape X.shape
    """
    coeff = [0] * n + [1]
    func = np.polynomial.hermite_e.hermeval(X, coeff)
    return func


def hermite_basis_varying_order(size, scale, max_order=None):
    print(f'size={size}, scale={scale}, max_order={max_order}')
    max_order = max_order or size - 1
    X = torch.linspace(-(size // 2), size // 2, size)
    Y = torch.linspace(-(size // 2), size // 2, size)
    order_y, order_x = np.indices([max_order + 1, max_order + 1])

    G = torch.exp(-X**2 / (2 * scale**2)) / scale

    basis_x = [G * hermite_poly(X / scale, n) for n in order_x.ravel()]
    basis_y = [G * hermite_poly(Y / scale, n) for n in order_y.ravel()]
    basis_x = torch.stack(basis_x)
    basis_y = torch.stack(basis_y)
    basis = torch.bmm(basis_x[:, :, None], basis_y[:, None, :])
    return basis


def hermite_basis_varying_sigma(size, base_scale, max_order=4, mult=2, num_funcs=None):
    '''Basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale (sigma).
    '''
    num_funcs = num_funcs or size ** 2
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2)) // 2
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]

    basis_x = []
    basis_y = []

    X = torch.linspace(-(size // 2), size // 2, size)
    Y = torch.linspace(-(size // 2), size // 2, size)

    for scale in scales:
        G = torch.exp(-X**2 / (2 * scale**2)) / scale

        order_y, order_x = np.indices([max_order + 1, max_order + 1])
        mask = order_y + order_x <= max_order
        bx = [G * hermite_poly(X / scale, n) for n in order_x[mask]]
        by = [G * hermite_poly(Y / scale, n) for n in order_y[mask]]

        basis_x.extend(bx)
        basis_y.extend(by)

    basis_x = torch.stack(basis_x)[:num_funcs]
    basis_y = torch.stack(basis_y)[:num_funcs]
    return torch.bmm(basis_x[:, :, None], basis_y[:, None, :])


def steerable_A(size, scales, effective_size):
    max_order = effective_size - 1
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        basis = hermite_basis_varying_order(size_before_pad, scale, max_order)
        basis = basis[None, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    return torch.stack(basis_tensors, 1)


def steerable_B(size, scales, effective_size, mult=1.4, max_order=4):
    num_funcs = effective_size**2
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        basis = hermite_basis_varying_sigma(size_before_pad,
                                            base_scale=scale,
                                            max_order=max_order,
                                            mult=mult,
                                            num_funcs=num_funcs)
        basis = basis[None, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    return torch.stack(basis_tensors, 1)


def mhg_grid(size, base_scale, mult, max_func_order, max_scale_order):
    num_funcs = ((max_func_order + 1) * (max_func_order + 2)) // 2
    num_funcs = num_funcs * max_scale_order
    grid = hermite_basis_varying_sigma(
        size, base_scale, max_order=max_func_order, num_funcs=num_funcs)
    grid = grid.view(max_scale_order, -1, size, size)
    return grid


def precon_scale_first(size, base_scale, mult, max_order, num_funcs):
    max_scale_order = max_order
    max_func_order = num_funcs // max_scale_order
    max_func_order = int((2 * max_func_order)**0.5)
    funcs = mhg_grid(size, base_scale, mult, max_func_order, max_scale_order)
    funcs = funcs.permute(1, 0, 2, 3).contiguous().view(-1, size, size)
    return funcs[:num_funcs]
