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
"B-Spline CNNs on Lie Groups"
Erik J Bekkers, ICLR 2020
arxiv: https://arxiv.org/abs/1909.12057
code: https://github.com/ebekkers/gsplinets

---------------------------------------------------------------------------

MIT License. Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
MIT License. Copyright (c) 2019 Erik J Bekkers

----------------------------------------------------------------------

Below is the original description

Implementation for B-splines of degree up to 50. For speed considerations the
splines of degrees up to 50 are hard-coded. This file was generated using a 
Wolfram Mathematica script in which the expressions are generated via the inverse Fourier transform
of the Fourier B-spline expression
   BF[n_][w_]:=(Sin[w/2]/(w/2))^(n+1)
with handling of the case w = 0 via
   Do[BF[n][0]=1;BF[n][0.]=1;,{n,0,nMax}]
and the spatial/time domain B-spline expression is then obtained via
   InverseFourierTransform[BF[n][w],w,x,FourierParameters{1,-1}]
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .base import _Basis, normalize_basis_by_min_scale


class BSplineBasis(_Basis):

    def __init__(self, size, scales, effective_size, basis_mult, basis_max_order, cropped=True):
        super().__init__(size=size, scales=scales, effective_size=effective_size)
        basis = steerable_cropped(size, scales, effective_size,
                                  mult=basis_mult, order=basis_max_order)

        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)


############################################
############     UTILS      ################
############################################
def b_spline_at(xy, size, mult, order):
    spline_func = B(order)
    X = np.linspace(-1, 1, size)
    Y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(X, Y)
    grid = torch.Tensor(np.stack([X, Y], -1) - xy.reshape(1, 1, 2))
    grid = grid * mult
    return spline_func(grid)


def b_spline_basis(size, scale, effective_size, mult, order):
    offsets_x = np.linspace(-1, 1, effective_size)
    offsets_y = np.linspace(-1, 1, effective_size)
    basis_tensor = []
    for dx in offsets_x:
        for dy in offsets_y:
            xy = np.array([dx, dy]) * scale
            basis_ = b_spline_at(xy, size=size, mult=mult / scale, order=order)
            basis_tensor.append(basis_)

    return torch.stack(basis_tensor, 0)


def steerable_cropped(size, scales, effective_size, mult, order):
    print('MULT={}'.format(mult))
    print('ORDER={}'.format(order))
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
        print('SIZE: {}'.format(size_before_pad))
        basis = b_spline_basis(size_before_pad, min(scales),
                               effective_size, mult=mult, order=order)
        basis = basis[None, :, :, :] / scale**2
        pad_size = (size - size_before_pad) // 2
        basis = F.pad(basis, [pad_size] * 4)[0]
        basis_tensors.append(basis)
    return torch.stack(basis_tensors, 1)


############################################
##########     B-Splines      ##############
############################################
def B(n):
    """ Returns a d-dimensional B-spline basis function of degree "n" (centered
        around zero). 

        INPUT:
            - degree n, an integer

        OUTPUT:
            - func, a python function which takes as input a torch.Tensor whose last
              dimension encodes the coordinates. E.g. B(2)([0,0.5]) computes the
              value at coordinate [0,0.5] and B(2)([[0,0.5],[0.5,0.5]]) returns 
              the values at coordinates [0,0.5] and [0.5,0.5]. This is also the
              case for a 1D B-spline: B(2)([[0],[0.5]]) returns the values of the
              1D B-spline at coordinates 0 and 0.5.
    """
    def B_Rd(x):
        return torch.prod(B_R1(n)(x), -1)
    return B_Rd


# The 1-dimensional B-spline
def B_R1(n):
    """ Returns a 1D B-spline basis function of degree "n" (centered around
        zero).

        INPUT:
            - degree n, an integer

        OUTPUT:
            - func, a python function which takes as input a position x, or a
                torch tensor array of positions, and returns the function value(s) 
                of the B-Spline basis function.
    """
    assert n >= 0 and n <= 3
    if n == 0:
        def func(x):
            return (torch.sign(1 / 2 - x) + torch.sign(1 / 2 + x)) / 2

    if n == 1:
        def func(x):
            return (-((-1 + x) * torch.sign(1 - x)) - 2 * x * torch.sign(x) + (1 + x) * torch.sign(1 + x)) / 2

    if n == 2:
        def func(x):
            return (-3 * (-1 / 2 + x) ** 2 * torch.sign(1 / 2 - x)
                    + (-3 / 2 + x) ** 2 * torch.sign(3 / 2 - x)
                    - (3 * (1 + 2 * x) ** 2 * torch.sign(1 / 2 + x)) / 4
                    + ((3 + 2 * x) ** 2 * torch.sign(3 / 2 + x)) / 4) / 4

    if n == 3:
        def func(x):
            return (4 * (-1 + x) ** 3 * torch.sign(1 - x) - (-2 + x) ** 3 * torch.sign(2 - x)
                    + 6 * x ** 3 * torch.sign(x) - 4 * (1 + x) ** 3 * torch.sign(1 + x)
                    + (2 + x) ** 3 * torch.sign(2 + x)) / 12

    return func
