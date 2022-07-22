'''
This file is a part of the official implementation of
1) "DISCO: accurate Discrete Scale Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, BMVC 2021
    arxiv: https://arxiv.org/abs/2106.02733

2) "How to Transform Kernels for Scale-Convolutions"
    by Ivan Sosnovik, Artem Moskalev, Arnold Smeulders, ICCV VIPriors 2021
    pdf: https://openaccess.thecvf.com/content/ICCV2021W/VIPriors/papers/Sosnovik_How_To_Transform_Kernels_for_Scale-Convolutions_ICCVW_2021_paper.pdf

---------------------------------------------------------------------------

MIT License. Copyright (c) 2021 Ivan Sosnovik, Artem Moskalev
'''

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .base import _Basis, normalize_basis_by_min_scale
from .hermite_basis import hermite_basis_varying_order, hermite_basis_varying_sigma


def get_basis_filename(size, effective_size, scales):
    scales = [round(x, 2) for x in scales]

    fname = 'size={size}_eff_size={effective_size}_scales={scales}.pt'
    fname = fname.format(size=size, effective_size=effective_size, scales=scales)
    return fname


def dilated_standard_basis(size, effective_size, scale):
    scale = int(scale)
    _size = ((effective_size - 1) // 2) * scale * 2 + 1
    K = torch.eye(_size**2).view(_size, _size, _size, _size)
    K = K[::scale, ::scale].reshape(-1, 1, _size, _size)
    K = F.pad(K, [(size - _size) // 2] * 4)[:, 0]
    return K


def apply_rescaling(x, scale):
    return F.interpolate(x, scale_factor=1 / scale, mode='bicubic', align_corners=False)


def conv_resize(x, kernel, scale):
    kernel_size = kernel.shape[-1]
    x = F.pad(x, [kernel_size // 2] * 4, mode='circular')
    x = F.conv2d(x, kernel)
    x = apply_rescaling(x, scale)
    return x


def resize_conv(x, kernel, scale):
    kernel_size = kernel.shape[-1]
    x = apply_rescaling(x, scale)
    x = F.pad(x, [kernel_size // 2] * 4, mode='circular')
    x = F.conv2d(x, kernel)
    return x


class ApproximateProxyBasis(_Basis):

    def __init__(self, size, scales, effective_size):
        super().__init__(size, scales, effective_size)
        self.bases_list = nn.ParameterList()

        for scale in scales:
            size_before_pad = int(size * scale / max(scales)) // 2 * 2 + 1
            if float(scale).is_integer():
                basis = dilated_standard_basis(size, effective_size, scale)
                basis = nn.Parameter(basis, requires_grad=False)
            else:
                basis = torch.randn(self.num_funcs, size, size)

                basis = nn.Parameter(basis)
            self.bases_list.append(basis)

    def get_basis(self):
        basis_tensor = []
        for basis in self.bases_list:
            pad = (self.size - basis.shape[-1]) // 2
            if pad > 0:
                basis = F.pad(basis, [pad] * 4)
            basis_tensor.append(basis)
        return torch.stack(basis_tensor, 1)

    def get_equivariance_loss(self, x):
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        kernels = self.get_basis()[:, None]

        total_loss = 0.0
        for i in range(self.num_scales):
            for j in range(i + 1, self.num_scales):
                scale = self.scales[j] / self.scales[i]

                x_resize_conv = resize_conv(x, kernels[:, :, i], scale)
                x_conv_resize = conv_resize(x, kernels[:, :, j], scale)

                crop = self.size // 2 + 1
                loss = (x_conv_resize - x_resize_conv)[:, :, crop:-crop, crop:-crop].norm(2)
                total_loss += loss
        return total_loss


class DISCOBasisA(_Basis):

    def __init__(self, size, scales, effective_size, basis_save_dir, basis_min_scale):

        super().__init__(size=size, scales=scales, effective_size=effective_size)
        self.basis_save_dir = basis_save_dir
        self.basis_min_scale = basis_min_scale

        fname = get_basis_filename(size=size, effective_size=effective_size, scales=scales)
        fpath = os.path.join(basis_save_dir, fname)
        self.fpath = fpath

        try:
            basis = torch.load(fpath, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError(
                "No basis found at {}. Calculate it first with `calculate_disco_basis.py`".format(fpath))

        W = hermite_basis_varying_order(
            effective_size, basis_min_scale, max_order=effective_size - 1).view(self.num_funcs, -1)
        basis = (W @ basis.view(self.num_funcs, -1)).view(self.num_funcs, len(scales), size, size)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

    def extra_repr(self):
        s = super().extra_repr()
        s += '\nbasis_save_dir={basis_save_dir}, basis_min_scale={basis_min_scale},'
        return s.format(**self.__dict__)


class DISCOBasisB(_Basis):

    def __init__(self, size, scales, effective_size,
                 basis_save_dir, basis_mult, basis_max_order, basis_min_scale):

        super().__init__(size=size, scales=scales, effective_size=effective_size)
        self.basis_save_dir = basis_save_dir
        self.basis_mult = basis_mult
        self.basis_max_order = basis_max_order
        self.basis_min_scale = basis_min_scale

        fname = get_basis_filename(size=size, effective_size=effective_size, scales=scales)
        fpath = os.path.join(basis_save_dir, fname)
        self.fpath = fpath

        try:
            basis = torch.load(fpath, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError(
                "No basis found at {}. Calculate it first with `calculate_disco_basis.py`".format(fpath))
        W = hermite_basis_varying_sigma(
            effective_size, basis_min_scale, basis_max_order, basis_mult)
        W = W.view(self.num_funcs, -1)
        basis = (W @ basis.view(self.num_funcs, -1)).view(self.num_funcs, len(scales), size, size)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

    def extra_repr(self):
        s = super().extra_repr()
        s += '\nbasis_save_dir={basis_save_dir}, basis_min_scale={basis_min_scale},'
        s += '\nbasis_mult={basis_mult}, basis_max_order={basis_max_order}'
        return s.format(**self.__dict__)
