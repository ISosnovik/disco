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

import torch
import torch.nn as nn
import numpy as np


class _Basis(nn.Module):

    def __init__(self, size, scales, effective_size):
        super().__init__()
        self.size = size
        self.scales = scales
        self.num_scales = len(scales)
        self.effective_size = effective_size
        self.num_funcs = effective_size**2

    def get_basis(self):
        return self.basis

    def forward(self, weight):
        basis = self.get_basis()
        kernel = weight @ basis.view(self.num_funcs, -1)
        kernel = kernel.view(*weight.shape[:-1], self.num_scales, self.size, self.size)
        return kernel

    def extra_repr(self):
        s = '{size}x{size} | scales={scales} | ranks={ranks} | num_funcs={num_funcs}'
        with torch.no_grad():
            ranks = [_calculate_rank(self.get_basis().cpu(), i) for i in range(len(self.scales))]
            if min(ranks) == 0:
                raise AttributeError('One of attributes is incorrect. rank = 0')
        return s.format(**self.__dict__, ranks=ranks)


def normalize_basis_by_min_scale(basis):
    norm = basis.pow(2).sum([2, 3], keepdim=True).sqrt()[:, [0]]
    return basis / norm


def _calculate_rank(basis, scale):
    b = basis[:, scale]
    b = b.view(len(b), -1)
    return np.linalg.matrix_rank(b.detach().numpy())
