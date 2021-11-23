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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ses_conv import SESConv_Z2_H, SESConv_H_H, SESMaxProjection


class MNIST_DISCO(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0],
                 basis_type='hermite_a', dropout=0.7, **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            SESConv_Z2_H(1, C1, kernel_size, 7, scales=scales,
                         padding=kernel_size // 2, bias=True,
                         basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.AvgPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C1),

            SESConv_H_H(C1, C2, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.AvgPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C2),

            SESConv_H_H(C2, C3, 1, kernel_size, 7, scales=scales,
                        padding=kernel_size // 2, bias=True,
                        basis_type=basis_type, **kwargs),
        )

        self.pool = nn.Sequential(
            SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size - 1, stride=1),
            nn.AvgPool2d(2, stride=4),
            nn.BatchNorm2d(C3)
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def mnist_disco(basis_min_scale, basis_save_dir, basis_mult, **kwargs):
    q = 2 ** (1 / 3)
    scales = [q**i for i in range(4)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_DISCO(pool_size=8, kernel_size=15, scales=scales, basis_type='disco_b',
                        basis_save_dir=basis_save_dir, basis_mult=basis_mult,
                        basis_max_order=4, basis_min_scale=basis_min_scale)
    return model
