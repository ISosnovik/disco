'''
This file is a part of the official implementation of
"Scale-Equivariant Steerable Networks"
by Ivan Sosnovik, Michał Szmaja, and Arnold Smeulders, ICLR 2020
arxiv: https://arxiv.org/abs/1910.11093
code: https://github.com/ISosnovik/sesn

---------------------------------------------------------------------------

MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
'''


def get_num_parameters(module):
    params = [p.nelement() for p in module.parameters() if p.requires_grad]
    num = sum(params)
    return num
