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


from collections import OrderedDict


def repr1line(obj):
    '''custom convenience dumper to YAML compatible format
    the output is 1 line. keys a sorted by two rules: models (Y/n), abc
    '''
    if obj is None:
        return 'null'

    if isinstance(obj, bool):
        return str(obj).lower()

    if isinstance(obj, str):
        return "'{}'".format(obj)

    if isinstance(obj, (int, float)):
        return str(obj)

    if isinstance(obj, (tuple, set)):
        return repr1line(list(obj))

    if isinstance(obj, list):
        els = map(lambda x: repr1line(x), obj)
        return '[{}]'.format(', '.join(els))

    if isinstance(obj, dict):
        keys = list(obj.keys())
        keys.remove('model')
        keys = ['model'] + sorted(keys)
        items = [(k, obj[k]) for k in keys]
        obj = OrderedDict(items)
        s = '{'
        for i, (k, v) in enumerate(obj.items()):
            s += '{}: {}, '.format(k, repr1line(v))
        s = s[:-2] + '}'
        return s


def dump_list_element_1line(obj):
    return '- {}\n'.format(repr1line(obj))


def pretty_seconds(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)
