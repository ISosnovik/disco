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

import numpy as np


def train_xent(model, optimizer, loader, device=torch.device('cuda')):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test_acc(model, loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    accuracy /= len(loader.dataset)
    return accuracy


def train_equi_loss(basis, optimizer, loader, device=torch.device('cuda')):
    losses = []
    basis.train()
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()

        loss = basis.get_equivariance_loss(data)
        loss.backward()
        optimizer.step()

        losses.append(loss.item() / data.nelement())

    return np.mean(losses)
