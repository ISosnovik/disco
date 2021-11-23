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
import time
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.basis import ApproximateProxyBasis
from models.basis.disco import get_basis_filename
from utils import loaders
from utils.train_utils import train_equi_loss
from utils.model_utils import get_num_parameters


#########################################
# arguments
#########################################
parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=40)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_steps', type=int, nargs='+', default=[20, 30])
parser.add_argument('--lr_gamma', type=float, default=0.1)

parser.add_argument('--cuda', action='store_true', default=False)

# basis hyperparameters
parser.add_argument('--basis_size', type=int, default=7)
parser.add_argument('--basis_effective_size', type=int, default=3)
parser.add_argument('--basis_scales', type=float, nargs='+', default=[1.0])
parser.add_argument('--basis_save_dir', type=str, required=True)


args = parser.parse_args()

print("Args:")
for k, v in vars(args).items():
    print("  {}={}".format(k, v))

print(flush=True)


#########################################
# Data
#########################################
loader = loaders.random_loader(args.batch_size)


print('Dataset:')
print(loader.dataset)


#########################################
# Model
#########################################
basis = ApproximateProxyBasis(size=args.basis_size, scales=args.basis_scales,
                              effective_size=args.basis_effective_size)

print('\nBasis:')
print(basis)
print()

use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device: {}'.format(device))

if use_cuda:
    cudnn.enabled = True
    cudnn.benchmark = True
    print('CUDNN is enabled. CUDNN benchmark is enabled')
    basis.cuda()

print(flush=True)

#########################################
# optimizer
#########################################
parameters = filter(lambda x: x.requires_grad, basis.parameters())
optimizer = optim.Adam(parameters, lr=args.lr)
print(optimizer)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.lr_gamma)


#########################################
# Paths
#########################################


save_basis_postfix = get_basis_filename(size=args.basis_size,
                                        effective_size=args.basis_effective_size,
                                        scales=args.basis_scales)
save_basis_path = os.path.join(args.basis_save_dir, save_basis_postfix)
print('Basis path: ', save_basis_path)
print()

if not os.path.isdir(args.basis_save_dir):
    os.makedirs(args.basis_save_dir)

#########################################
# Training
#########################################

print('\nTraining\n' + '-' * 30)
start_time = time.time()
best_loss = float('inf')

for epoch in range(args.epochs):
    loss = train_equi_loss(basis, optimizer, loader, device)
    print('Epoch {:3d}/{:3d}| Loss: {:.2e}'.format(epoch + 1, args.epochs, loss), flush=True)
    if loss < best_loss:
        best_loss = loss

        with torch.no_grad():
            torch.save(basis.get_basis().cpu(), save_basis_path)

    lr_scheduler.step()

print('-' * 30)
print('Training is finished')
print('Best Loss: {:.2e}'.format(best_loss), flush=True)
end_time = time.time()
elapsed_time = end_time - start_time
time_per_epoch = elapsed_time / args.epochs

print('Total Time Elapsed: {:.2f}'.format(elapsed_time))
print('Time per Epoch: {:.2f}'.format(time_per_epoch))
