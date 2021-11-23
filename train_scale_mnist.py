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


import os
import time
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import models
from utils.train_utils import train_xent, test_acc
from utils import loaders
from utils.model_utils import get_num_parameters
from utils.misc import dump_list_element_1line


#########################################
# arguments
#########################################
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_steps', type=int, nargs='+', default=[20, 40])
parser.add_argument('--lr_gamma', type=float, default=0.1)


parser.add_argument('--model', type=str, choices=model_names, required=True)
parser.add_argument('--extra_scaling', type=float, default=1.0,
                    required=False, help='add scaling data augmentation')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--save_model_path', type=str, default='')
parser.add_argument('--data_dir', type=str)


# bases hyperparameters
parser.add_argument('--basis_max_order', type=int, default=4)
parser.add_argument('--basis_num_rotations', type=int, default=5)
parser.add_argument('--basis_min_scale', type=float, default=1.5)
parser.add_argument('--basis_mult', type=float, default=1.5)
parser.add_argument('--basis_sigma', type=float, default=0.2)
parser.add_argument('--basis_save_dir', type=str, default="")


args = parser.parse_args()

print("Args:")
for k, v in vars(args).items():
    print("  {}={}".format(k, v))

print(flush=True)
assert len(args.save_model_path)


#########################################
# Data
#########################################
train_loader = loaders.scale_mnist_train_loader(args.batch_size, args.data_dir, args.extra_scaling)
val_loader = loaders.scale_mnist_val_loader(args.batch_size, args.data_dir)
test_loader = loaders.scale_mnist_test_loader(args.batch_size, args.data_dir)


print('Train:')
print(train_loader.dataset)
print('\nVal:')
print(val_loader.dataset)
print('\nTest:')
print(test_loader.dataset)

#########################################
# Model
#########################################
model = models.__dict__[args.model]
model = model(**vars(args))
print('\nModel:')
print(model)
print()

use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device: {}'.format(device))

if use_cuda:
    cudnn.enabled = True
    cudnn.benchmark = True
    print('CUDNN is enabled. CUDNN benchmark is enabled')
    model.cuda()

print('num_params:', get_num_parameters(model))
print(flush=True)


#########################################
# optimizer
#########################################
parameters = filter(lambda x: x.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=args.lr)


print(optimizer)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.lr_gamma)


#########################################
# training
#########################################
print('\nTraining\n' + '-' * 30)

if args.save_model_path:
    if not os.path.isdir(os.path.dirname(args.save_model_path)):
        os.makedirs(os.path.dirname(args.save_model_path))

start_time = time.time()
best_acc = 0.0

for epoch in range(args.epochs):
    train_xent(model, optimizer, train_loader, device)
    acc = test_acc(model, val_loader, device)
    print('Epoch {:3d}/{:3d}| Acc@1: {:3.1f}%'.format(
        epoch + 1, args.epochs, 100 * acc), flush=True)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), args.save_model_path)

    lr_scheduler.step()

print('-' * 30)
print('Training is finished')
print('Best Acc@1: {:3.1f}%'.format(best_acc * 100), flush=True)
end_time = time.time()
elapsed_time = end_time - start_time
time_per_epoch = elapsed_time / args.epochs

print('\nTesting\n' + '-' * 30)
model.load_state_dict(torch.load(args.save_model_path))
final_acc = test_acc(model, test_loader, device)
print('Test Acc:', final_acc)

#########################################
# save results
#########################################
results = vars(args)
results.update({
    'dataset': 'scale_mnist',
    'elapsed_time': int(elapsed_time),
    'time_per_epoch': int(time_per_epoch),
    'num_parameters': int(get_num_parameters(model)),
    'acc': final_acc,
})

with open('results.yml', 'a') as f:
    f.write(dump_list_element_1line(results))
