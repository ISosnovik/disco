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
from utils.misc import dump_list_element_1line, pretty_seconds


#########################################
# arguments
#########################################
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=1000)

parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', action='store_true', default=False)
parser.add_argument('--decay', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--lr_steps', type=int, nargs='+', default=[300, 400, 600, 800])
parser.add_argument('--lr_gamma', type=float, default=0.2)


parser.add_argument('--model', type=str, choices=model_names, required=True)
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--save_model_path', type=str, default='')
parser.add_argument('--data_dir', type=str)

parser.add_argument('--basis_save_dir', type=str, default="")

args = parser.parse_args()

print("Args:")
for k, v in vars(args).items():
    print("  {}={}".format(k, v))

print(flush=True)


#########################################
# Data
#########################################
train_loader = loaders.stl10_plus_train_loader(args.batch_size, args.data_dir)
test_loader = loaders.stl10_test_loader(args.batch_size, args.data_dir)

print('Train:')
print(train_loader.dataset)
print()
print('Test:')
print(test_loader.dataset)
print(flush=True)

#########################################
# Model
#########################################
model = models.__dict__[args.model]
model = model(**vars(args))
print(model)

use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device: {}'.format(device))

if use_cuda:
    num_gpus = torch.cuda.device_count()
    cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, range(num_gpus))
    print('model is using {} GPU(s)'.format(num_gpus))

print('num_params:', get_num_parameters(model))
print(flush=True)


#########################################
# optimizer
#########################################
parameters = filter(lambda x: x.requires_grad, model.parameters())
optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                      weight_decay=args.decay, nesterov=args.nesterov)

print(optimizer)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.lr_gamma)


#########################################
# training
#########################################
print('\nTraining\n' + '-' * 30)

start_time = time.time()
best_acc = 0.0

for epoch in range(args.epochs):
    train_xent(model, optimizer, train_loader, device)
    if epoch % 50 == 0:
        current_time = time.time() - start_time
        eta = current_time * (args.epochs - epoch - 1) / (epoch + 1)
        print('Epoch {:3d}/{:3d}| Time={:.0f}s| '
              'ETA={}'.format(epoch + 1, args.epochs, current_time, pretty_seconds(eta)), flush=True)

    if epoch % 100 == 0:
        acc = test_acc(model, test_loader, device)
        print('Epoch {:3d}/{:3d}| '
              'Acc@1: {:3.1f}%'.format(epoch + 1, args.epochs, 100 * acc), flush=True)
        if acc > best_acc:
            best_acc = acc

    lr_scheduler.step()

print('-' * 30)
print('Training is finished')
print('Testing...')
final_acc = test_acc(model, test_loader, device)
print('Final Acc@1: {:3.1f}%'.format(final_acc * 100))
print('Best Acc@1: {:3.1f}%'.format(best_acc * 100), flush=True)
end_time = time.time()
elapsed_time = end_time - start_time
time_per_epoch = elapsed_time / args.epochs


#########################################
# save results
#########################################
if args.save_model_path:
    if not os.path.isdir(os.path.dirname(args.save_model_path)):
        os.makedirs(os.path.dirname(args.save_model_path))

    torch.save(model.state_dict(), args.save_model_path)
    print('Model saved: "{}"'.format(args.save_model_path))

results = vars(args)
results.update({
    'dataset': 'stl10+',
    'elapsed_time': int(elapsed_time),
    'time_per_epoch': int(time_per_epoch),
    'num_parameters': int(get_num_parameters(model)),
    'acc': final_acc
})

with open('results.yml', 'a') as f:
    f.write(dump_list_element_1line(results))
