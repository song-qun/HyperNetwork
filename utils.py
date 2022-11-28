import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions.multivariate_normal as N
import torch.distributions.uniform as U
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import math
import numpy as np
from bisect import bisect_right,bisect_left

import os
import sys
import time
import csv
import scipy.spatial
from scipy import misc
from torch.autograd import Variable

from tqdm import tqdm

def sample_hypernet_mnist(args ,hypernet, num):
    netE, W1, W2, W3 = hypernet
    x_dist = create_d(args.ze)
    z = sample_d(x_dist, num)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    return l1, l2, l3, codes


def sample_hypernet_cifar(args, hypernet, num):
    netE, W1, W2, W3, W4, W5 = hypernet
    x_dist = create_d(args.ze)
    z = sample_d(x_dist, num)
    codes = netE(z)
    l1 = W1(codes[0])
    l2 = W2(codes[1])
    l3 = W3(codes[2])
    l4 = W4(codes[3])
    l5 = W5(codes[4])
    return l1, l2, l3, l4, l5, codes


def weights_to_clf(weights, model, names):
    state = model.state_dict()
    layers = zip(names, weights)
    for i, (name, params) in enumerate(layers):
        name = name + '.weight'
        loader = state[name]
        state[name] = params.detach()
        model.load_state_dict(state)
    return model


class CyclicCosAnnealingLR(_LRScheduler):
    def __init__(self, optimizer,milestones, eta_min=0, last_epoch=-1):
        self.eta_min = eta_min
        self.milestones=milestones
        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]
        idx = bisect_right(self.milestones,self.last_epoch)
        left_barrier = 0 if idx==0 else self.milestones[idx-1]
        right_barrier = self.milestones[idx]
        width = right_barrier - left_barrier
        curr_pos = self.last_epoch- left_barrier 
        return [self.eta_min + (base_lr - self.eta_min) *
               (1 + math.cos(math.pi * curr_pos/ width)) / 2
                for base_lr in self.base_lrs]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.1)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    minimum = 100*torch.ones(3)
    maximum = -100*torch.ones(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
            minimum[i] = min(minimum[i],inputs[:,i,:,:].min())
            maximum[i] = max(maximum[i],inputs[:,i,:,:].max())
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std, minimum, maximum


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time



def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' ' + msg)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def WriteToFile(fp, src):
    if not os.path.exists(fp):
        open(fp, 'w').close()
    with open(fp, mode='a') as file:
        file.write('%s\n' % (src))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.1)


def rescale(tensor, mean, std):
    #TODO: add args for mean and std for diff dsets
    for t, m, s in zip(tensor, mean, std):
        t = t.data.mul_(s).add_(m)
    return tensor


def find_boundaries(train_loader):
    curr_max = 0
    curr_min = int(1e3)
    for batch_idx, (data, cls) in tqdm(enumerate(train_loader)):
        batch_size = data.size(0)
        data = Variable(data).cuda()
        prop_max = torch.max(data).data[0]
        if prop_max > curr_max:
            curr_max = prop_max
        prop_min = torch.min(data).data[0]
        if prop_min < curr_min:
            curr_min = prop_min
    return curr_min, curr_max 

