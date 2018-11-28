# -*- coding: utf-8 -*-
# code is from [schedulers.py](https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/schedulers/schedulers.py)

import torch

from torch.optim.lr_scheduler import _LRScheduler

class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
