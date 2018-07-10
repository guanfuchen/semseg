# -*- coding: utf-8 -*-
import torch.nn.functional as F

def cross_entropy2d(input, target, weight=None, size_average=True):
    # print(input.data.size())
    # print(target.data.size())
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # print(log_p.size())
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    # print(log_p.data.size())
    # print(target.data.size())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss
