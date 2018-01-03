#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from semseg.loss import cross_entropy2d

from semseg.modelloader.pspnet1 import pspnet1
from semseg.modelloader.pspnet2 import pspnet2


def pspnet(**kwargs):
    # pspnet1有问题，目前使用pspnet2
    model = pspnet2(**kwargs)
    return model

if __name__ == '__main__':
    n_classes = 21
    model = pspnet(n_classes=n_classes, pretrained=False, use_aux=False)
    # model.init_vgg16()
    x = Variable(torch.randn(1, 3, 360, 480))
    y = Variable(torch.LongTensor(np.ones((1, 360, 480), dtype=np.int)))
    # print(x.shape)

    # ---------------------------pspnet模型运行时间-----------------------
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)

    print(pred.data.size())
    loss = cross_entropy2d(pred, y)
    print(loss)
