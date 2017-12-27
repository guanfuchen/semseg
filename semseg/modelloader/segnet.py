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
from semseg.modelloader.utils import segnetDown2, segnetDown3, segnetUp2, segnetUp3


class segnet(nn.Module):
    def __init__(self, n_classes=21, pretrained=False):
        super(segnet, self).__init__()
        self.down1 = segnetDown2(3, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, n_classes)

        # if pretrained:
        #     self.init_vgg16()

    def forward(self, x):
        x, pool_indices1, unpool_shape1 = self.down1(x)
        x, pool_indices2, unpool_shape2 = self.down2(x)
        x, pool_indices3, unpool_shape3 = self.down3(x)
        x, pool_indices4, unpool_shape4 = self.down4(x)
        x, pool_indices5, unpool_shape5 = self.down5(x)

        x = self.up5(x, pool_indices=pool_indices5, unpool_shape=unpool_shape5)
        x = self.up4(x, pool_indices=pool_indices4, unpool_shape=unpool_shape4)
        x = self.up3(x, pool_indices=pool_indices3, unpool_shape=unpool_shape3)
        x = self.up2(x, pool_indices=pool_indices2, unpool_shape=unpool_shape2)
        x = self.up1(x, pool_indices=pool_indices1, unpool_shape=unpool_shape1)
        return x

    def init_vgg16(self):
        vgg16 = models.vgg16(pretrained=True)

        # -----------赋值前面2+2+3+3+3层feature的特征-------------
        # 由于vgg16的特征是Sequential，获得其中的子类通过children()
        vgg16_features = list(vgg16.features.children())

        conv_blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]
        conv_ids_vgg = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 30]]

        for conv_block_id, conv_block in enumerate(conv_blocks):
            # print(conv_block_id)
            conv_id_vgg = conv_ids_vgg[conv_block_id]
            for l1, l2 in zip(conv_block, vgg16_features[conv_id_vgg[0]:conv_id_vgg[1]]):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    # 赋值的是数据
                    l1.weight.data = l2.weight.data
                    l1.bias.data = l2.bias.data
                    # print(l1)
                    # print(l2)

if __name__ == '__main__':
    n_classes = 21
    model = segnet(n_classes=n_classes, pretrained=False)
    # model.init_vgg16()
    x = Variable(torch.randn(1, 3, 360, 480))
    y = Variable(torch.LongTensor(np.ones((1, 360, 480), dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    # print(pred.shape)
    loss = cross_entropy2d(pred, y)
    # print(loss)
