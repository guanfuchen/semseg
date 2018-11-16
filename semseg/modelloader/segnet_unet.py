#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 参考[Seg-UNet](https://github.com/ykamikawa/Seg-UNet/blob/master/model.py)Tensorflow实现
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.models.squeezenet import Fire

from semseg.loss import cross_entropy2d
from semseg.modelloader.utils import segnetUNetDown2, segnetUNetDown3, segnetUNetUp2, segnetUNetUp3, \
    conv2DBatchNormRelu, segnetDown2, segnetUp2, segnetDown3, segnetUp3


class segnet_unet(nn.Module):
    def __init__(self, n_classes=21, pretrained=False):
        super(segnet_unet, self).__init__()
        # self.down1 = segnetUNetDown2(3, 64)
        self.down1 = segnetDown2(3, 64)
        # self.down2 = segnetUNetDown2(64, 128)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetUNetDown3(256, 512)
        self.down5 = segnetUNetDown3(512, 512)

        self.up5 = segnetUNetUp3(512, 512)
        self.up4 = segnetUNetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        # self.up1 = segnetUNetUp2(64, n_classes)
        self.up1 = segnetUp2(64, n_classes)

        self.init_vgg16(pretrained=pretrained)

    def forward(self, x):
        x_down1, pool_indices1, unpool_shape1 = self.down1(x)
        x_down2, pool_indices2, unpool_shape2 = self.down2(x_down1)
        x_down3, pool_indices3, unpool_shape3 = self.down3(x_down2)
        x_down4, pool_indices4, unpool_shape4, x_undown4 = self.down4(x_down3)
        x_down5, pool_indices5, unpool_shape5, x_undown5 = self.down5(x_down4)

        x_up5 = self.up5(x_down5, pool_indices=pool_indices5, unpool_shape=unpool_shape5, concat_net=x_undown5)
        x_up4 = self.up4(x_up5, pool_indices=pool_indices4, unpool_shape=unpool_shape4, concat_net=x_undown4)
        x_up3 = self.up3(x_up4, pool_indices=pool_indices3, unpool_shape=unpool_shape3)
        x_up2 = self.up2(x_up3, pool_indices=pool_indices2, unpool_shape=unpool_shape2)
        x_up1 = self.up1(x_up2, pool_indices=pool_indices1, unpool_shape=unpool_shape1)
        return x_up1

    def init_vgg16(self, pretrained=False):
        vgg16 = models.vgg16(pretrained=pretrained)

        # -----------赋值前面2+2+3+3+3层feature的特征-------------
        # 由于vgg16的特征是Sequential，获得其中的子类通过children()
        vgg16_features = list(vgg16.features.children())
        vgg16_conv_layers = []
        for layer in vgg16_features:
            if isinstance(layer, nn.Conv2d):
                # print(layer)
                vgg16_conv_layers.append(layer)


        conv_blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]
        conv_ids_vgg = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 30]]

        segnet_down_conv_layers = []
        for conv_block_id, conv_block in enumerate(conv_blocks):
            # print(conv_block_id)
            # print(conv_block)
            conv_block_children =  list(conv_block.children())
            for conv_block_child in conv_block_children:
                if isinstance(conv_block_child, conv2DBatchNormRelu):
                    # print(conv_block_child)
                    if hasattr(conv_block_child, 'cbr_seq'):
                        # print(conv_block_child.cbr_seq)
                        layer_lists = list(conv_block_child.cbr_seq)
                        for layer in conv_block_child.cbr_seq:
                            # print(layer)
                            if isinstance(layer, nn.Conv2d):
                                # print(layer)
                                segnet_down_conv_layers.append(layer)

        # print('len(segnet_down_conv_layers):', len(segnet_down_conv_layers))
        # print('len(vgg16_conv_layers)', len(vgg16_conv_layers))

        for l1, l2 in zip(segnet_down_conv_layers, vgg16_conv_layers):
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
    model = segnet_unet(n_classes=n_classes, pretrained=False)
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
    torch.save(model.state_dict(), '/tmp/tmp.pt')
