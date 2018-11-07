#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
from torchvision.models.squeezenet import Fire

from semseg.loss import cross_entropy2d
from semseg.modelloader.utils import segnetDown2, segnetDown3, segnetUp2, segnetUp3, conv2DBatchNormRelu


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

        self.init_vgg16(pretrained=pretrained)

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

# class Fire(nn.Module):
#
#     def __init__(self, inplanes, squeeze_planes,
#                  expand1x1_planes, expand3x3_planes):
#         super(Fire, self).__init__()
#         self.inplanes = inplanes
#         self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
#         self.squeeze_activation = nn.ReLU(inplace=True)
#         self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
#                                    kernel_size=1)
#         self.expand1x1_activation = nn.ReLU(inplace=True)
#         self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
#                                    kernel_size=3, padding=1)
#         self.expand3x3_activation = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.squeeze_activation(self.squeeze(x))
#         return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)

class segnet_squeeze(nn.Module):
    """
    参考论文
    Squeeze-SegNet: A new fast Deep Convolutional Neural Network for Semantic Segmentation
    """
    def __init__(self, n_classes=21, pretrained=False):
        super(segnet_squeeze, self).__init__()

        # ----下采样----
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True, return_indices=True)

        self.fire2 = Fire(96, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.fire4 = Fire(128, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True, return_indices=True)

        self.fire5 = Fire(256, 32, 128, 128)
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True, return_indices=True)

        self.fire9 = Fire(512, 64, 256, 256)

        self.conv10 = nn.Conv2d(512, 1000, kernel_size=1)
        self.relu10 = nn.ReLU(inplace=True)

        # ----上采样----
        self.conv10_D = nn.Conv2d(1000, 512, kernel_size=1)
        self.relu10_D = nn.ReLU(inplace=True)

        self.fire9_D = Fire(512, 64, 256, 256)

        self.unpool3 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.fire8_D = Fire(512, 48, 192, 192)
        self.fire7_D = Fire(384, 48, 192, 192)
        self.fire6_D = Fire(384, 32, 128, 128)
        self.fire5_D = Fire(256, 32, 128, 128)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.fire4_D = Fire(256, 16, 64, 64)
        self.fire3_D = Fire(128, 16, 64, 64)
        self.fire2_D = Fire(128, 12, 48, 48)

        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        # self.conv1_D = nn.ConvTranspose2d(96, n_classes, kernel_size=8, stride=2)
        self.conv1_D = nn.ConvTranspose2d(96, n_classes, kernel_size=10, stride=2, padding=1)

        self.init_weights(pretrained)

    def init_weights(self, pretrained=False):
        sequeeze = models.squeezenet1_0(pretrained=pretrained)

        sequeeze_features = list(sequeeze.features.children())
        sequeeze_conv_layers = []
        fire_counts = 0
        for layer in sequeeze_features:
            if isinstance(layer, nn.Conv2d):
                # print(layer)
                sequeeze_conv_layers.append(layer)
            if  isinstance(layer, Fire):
                fire_children = list(layer.children())
                # print(fire_children)
                for fire_children_layer in fire_children:
                    if isinstance(fire_children_layer, nn.Conv2d):
                        pass
                        # print(fire_children_layer)
                        sequeeze_conv_layers.append(fire_children_layer)
                # fire_counts += 1
                # if fire_counts==8:
                #     break

        segnet_squeeze_down_conv_layers= []

        features = list(self.children())
        for layer in features:
            if len(segnet_squeeze_down_conv_layers)==len(sequeeze_conv_layers):
                break
            if isinstance(layer, nn.Conv2d):
                # print(layer)
                segnet_squeeze_down_conv_layers.append(layer)
            if  isinstance(layer, Fire):
                fire_children = list(layer.children())
                # print(fire_children)
                for fire_children_layer in fire_children:
                    if isinstance(fire_children_layer, nn.Conv2d):
                        pass
                        # print(fire_children_layer)
                        segnet_squeeze_down_conv_layers.append(fire_children_layer)
        for l1, l2 in zip(segnet_squeeze_down_conv_layers, sequeeze_conv_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                # 赋值的是数据
                l1.weight.data = l2.weight.data
                l1.bias.data = l2.bias.data
                # print(l1)
                # print(l2)


    def forward(self, x):
        # ----下采样----
        x = self.conv1(x)
        x = self.relu1(x)
        unpool_shape1 = x.size()
        x, pool_indices1 = self.pool1(x)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        unpool_shape2 = x.size()
        x, pool_indices2 = self.pool2(x)

        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        unpool_shape3 = x.size()
        x, pool_indices3 = self.pool3(x)

        x = self.fire9(x)
        x = self.conv10(x)
        x = self.relu10(x)

        # ----上采样----
        x = self.conv10_D(x)
        x = self.relu10_D(x)
        x = self.fire9_D(x)

        x = self.unpool3(x, indices=pool_indices3, output_size=unpool_shape3)
        x = self.fire8_D(x)
        x = self.fire7_D(x)
        x = self.fire6_D(x)
        x = self.fire5_D(x)

        x = self.unpool2(x, indices=pool_indices2, output_size=unpool_shape2)
        x = self.fire4_D(x)
        x = self.fire3_D(x)
        x = self.fire2_D(x)

        x = self.unpool1(x, indices=pool_indices1, output_size=unpool_shape1)
        x = self.conv1_D(x)
        return x


if __name__ == '__main__':
    # n_classes = 21
    # model = segnet(n_classes=n_classes, pretrained=False)
    # # model.init_vgg16()
    # x = Variable(torch.randn(1, 3, 360, 480))
    # y = Variable(torch.LongTensor(np.ones((1, 360, 480), dtype=np.int)))
    # # print(x.shape)
    # start = time.time()
    # pred = model(x)
    # end = time.time()
    # print(end-start)
    # # print(pred.shape)
    # print('pred.type:', pred.type)
    # loss = cross_entropy2d(pred, y)
    # # print(loss)

    n_classes = 21
    model = segnet_squeeze(n_classes=n_classes, pretrained=False)
    # model.init_vgg16()
    x = Variable(torch.randn(1, 3, 360, 480))
    y = Variable(torch.LongTensor(np.ones((1, 360, 480), dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    print('pred.shape:', pred.shape)
    # print('pred.type:', pred.type)
    loss = cross_entropy2d(pred, y)
    # print(loss)
