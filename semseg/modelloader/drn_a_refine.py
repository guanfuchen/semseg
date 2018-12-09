# -*- coding: utf-8 -*-
import time
import os
import math
from torch.utils import model_zoo
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from semseg.loss import cross_entropy2d


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)

def conv3x3_bn_relu(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

# drn基本构成块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        # dilation默认为(1,1)由两个dilation的卷积模块构成，由于stride=1，dilation为1，kernel为3
        # 那么相当于kernel为6的卷积核，padding为1
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        # self.conv1 = conv3x3_asymmetric(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        # self.conv2 = conv3x3_asymmetric(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        # print(x.data.size())
        out = self.conv1(x)
        # print(out.data.size())
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

def drnsegrefine_a_18(pretrained=False, n_classes=21):
    model = DRNSegRefine_A(BasicBlock, [2, 2, 2, 2], n_classes=n_classes)
    return model

class RefineUnit(nn.Module):
    def __init__(self, f2_channel, n_classes=21):
        super(RefineUnit, self).__init__()
        self.f2_channel = f2_channel
        self.n_classes = n_classes
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, padding=1, output_padding=0, groups=n_classes, bias=False)
        self.out_conv_1 = conv3x3_bn_relu(in_planes=self.f2_channel, out_planes=self.n_classes)
        # self.out_conv_2 = conv3x3_bn_relu(in_planes=2*self.n_classes, out_planes=self.n_classes)
        self.out_conv_2 = conv3x3_bn_relu(in_planes=self.n_classes, out_planes=self.n_classes)

    def forward(self, f3, f2):
        # print('f3.shape:', f3.shape)
        # print('f2.shape:', f2.shape)
        m1 = self.up(f3)
        f2_1 = self.out_conv_1(f2)
        # m2 = torch.cat((m1, f2_1), 1)
        m2 = m1 + f2_1
        o2 = self.out_conv_2(m2)
        return o2

class DRNSegRefine_A(nn.Module):

    def __init__(self, block, layers, n_classes=21):
        super(DRNSegRefine_A, self).__init__()
        self.inplanes = 64
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.out_conv = nn.Conv2d(self.out_dim, n_classes, kernel_size=1)


        # ----no refine----
        self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        # ----no refine----

        # ----refine 1----
        # self.refineunit_1 = RefineUnit(64, n_classes)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        # ----refine 1----

        # ----refine 2----
        self.refineunit_1 = RefineUnit(64, n_classes)
        self.refineunit_2 = RefineUnit(64, n_classes)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # ----refine 2----

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x = self.bn1(x_conv1)
        x = self.relu(x)
        x_pool = self.maxpool(x)

        x_layer1 = self.layer1(x_pool)
        x_layer2 = self.layer2(x_layer1)
        x = self.layer3(x_layer2)
        x = self.layer4(x)

        x = self.out_conv(x)

        # ----no refine----
        # x = self.up(x)
        # ----no refine----

        # ----refine 1----
        # x_refine1 = self.refineunit_1(x, x_layer1)
        # x = self.up(x_refine1)
        # ----refine 1----

        # ----refine 2----
        x_refine1 = self.refineunit_1(x, x_layer1)
        x_refine2 = self.refineunit_2(x_refine1, x_conv1)
        x = self.up(x_refine2)
        # ----refine 2----

        return x
