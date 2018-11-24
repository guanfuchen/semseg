#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  stride, padding, bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_seq = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(int(out_channels)),
        )

    def forward(self, inputs):
        outputs = self.cb_seq(inputs)
        return outputs

class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()
        self.cbr_seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cbr_seq(x)
        return x

class unetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetDown, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class unetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unetUp, self).__init__()
        self.upConv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )


    def forward(self, x_cur, x_prev):
        x = self.upConv(x_cur)
        x = torch.cat([F.upsample_bilinear(x_prev, size=x.size()[2:]), x], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class segnetDown2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpool_shape = x.size()
        # print(unpool_shape)
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape


class segnetDown3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpool_shape = x.size()
        # print(unpool_shape)
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape

class segnetDown4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetDown4, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        unpool_shape = x.size()
        # print(unpool_shape)
        x, pool_indices = self.max_pool(x)
        return x, pool_indices, unpool_shape


class segnetUp2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUp2, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class segnetUNetDown2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUNetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        unpool_shape = x.size()
        # print(unpool_shape)
        x_pool, pool_indices = self.max_pool(x)
        return x_pool, pool_indices, unpool_shape, x


class segnetUNetDown3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUNetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        unpool_shape = x.size()
        # print(unpool_shape)
        x_pool, pool_indices = self.max_pool(x)
        return x_pool, pool_indices, unpool_shape, x

class segnetUp3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUp3, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class segnetUp4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUp4, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class segnetUNetUp2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUNetUp2, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        # self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape, concat_net):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        # print('concat_net.size():', concat_net.size())
        # print('x.size():', x.size())
        x = torch.cat([concat_net, x], 1)
        # x = concat_net+x
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class segnetUNetUp3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(segnetUNetUp3, self).__init__()
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv1 = conv2DBatchNormRelu(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        # self.conv1 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        pass

    def forward(self, x, pool_indices, unpool_shape, concat_net):
        x = self.max_unpool(x, indices=pool_indices, output_size=unpool_shape)
        # print('concat_net.size():', concat_net.size())
        # print('x.size():', x.size())
        x = torch.cat([concat_net, x], 1)
        # x = x + concat_net
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(residualBlock, self).__init__()

        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, out_channels, 3,  stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(out_channels, out_channels, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class linknetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(linknetUp, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, out_channels/2, kernel_size=1, stride=1, padding=1)

        # B, C/2, H, W -> B, C/2, H, W
        self.deconvbnrelu2 = deconv2DBatchNormRelu(out_channels/2, out_channels/2, kernel_size=3,  stride=2, padding=0,)

        # B, C/2, H, W -> B, C, H, W
        self.convbnrelu3 = conv2DBatchNormRelu(out_channels/2, out_channels, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x

class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(out_channels)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1):
        super(bottleNeckIdentifyPSP, self).__init__()

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, 1, 0, bias=False)
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, 1,
                                            padding=dilation, bias=False,
                                            dilation=dilation)
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                            stride=1, padding=1,
                                            bias=False, dilation=1)
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x + residual, inplace=True)

class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 stride, dilation=1):
        super(bottleNeckPSP, self).__init__()

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, 1, 0, bias=False)
        if dilation > 1:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3, 1,
                                            padding=dilation, bias=False,
                                            dilation=dilation)
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                            stride=stride, padding=1,
                                            bias=False, dilation=1)
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, 1, 0)
        self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv + residual, inplace=True)


class residualBlockPSP(nn.Module):
    
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation=1):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        layers = [bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation)]
        for i in range(n_blocks):
            layers.append(bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class pyramidPooling(nn.Module):
    """
    金字塔池化模块，主要是将特征图通过不同的pool构造池化输出，最后的输出是和输入相同分辨率大小的concat的特征图
    """

    def __init__(self, in_channels, pool_sizes):
        super(pyramidPooling, self).__init__()

        self.paths = []

        for i in range(len(pool_sizes)):
            # 1*1卷积输出为in_channels/level的
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        # 输出已经默认包括x
        output_slices = [x]
        # 输出宽度需要和x相同
        h, w = x.view()[2:]

        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            # 金字塔池化操作，分别对每一个module和sizes进行操作
            # 首先使用平均池化层操作获得相应size的池化特征图
            out = F.avg_pool2d(x, pool_size, stride=1, padding=0)
            # 通过module减小维度降低计算量
            out = module(out)
            # 然后上采样即可
            out = F.upsample(out, size=(h,w), mode='bilinear')
            output_slices.append(out)

        # 最后把[x, pool1_up, pool2_up, pool3_up, pool4_up] concat即可
        return torch.cat(output_slices, dim=1)


class AlignedResInception(nn.Module):
    """
    Aligned残差Inception结构
    """
    def __init__(self, in_planes, stride=1):
        super(AlignedResInception, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//4, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_planes//4),
            nn.ReLU(True),
            nn.Conv2d(in_planes//4, in_planes//4, kernel_size=3, stride=1, padding=1),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//8, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_planes//8),
            nn.ReLU(True),
            nn.Conv2d(in_planes//8, in_planes//8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_planes//8),
            nn.ReLU(True),
            nn.Conv2d(in_planes//8, in_planes//8, kernel_size=3, stride=1, padding=1),
        )

        self.b3 = nn.Sequential(
            nn.BatchNorm2d(in_planes//8*3),
            nn.ReLU(True),
        )

        self.b4 = nn.Sequential(
            nn.Conv2d(in_planes//8*3, in_planes, kernel_size=1, stride=stride),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
        )

        self.downsample = None
        if stride>1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_planes),
            )

    def forward(self, x):
        y1 = self.b1(x)
        # print('y1.size():', y1.size())
        y2 = self.b2(x)
        # print('y2.size():', y2.size())
        y3 = torch.cat([y1,y2], 1)
        y3 = self.b3(y3)
        # print('y3.size():', y3.size())
        out = self.b4(y3)
        # print('out.size():', out.size())
        if self.downsample is not None:
            out = out + self.downsample(x)
        else:
            out = out + x
        out = self.relu(out)
        return out

class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

class ResInception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, stride=1):
        super(ResInception, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1, stride=stride),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1, stride=stride),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1, stride=stride),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

        self.downsample = None
        if stride>1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_planes),
            )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        out = torch.cat([y1,y2,y3,y4], 1)
        if self.downsample is not None:
            out = out + self.downsample(x)
        else:
            out = out + x
        out = self.relu(out)
        return out


class CascadeResInception(nn.Module):
    def __init__(self):
        super(CascadeResInception, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResInception(512, 128, 128, 256, 24,  64,  64, stride=2)
        self.res2 = ResInception(512, 128, 128, 256, 24,  64,  64, stride=7)
        self.res3 = ResInception(512, 128, 128, 256, 24,  64,  64, stride=14)

    def forward(self, x):
        y1 = self.res1(x)
        y2 = self.res2(x)
        y3 = self.res3(x)
        # print('y1:', y1.size())
        # print('y2:', y2.size())
        # print('y3:', y3.size())
        y1 = F.upsample_bilinear(y1, x.size()[2:])
        y2 = F.upsample_bilinear(y2, x.size()[2:])
        y3 = F.upsample_bilinear(y3, x.size()[2:])
        out = x + y1 + y2 + y3
        out = self.relu(out)
        return out

class CascadeAlignedResInception(nn.Module):
    def __init__(self, in_planes):
        super(CascadeAlignedResInception, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.res1 = AlignedResInception(in_planes=in_planes, stride=2)
        self.res2 = AlignedResInception(in_planes=in_planes, stride=7)
        self.res3 = AlignedResInception(in_planes=in_planes, stride=14)

    def forward(self, x):
        y1 = self.res1(x)
        y2 = self.res2(x)
        y3 = self.res3(x)
        # print('y1:', y1.size())
        # print('y2:', y2.size())
        # print('y3:', y3.size())
        y1 = F.upsample_bilinear(y1, x.size()[2:])
        y2 = F.upsample_bilinear(y2, x.size()[2:])
        y3 = F.upsample_bilinear(y3, x.size()[2:])
        out = x + y1 + y2 + y3
        out = self.relu(out)
        return out

class ASPP_Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, n_classes, in_channels=2048):
        super(ASPP_Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(in_channels, n_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out
