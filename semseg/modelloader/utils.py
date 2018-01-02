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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(conv2DBatchNormRelu, self).__init__()
        self.cbr_seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cbr_seq(x)
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

# just a try, not recommend to use
class Conv2dDeformable(nn.Module):
    def __init__(self, regular_filter, cuda=True):
        super(Conv2dDeformable, self).__init__()
        assert isinstance(regular_filter, nn.Conv2d)
        self.regular_filter = regular_filter
        self.offset_filter = nn.Conv2d(regular_filter.in_channels, 2 * regular_filter.in_channels, kernel_size=3,
                                       padding=1, bias=False)
        self.offset_filter.weight.data.normal_(0, 0.0005)
        self.input_shape = None
        self.grid_w = None
        self.grid_h = None
        self.cuda = cuda

    def forward(self, x):
        x_shape = x.size()  # (b, c, h, w)
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
        offset_w, offset_h = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w.cuda()
                grid_h = grid_h.cuda()
            self.grid_w = nn.Parameter(grid_w)
            self.grid_h = nn.Parameter(grid_h)
        offset_w = offset_w + self.grid_w  # (b*c, h, w)
        offset_h = offset_h + self.grid_h  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])).unsqueeze(1)  # (b*c, 1, h, w)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))  # (b, c, h, w)
        x = self.regular_filter(x)
        return x
