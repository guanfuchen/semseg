# -*- coding: utf-8 -*-
# !!!code is from [frrn.py](https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/frrn.py)!!!

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import time
import numpy as np

from semseg.modelloader.utils import conv2DGroupNormRelu, conv2DBatchNormRelu, conv2DGroupNorm, conv2DBatchNorm

frrn_specs_dic = {
    "A": {
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16]],
        "decoder": [[2, 192, 8], [2, 192, 4], [2, 48, 2]],
    },
    "B": {
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16], [2, 384, 32]],
        "decoder": [[2, 192, 16], [2, 192, 8], [2, 192, 4], [2, 48, 2]],
    },
}


class RU(nn.Module):
    """
    Residual Unit for FRRN
    """

    def __init__(self, channels, kernel_size=3, strides=1, group_norm=False, n_groups=None):
        super(RU, self).__init__()
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(channels, channels, kernel_size=kernel_size, stride=strides, padding=1, bias=False,n_groups=self.n_groups)
            self.conv2 = conv2DGroupNorm(channels, channels, kernel_size=kernel_size, stride=strides, padding=1, bias=False,n_groups=self.n_groups)

        else:
            self.conv1 = conv2DBatchNormRelu(channels, channels, kernel_size=kernel_size, stride=strides, padding=1, bias=False,)
            self.conv2 = conv2DBatchNorm(channels, channels, kernel_size=kernel_size, stride=strides, padding=1, bias=False,)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming

class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self, prev_channels, out_channels, scale, group_norm=False, n_groups=None):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.n_groups = n_groups


        if self.group_norm:
            conv_unit = conv2DGroupNormRelu
            self.conv1 = conv_unit(prev_channels + 32, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_groups=self.n_groups)
            self.conv2 = conv_unit(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_groups=self.n_groups)

        else:
            conv_unit = conv2DBatchNormRelu
            self.conv1 = conv_unit(prev_channels + 32, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = conv_unit(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        upsample_size = torch.Size([_s * self.scale for _s in y_prime.shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode="nearest")
        # print('z.shape:', z.shape)
        # print('x.shape:', x.shape)
        z_prime = z + x

        return y_prime, z_prime


class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323
    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self,
                 n_classes=21,
                 model_type=None,
                 group_norm=False,
                 n_groups=16):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(3, 48, 5, 1, 2)
        else:
            self.conv1 = conv2DBatchNormRelu(3, 48, 5, 1, 2)

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(RU(channels=48,
                                             kernel_size=3,
                                             strides=1,
                                             group_norm=self.group_norm,
                                             n_groups=self.n_groups))
            self.down_residual_units.append(RU(channels=48,
                                               kernel_size=3,
                                               strides=1,
                                               group_norm=self.group_norm,
                                               n_groups=self.n_groups))

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv = nn.Conv2d(
            48, 32, kernel_size=1, padding=0, stride=1, bias=False
        )

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]["encoder"]

        self.decoder_frru_specs = frrn_specs_dic[self.model_type]["decoder"]

        # encoding
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels,
                                        out_channels=channels,
                                        scale=scale,
                                        group_norm=self.group_norm,
                                        n_groups=self.n_groups),)
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels,
                                        out_channels=channels,
                                        scale=scale,
                                        group_norm=self.group_norm,
                                        n_groups=self.n_groups),)
            prev_channels = channels

        self.merge_conv = nn.Conv2d(
            prev_channels + 32, 48, kernel_size=1, padding=0, stride=1, bias=False
        )

        self.classif_conv = nn.Conv2d(
            48, self.n_classes, kernel_size=1, padding=0, stride=1, bias=True
        )

    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)

        # pass through residual units
        for i in range(3):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        z = self.split_conv(x)

        prev_channels = 48
        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = "_".join(
                    map(str, ["encoding_frru", n_blocks, channels, scale, block])
                )
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([_s * 2 for _s in y.size()[-2:]])
            y_upsampled = F.upsample(y, size=upsample_size, mode="bilinear", align_corners=True)
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(
                    map(str, ["decoding_frru", n_blocks, channels, scale, block])
                )
                # print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                # print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels

        # merge streams
        x = torch.cat([F.upsample(y, scale_factor=2, mode="bilinear", align_corners=True), z], dim=1)
        x = self.merge_conv(x)

        # pass through residual units
        for i in range(3):
            x = self.down_residual_units[i](x)

        # final 1x1 conv to get classification
        x = self.classif_conv(x)

        return x

def frrn_A(n_classes=21, pretrained=True):
    model = frrn(n_classes=n_classes, model_type='A')
    return model

def frrn_B(n_classes=21, pretrained=True):
    model = frrn(n_classes=n_classes, model_type='B')
    return model

if __name__ == '__main__':
    batch_size = 1
    n_classes = 21
    img_height, img_width = 360, 480
    img_height, img_width = 480, 480
    # img_height, img_width = 1024, 512
    model = frrn_A(n_classes=n_classes, pretrained=True)
    x = Variable(torch.randn(batch_size, 3, img_height, img_width))
    y = Variable(torch.LongTensor(np.ones((batch_size, img_height, img_width), dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    # print(pred.shape)
    # print('pred.type:', pred.type)
    # loss = cross_entropy2d(pred, y)
    # print(loss)
