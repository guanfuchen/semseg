# -*- coding: utf-8 -*-
# !!!code is from [BiSeNet](https://github.com/ooooverflow/BiSeNet/blob/master/model/build_BiSeNet.py)!!!
# please refer the origin implement and I will just use the code for my own usage

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import numpy as np
import time

# ----Context Module----
class resnet18(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet18, self).__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32

        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        # 16 and 32 down features are used with ARM
        return feature3, feature4, tail


class resnet101(nn.Module):
    def __init__(self, pretrained=True):
        super(resnet101, self).__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


def Context_path(name, pretrained=False):
    if name=='resnet18':
        return resnet18(pretrained=pretrained)
    elif name=='resnet101':
        return resnet101(pretrained=pretrained)

# ----Context Module----

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x= self.relu(x)
        return x

class Spatial_path(nn.Module):
    def __init__(self):
        """
        Spatial Path is combined by 3 blocks including Conv+BN+ReLU, and here every block is 2 stride
        """
        super(Spatial_path, self).__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels

    def forward(self, input):
        # global average pooling
        x = torch.mean(input, 3, keepdim=True)
        # print('input.shape:', input.shape)
        # print('x.shape:', x.shape)
        x = torch.mean(x, 2, keepdim=True)
        # print('input.shape:', input.shape)
        # print('x.shape:', x.shape)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.bn(x)
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(FeatureFusionModule, self).__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # self.in_channels = 3328
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = torch.mean(feature, 3, keepdim=True)
        x = torch.mean(x, 2 ,keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.relu(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class BiSeNet(nn.Module):
    def __init__(self, n_classes=21, pretrained=True, context_path='resnet18'):
        super(BiSeNet, self).__init__()
        self.n_classes = n_classes

        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = Context_path(name=context_path, pretrained=pretrained)

        # build attention refinement module
        if context_path=='resnet18':
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
        elif context_path=='resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)

        # build feature fusion module
        if context_path=='resnet18':
            self.feature_fusion_module = FeatureFusionModule(self.n_classes, in_channels=1024)
        elif context_path=='resnet101':
            self.feature_fusion_module = FeatureFusionModule(self.n_classes, in_channels=3328)

        # build final convolution
        self.conv = nn.Conv2d(in_channels=self.n_classes, out_channels=self.n_classes, kernel_size=1)

    def forward(self, input):
        input_size = input.size()
        # output of spatial path
        sx = self.saptial_path(input)
        # print('sx.shape:', sx.shape)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        # print('cx1.shape:', cx1.shape)
        # print('cx2.shape:', cx2.shape)
        # print('tail.shape:', tail.shape)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = F.upsample_bilinear(cx1, (input_size[2]//8, input_size[3]//8))
        cx2 = F.upsample_bilinear(cx2, (input_size[2]//8, input_size[3]//8))
        # print('cx1.shape:', cx1.shape)
        # print('cx2.shape:', cx2.shape)
        cx = torch.cat((cx1, cx2), dim=1)
        # print('cx.shape:', cx.shape)

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = F.upsample_bilinear(result, scale_factor=8)
        result = self.conv(result)
        return result

if __name__ == '__main__':
    batch_size = 1
    n_classes = 12
    img_height, img_width = 360, 480
    # img_height, img_width = 1024, 512
    model = BiSeNet(n_classes=n_classes, pretrained=False, context_path='resnet18')

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
