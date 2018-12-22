# -*- coding: utf-8 -*-
import torch
from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.models import vgg
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

from semseg.loss import cross_entropy2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def deconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, output_padding=1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DeconvBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(DeconvBasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv3x3(inplanes, planes, stride)
        else:
            self.conv1 = deconv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        # print('out.shape:', out.shape)
        # print('residual.shape:', residual.shape)
        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        # print('out.shape:', out.shape)
        # print('shortcut.shape:', shortcut.shape)
        out += shortcut
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        # print('out.shape:', out.shape)
        # print('shortcut.shape:', shortcut.shape)

        out += shortcut
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dlayer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.dlayer2 = self._make_downlayer(downblock, 128, num_layers[1], stride=2)
        self.dlayer3 = self._make_downlayer(downblock, 256, num_layers[2], stride=2)
        self.dlayer4 = self._make_downlayer(downblock, 512, num_layers[3], stride=2)

        self.uplayer1 = self._make_up_block(upblock, 512, num_layers[3], stride=2)
        self.uplayer2 = self._make_up_block(upblock, 256, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(upblock, 128, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(upblock, 64, num_layers[0], stride=2)

        # print('self.in_channels:', self.in_channels)
        # upsample = nn.Sequential(
        #     nn.ConvTranspose2d(self.in_channels, self.in_channels*upblock.expansion // 2, kernel_size=1, stride=2, bias=False, output_padding=1),
        #     nn.BatchNorm2d(self.in_channels * upblock.expansion // 2),
        # )
        # self.uplayer_top = upblock(self.in_channels, self.in_channels // 2, 2, upsample)
        # self.conv1_1 = nn.ConvTranspose2d(self.in_channels // 2, n_classes, kernel_size=1, stride=1, bias=False)
        self.dconv1 = nn.ConvTranspose2d(self.in_channels, n_classes, kernel_size=1, stride=2, bias=False, output_padding=1)

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels*block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # print('init_channels:', init_channels)
        if stride != 1 or self.in_channels != init_channels * block.expansion // 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels*block.expansion // 2, kernel_size=1, stride=stride, bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels*block.expansion // 2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))
        layers.append(block(self.in_channels, init_channels // 2, stride, upsample))
        self.in_channels = init_channels * block.expansion // 2
        return nn.Sequential(*layers)

    def forward(self, x):
        # img = x
        # x_size = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dlayer1(x)
        # print('dlayer1_x.shape:', x.shape)
        x = self.dlayer2(x)
        # print('dlayer2_x.shape:', x.shape)
        x = self.dlayer3(x)
        # print('dlayer3_x.shape:', x.shape)
        x = self.dlayer4(x)
        # print('dlayer4_x.shape:', x.shape)

        x = self.uplayer1(x)
        # print('uplayer1_x.shape:', x.shape)
        x = self.uplayer2(x)
        # print('uplayer2_x.shape:', x.shape)
        x = self.uplayer3(x)
        # print('uplayer3_x.shape:', x.shape)
        x = self.uplayer4(x)
        # print('uplayer4_x.shape:', x.shape)
        # x = self.uplayer_top(x)
        # print('uplayer_top_x.shape:', x.shape)
        #
        x = self.dconv1(x)
        # print('dconv1_x.shape:', x.shape)

        return x


def DeConvResNet50(n_classes, pretrained=False):
    return ResNet(Bottleneck, DeconvBottleneck, [3, 4, 6, 3], n_classes=n_classes)

def DeConvResNet18(n_classes, pretrained=False):
    return ResNet(BasicBlock, DeconvBasicBlock, [2, 2, 2, 2], n_classes=n_classes)

def DeConvResNet34(n_classes, pretrained=False):
    return ResNet(BasicBlock, DeconvBasicBlock, [3, 4, 6, 3], n_classes=n_classes)

if __name__ == '__main__':
    batch_size = 1
    n_classes = 21
    img_height, img_width = 480, 480
    # img_height, img_width = 1024, 512
    model = DeConvResNet18(n_classes=n_classes)
    x = Variable(torch.randn(batch_size, 3, img_height, img_width))
    y = Variable(torch.LongTensor(np.ones((batch_size, img_height, img_width), dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    print(pred.shape)
    # print('pred.type:', pred.type)
    loss = cross_entropy2d(pred, y)
    print(loss)

