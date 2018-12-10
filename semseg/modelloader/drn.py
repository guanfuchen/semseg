#!/usr/bin/python
# -*- coding: UTF-8 -*-
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
from semseg.modelloader.utils import AlignedResInception, ASPP_Classifier_Module, IBN

# from semseg.pytorch_modelsize import SizeEstimator

webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)

def conv3x3_asymmetric(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=stride, padding=(padding, 0), bias=False, dilation=dilation),
        nn.Conv2d(out_planes, out_planes, kernel_size=(1, 3), stride=1, padding=(0, padding), bias=False, dilation=dilation),
    )

# drn基本构成块
class BasicBlock_asymmetric(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(BasicBlock_asymmetric, self).__init__()
        # dilation默认为(1,1)由两个dilation的卷积模块构成，由于stride=1，dilation为1，kernel为3
        # 那么相当于kernel为6的卷积核，padding为1
        # self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        self.conv1 = conv3x3_asymmetric(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.conv2 = conv3x3_asymmetric(planes, planes, padding=dilation[1], dilation=dilation[1])
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

class BasicBlock_asymmetric_ibn_a(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(BasicBlock_asymmetric_ibn_a, self).__init__()
        # dilation默认为(1,1)由两个dilation的卷积模块构成，由于stride=1，dilation为1，kernel为3
        # 那么相当于kernel为6的卷积核，padding为1
        # self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        self.conv1 = conv3x3_asymmetric(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = IBN(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.conv2 = conv3x3_asymmetric(planes, planes, padding=dilation[1], dilation=dilation[1])
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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DRN(nn.Module):

    def __init__(self, block, layers, n_classes=21, channels=(16, 32, 64, 128, 256, 512, 512, 512), out_map=False, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        # print(layers)
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        # 默认架构为arch=D
        self.arch = arch

        # 不同架构主要在构成的网络模块基本组成模块上不同，在C架构上主要由basic block块组成，而其他由conv组成
        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D' or arch == 'E':
            # -7+2*3/1+1=0将channel为3的rgb原始图像数据转换为channels[0]的数据
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True)
            )

            self.layer1 = self._make_conv_layers(channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(channels[1], layers[1], stride=2)

        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4], dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else self._make_layer(block, channels[5], layers[5], dilation=4, new_level=False)

        if arch == 'C':
            # 无残差模块
            self.layer7 = None if layers[6] == 0 else self._make_layer(BasicBlock, channels[6], layers[6], dilation=2, new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else self._make_layer(BasicBlock, channels[7], layers[7], dilation=1, new_level=False, residual=False)
        elif arch == 'D' or arch == 'E':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(channels[7], layers[7], dilation=1)

        self.layer9 = None
        if arch == 'E':
            # self.layer9 = Inception(512, 128, 128, 256, 24,  64,  64)
            # self.layer9 = ResInception(512, 128, 128, 256, 24,  64,  64)
            # self.layer9 = CascadeResInception()
            # self.layer9 = CascadeAlignedResInception(in_planes=512)
            self.layer9 = AlignedResInception(in_planes=512)

        # 最后的网络输出语义图
        # if num_classes > 0:
        #     self.avgpool = nn.AvgPool2d(pool_size)
        #     self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.layer10 = None
        # self.layer10 = self._make_pred_layer(ASPP_Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], n_classes, in_channels=512*block.expansion)
        if self.layer10 is not None:
            self.out_dim = n_classes
            pass
        else:
            self.out_dim = 512 * block.expansion
            pass

        # 网络模块权重和偏置初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 这种构成网络层的方法类似于Residual Neural Network，new_level表示第一个conv block和后面的conv是否空洞率相同
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(
            block(self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation),
            residual=residual
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual, dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    # 创建卷积层，输入通道，卷积个数，stride，dilation等等
    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        # 创建卷积的个数，当stride为2时，即卷积有两层的情况下，输出维度为原来的1／2
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def _make_pred_layer(self, block, dilation_series, padding_series, n_classes, in_channels):
        return block(dilation_series, padding_series, n_classes, in_channels)

    def forward(self, x):
        # y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D' or self.arch == 'E':
            x = self.layer0(x)

        x = self.layer1(x)
        # y.append(x)
        x = self.layer2(x)
        # y.append(x)

        x = self.layer3(x)
        # y.append(x)

        x = self.layer4(x)
        # y.append(x)

        x = self.layer5(x)
        # y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            # y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            # y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            # y.append(x)

        # DRN E
        if self.layer9 is not None:
            x = self.layer9(x)
            # y.append(x)

        # ASPP
        if self.layer10 is not None:
            x = self.layer10(x)
            # y.append(x)

        # if self.out_map:
        #     x = self.fc(x)
        # else:
        #     x = self.avgpool(x)
        #     x = self.fc(x)
        #     x = x.view(x.size(0), -1)

        # if self.out_middle:
        #     return x, y
        # else:
        #     return x
        return x

class DRN_A(nn.Module):

    def __init__(self, block, layers, n_classes=21):
        self.inplanes = 64
        super(DRN_A, self).__init__()
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
        # self.avgpool = nn.AvgPool2d(28, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.layer5 = self._make_pred_layer(ASPP_Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], n_classes, in_channels=512*block.expansion)
        if self.layer5 is not None:
            self.out_dim = n_classes
            pass
        else:
            self.out_dim = 512 * block.expansion
            pass

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

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
            # print('blocks_i:', i)
            layers.append(block(self.inplanes, planes,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, n_classes, in_channels):
        return block(dilation_series, padding_series, n_classes, in_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # print('x.size():', x.size())
        x = self.layer5(x)
        # print('x.size():', x.size())

        return x

def drn_a_50(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'))
    return model

def drn_a_18(pretrained=False, **kwargs):
    model = DRN_A(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'))
    return model

def drn_a_n(pretrained=False, depth_n=18, **kwargs):
    model = DRN_A(BasicBlock, [2+depth_n-18, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'))
    return model

def drn_a_asymmetric_18(pretrained=False, **kwargs):
    model = DRN_A(BasicBlock_asymmetric, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'))
    return model

def drn_a_asymmetric_n(pretrained=False, depth_n=18, **kwargs):
    # print('depth_n:', depth_n)
    model = DRN_A(BasicBlock_asymmetric, [2+depth_n-18, 2, 2, 2], **kwargs)
    return model

def drn_a_asymmetric_ibn_a_18(pretrained=False, **kwargs):
    model = DRN_A(BasicBlock_asymmetric_ibn_a, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'))
    return model

def drn_a_34(pretrained=False, **kwargs):
    model = DRN_A(BasicBlock, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'))
    return model

def drn_a_asymmetric_34(pretrained=False, **kwargs):
    model = DRN_A(BasicBlock_asymmetric, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'))
    return model

def drn_c_26(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-c-26']))
    return model


def drn_c_42(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-c-42']))
    return model


def drn_c_58(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-c-58']))
    return model

# drn变种22
def drn_d_22(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
    return model

def drn_d_24(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-24']))
    return model


def drn_d_38(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    if pretrained:
        model_checkpoint_path = os.path.expanduser('~/.torch/models/drn_d_38-eebb45f0.pth')
        if os.path.exists(model_checkpoint_path):
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_checkpoint_path, map_location='cpu')
            # print('model_dict:', model_dict.keys())
            # print('pretrained_dict:', pretrained_dict.keys())
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            # print('new_dict:', new_dict.keys())
            model_dict.update(new_dict)
            model.load_state_dict(model_dict)
    return model


def drn_d_40(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-40']))
    return model


def drn_d_54(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
    return model


def drn_d_56(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-56']))
    return model


def drn_d_105(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
    return model


def drn_d_107(pretrained=False, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-107']))
    return model

# drn变种22
def drn_e_22(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='E', **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
    return model

# -------------------------semantic model----------------------------------

def drnseg_a_50(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_a_50', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_a_18(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_a_18', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_a_n(pretrained=False, n_classes=21, depth_n=18):
    print('depth_n:', depth_n)
    model = DRNSeg(model_name='drn_a_n', n_classes=n_classes, pretrained=pretrained, depth_n=depth_n)
    return model

def drnseg_a_asymmetric_18(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_a_asymmetric_18', n_classes=n_classes, pretrained=pretrained)
    return model

# drnseg模型n测试
def drnseg_a_asymmetric_n(pretrained=False, n_classes=21, depth_n=18):
    print('depth_n:', depth_n)
    model = DRNSeg(model_name='drn_a_asymmetric_n', n_classes=n_classes, pretrained=pretrained, depth_n=depth_n)
    return model

def drnseg_a_asymmetric_ibn_a_18(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_a_asymmetric_ibn_a_18', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_a_34(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_a_34', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_a_asymmetric_34(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_a_asymmetric_34', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_c_26(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_c_26', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_c_42(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_c_42', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_c_58(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_c_58', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_d_22(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_d_22', n_classes=n_classes, pretrained=pretrained)
    return model


def drnseg_d_24(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_d_24', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_d_38(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_d_38', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_d_40(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_d_40', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_d_54(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_d_54', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_d_56(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_d_56', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_d_105(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_d_105', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_d_107(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_d_107', n_classes=n_classes, pretrained=pretrained)
    return model

def drnseg_e_22(pretrained=False, n_classes=21):
    model = DRNSeg(model_name='drn_e_22', n_classes=n_classes, pretrained=pretrained)
    return model

# -----------------------------------------------------------

# 转置卷积权重初始化填充方法
def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

# drn segnet network
class DRNSeg(nn.Module):
    def __init__(self, model_name, n_classes, pretrained=False, use_torch_up=True, depth_n=-1):
        super(DRNSeg, self).__init__()
        # DRN分割模型不同变种
        # if model_name=='drn_d_22':
        #     model = drn_d_22(pretrained=pretrained, num_classes=1000)
        # if model_name=='drn_a_50':
        #     model = drn_a_50(pretrained=pretrained, num_classes=1000)
        # if model_name=='drn_a_18':
        #     model = drn_a_18(pretrained=pretrained, num_classes=1000)
        # if model_name=='drn_e_22':
        #     model = drn_e_22(pretrained=pretrained, num_classes=1000)

        if model_name=='drn_a_asymmetric_n':
            # print('depth_n:', depth_n)
            model = drn_a_asymmetric_n(pretrained=pretrained, n_classes=n_classes, depth_n=depth_n)
        elif model_name=='drn_a_n':
            # print('depth_n:', depth_n)
            model = drn_a_n(pretrained=pretrained, n_classes=n_classes, depth_n=depth_n)
        else:
            model = eval(model_name)(pretrained=pretrained, n_classes=n_classes)
        # pmodel = nn.DataParallel(model)
        # if pretrained_model is not None:
            # pmodel.load_state_dict(pretrained_model)
        # self.base = nn.Sequential(*list(model.children())[:-2])
        # self.base = nn.Sequential(*list(model.children()))
        self.base = model

        # 仅仅在最后一层seg layer上存有bias
        self.seg = nn.Conv2d(model.out_dim, n_classes, kernel_size=1)
        # self.softmax = nn.LogSoftmax()
        m = self.seg

        # 初始化分割图最后的卷积weights和bias
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()

        if use_torch_up:
            # 使用pytorch双线性上采样
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            # 使用转置卷积上采样
            up = nn.ConvTranspose2d(n_classes, n_classes, 16, stride=8, padding=4, output_padding=0, groups=n_classes, bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

        if pretrained and n_classes == 19 and model_name=='drn_d_38':
            model_checkpoint_path = os.path.expanduser('~/.torch/models/drn_d_38_cityscapes.pth')
            if os.path.exists(model_checkpoint_path):
                model_dict = self.state_dict()
                pretrained_dict = torch.load(model_checkpoint_path, map_location='cpu')
                # print('model_dict:', model_dict.keys()[:3])
                # print('pretrained_dict:', pretrained_dict.keys()[:3])
                # print('len(model_dict):', len(model_dict.keys()))
                # print('len(pretrained_dict):', len(pretrained_dict.keys()))
                # new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
                new_dict = {}
                for k, v in pretrained_dict.items():
                    if k.find('base.') != -1:
                        new_k = str('base.' + 'layer' + k[k.find('.') + 1:])
                        if new_k not in model_dict.keys():
                            print(new_k)
                        new_v = v
                        new_dict[new_k] = new_v
                    else:
                        # print(k)
                        new_k = k
                        new_v = v
                        new_dict[new_k] = new_v
                # print('new_dict:', new_dict.keys()[:3])
                # print('len(new_dict):', len(new_dict.keys()))
                model_dict.update(new_dict)
                self.load_state_dict(model_dict)
        if pretrained and model_name=='drn_a_18':
            model_checkpoint_path = os.path.expanduser('~/GitHub/Quick/semseg/best.pth')
            if os.path.exists(model_checkpoint_path):
                model_dict = self.state_dict()
                pretrained_dict = torch.load(model_checkpoint_path, map_location='cpu')
                model_dict_keys = model_dict.keys()
                # print('model_dict:', model_dict.keys()[:3])
                # print('pretrained_dict:', pretrained_dict.keys()[:3])
                # print('len(model_dict):', len(model_dict.keys()))
                # print('len(pretrained_dict):', len(pretrained_dict.keys()))
                new_dict = {}
                for k, v in pretrained_dict.items():
                    if 'base.{}'.format(k) in model_dict_keys:
                        new_k = 'base.{}'.format(k)
                        new_v = v
                        new_dict[new_k] = new_v
                    else:
                        pass
                        # print(k)
                    # if k.find('base.') != -1:
                    #     new_k = str('base.' + 'layer' + k[k.find('.') + 1:])
                    #     if new_k not in model_dict.keys():
                    #         print(new_k)
                    #     new_v = v
                    #     new_dict[new_k] = new_v
                    # else:
                    #     # print(k)
                    #     new_k = k
                    #     new_v = v
                    #     new_dict[new_k] = new_v
                # print('new_dict:', new_dict.keys()[:3])
                new_dict = {k: v for k, v in new_dict.items() if k in model_dict.keys()}
                # print('new_dict:', new_dict.keys())
                # print('len(new_dict):', len(new_dict.keys()))
                model_dict.update(new_dict)
                self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.base(x)

        # 将分割图对应到分割类别数上
        x = self.seg(x)

        # 使用双线性上采样或者转置卷积上采样8倍降采样率的分割图
        y = self.up(x)
        return y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

if __name__ == '__main__':
    n_classes = 21
    model = DRNSeg(model_name='drn_a_asymmetric_18', n_classes=n_classes, pretrained=False)
    # model = DRNSeg(model_name='drn_d_22', n_classes=n_classes, pretrained=False)
    # model.eval()
    # model.init_vgg16()
    x = Variable(torch.randn(1, 3, 360, 480))
    y = Variable(torch.LongTensor(np.ones((1, 360, 480), dtype=np.int)))
    # x = Variable(torch.randn(1, 3, 512, 1024))
    # y = Variable(torch.LongTensor(np.ones((1, 512, 1024), dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    print(pred.shape)
    loss = cross_entropy2d(pred, y)
    print(loss)

    # se = SizeEstimator(model, input_size=(1, 3, 360, 480))
    # print(se.estimate_size())
