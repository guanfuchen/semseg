# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from semseg.loss import cross_entropy2d


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class fcn_resnet(nn.Module):

    def __init__(self, block, layers, module_type='32s', n_classes=21, pretrained=False, upsample_method='upsample_bilinear'):
        """
        :param block:
        :param layers:
        :param module_type:
        :param n_classes:
        :param pretrained:
        :param upsample_method: 'upsample_bilinear' or 'ConvTranspose2d'
        """
        super(fcn_resnet, self).__init__()
        self.n_classes = n_classes
        self.module_type = module_type
        self.upsample_method = upsample_method

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.classifier = nn.Conv2d(512 * block.expansion, self.n_classes, 1)

        if self.upsample_method == 'upsample_bilinear':
            pass
        elif self.upsample_method == 'ConvTranspose2d':
            self.upsample_1 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 3, stride=2, padding=1)
            self.upsample_2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 3, stride=2)
            # self.upsample_3 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 3, stride=2, padding=1)

        if self.module_type=='16s' or self.module_type=='8s':
            self.score_pool4 = nn.Conv2d(256, self.n_classes, 1)
        if self.module_type=='8s':
            self.score_pool3 = nn.Conv2d(128, self.n_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_size = x.size()[2:]
        x_conv1 = self.conv1(x)
        x = self.bn1(x_conv1)
        x = self.relu(x)
        x = self.maxpool(x)

        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x = self.layer4(x_layer3)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        score = self.classifier(x)

        if self.module_type=='16s' or self.module_type=='8s':
            score_pool4 = self.score_pool4(x_layer3)
        if self.module_type=='8s':
            score_pool3 = self.score_pool3(x_layer2)

        if self.module_type=='16s' or self.module_type=='8s':
            # print('score_pool4.size():', score_pool4.size())
            # print('score.size():', score.size())
            if self.upsample_method == 'upsample_bilinear':
                score = F.upsample_bilinear(score, score_pool4.size()[2:])
            elif self.upsample_method == 'ConvTranspose2d':
                score = self.upsample_1(score)
            score += score_pool4
        if self.module_type=='8s':
            # print('score_pool3.size():', score_pool3.size())
            # print('score.size():', score.size())
            if self.upsample_method == 'upsample_bilinear':
                score = F.upsample_bilinear(score, score_pool3.size()[2:])
            elif self.upsample_method == 'ConvTranspose2d':
                score = self.upsample_2(score)
            score += score_pool3

        out = F.upsample_bilinear(score, x_size)

        return out

    def init_weight(self, model_name):
        pretrain_model = None
        if model_name=='fcn_resnet18':
            pretrain_model = models.resnet18(pretrained=True)
        elif model_name=='fcn_resnet34':
            pretrain_model = models.resnet34(pretrained=True)
        elif model_name=='fcn_resnet50':
            pretrain_model = models.resnet50(pretrained=True)
        elif model_name=='fcn_resnet101':
            pretrain_model = models.resnet101(pretrained=True)
        elif model_name=='fcn_resnet152':
            pretrain_model = models.resnet152(pretrained=True)
        if pretrain_model is not None:
            self.conv1.weight.data = pretrain_model.conv1.weight.data
            if self.conv1.bias is not None:
                self.conv1.bias.data = pretrain_model.conv1.bias.data

            initial_convs = []
            pretrain_model_convs = []

            layers = [self.layer1, self.layer2, self.layer3, self.layer3]
            for layer in layers:
                layer1_list = list(layer.children())
                for layer1_list_block in layer1_list:
                    layer1_list_block_list = list(layer1_list_block.children())
                    # print(layer1_list_block_list)
                    for layer1_list_item in layer1_list_block_list:
                        if isinstance(layer1_list_item, nn.Conv2d):
                            # print(layer1_list_item)
                            initial_convs.append(layer1_list_item)

            layers = [pretrain_model.layer1, pretrain_model.layer2, pretrain_model.layer3, pretrain_model.layer3]
            for layer in layers:
                layer1_list = list(layer.children())
                for layer1_list_block in layer1_list:
                    layer1_list_block_list = list(layer1_list_block.children())
                    # print(layer1_list_block_list)
                    for layer1_list_item in layer1_list_block_list:
                        if isinstance(layer1_list_item, nn.Conv2d):
                            # print(layer1_list_item)
                            pretrain_model_convs.append(layer1_list_item)

            for l1, l2 in zip(initial_convs, pretrain_model_convs):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # 赋值的是数据
                    assert l1.weight.size() == l2.weight.size()
                    l1.weight.data = l2.weight.data
                    if l1.bias is not None and l2.bias is not None:
                        assert l1.bias.size() == l2.bias.size()
                        l1.bias.data = l2.bias.data
                    # print(l1)
                    # print(l2)



def fcn_resnet18(module_type='32s', n_classes=21, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = fcn_resnet(BasicBlock, [2, 2, 2, 2], module_type=module_type, n_classes=n_classes, pretrained=pretrained)
    if pretrained:
        model.init_weight('fcn_resnet18')
    return model

def fcn_resnet18_32s(n_classes=21, pretrained=False):
    model = fcn_resnet18(module_type='32s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet18_16s(n_classes=21, pretrained=False):
    model = fcn_resnet18(module_type='16s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet18_8s(n_classes=21, pretrained=False):
    model = fcn_resnet18(module_type='8s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet34(module_type='32s', n_classes=21, pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = fcn_resnet(BasicBlock, [3, 4, 6, 3], module_type=module_type, n_classes=n_classes, pretrained=pretrained)
    if pretrained:
        model.init_weight('fcn_resnet34')
    return model

def fcn_resnet34_32s(n_classes=21, pretrained=False):
    model = fcn_resnet34(module_type='32s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet34_16s(n_classes=21, pretrained=False):
    model = fcn_resnet34(module_type='16s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet34_8s(n_classes=21, pretrained=False):
    model = fcn_resnet34(module_type='8s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet50(module_type='32s', n_classes=21, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = fcn_resnet(Bottleneck, [3, 4, 6, 3], module_type=module_type, n_classes=n_classes, pretrained=pretrained)
    if pretrained:
        model.init_weight('fcn_resnet50')
    return model

def fcn_resnet50_32s(n_classes=21, pretrained=False):
    model = fcn_resnet50(module_type='32s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet50_16s(n_classes=21, pretrained=False):
    model = fcn_resnet50(module_type='16s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet50_8s(n_classes=21, pretrained=False):
    model = fcn_resnet50(module_type='8s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_resnet101(module_type='32s', n_classes=21, pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = fcn_resnet(Bottleneck, [3, 4, 23, 3], module_type=module_type, n_classes=n_classes, pretrained=pretrained)
    if pretrained:
        model.init_weight('fcn_resnet101')
    return model


def fcn_resnet152(module_type='32s', n_classes=21, pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = fcn_resnet(Bottleneck, [3, 8, 36, 3], module_type=module_type, n_classes=n_classes, pretrained=pretrained)
    if pretrained:
        model.init_weight('fcn_resnet152')
    return model


if __name__ == '__main__':
    n_classes = 21
    model_fcn32s = fcn_resnet18(module_type='32s', n_classes=n_classes, pretrained=True)
    model_fcn16s = fcn_resnet18(module_type='16s', n_classes=n_classes, pretrained=True)
    model_fcn8s = fcn_resnet18(module_type='8s', n_classes=n_classes, pretrained=True)
    # model.init_vgg16()
    x = Variable(torch.randn(1, 3, 360, 480))
    y = Variable(torch.LongTensor(np.ones((1, 360, 480), dtype=np.int)))
    # print(x.shape)

    # ---------------------------fcn32s模型运行时间-----------------------
    start = time.time()
    pred = model_fcn32s(x)
    print('pred.shape:', pred.shape)
    end = time.time()
    print(end-start)

    # ---------------------------fcn16s模型运行时间-----------------------
    start = time.time()
    pred = model_fcn16s(x)
    end = time.time()
    print(end-start)

    # ---------------------------fcn8s模型运行时间-----------------------
    start = time.time()
    pred = model_fcn8s(x)
    end = time.time()
    print(end-start)

    # print(pred.shape)
    loss = cross_entropy2d(pred, y)
    # print(loss)