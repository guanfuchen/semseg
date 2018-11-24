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
import os

from semseg.loss import cross_entropy2d


class mobilenet_conv_bn_relu(nn.Module):
    """
    :param
    """
    def __init__(self, in_channels, out_channels, stride):
        super(mobilenet_conv_bn_relu, self).__init__()
        self.cbr_seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.cbr_seq(x)
        return x


class mobilenet_conv_dw_relu(nn.Module):
    """
    :param
    """
    def __init__(self, in_channels, out_channels, stride):
        super(mobilenet_conv_dw_relu, self).__init__()
        self.cbr_seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.cbr_seq(x)
        return x


def fcn_MobileNet_32s(n_classes=21, pretrained=False):
    model = fcn_MobileNet(module_type='32s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_MobileNet_16s(n_classes=21, pretrained=False):
    model = fcn_MobileNet(module_type='16s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_MobileNet_8s(n_classes=21, pretrained=False):
    model = fcn_MobileNet(module_type='8s', n_classes=n_classes, pretrained=pretrained)
    return model

class fcn_MobileNet(nn.Module):
    """
    :param
    """
    def __init__(self, module_type='32s', n_classes=21, pretrained=True):
        super(fcn_MobileNet, self).__init__()

        self.n_classes = n_classes
        self.module_type = module_type

        self.conv1_bn = mobilenet_conv_bn_relu(3, 32, 2)
        self.conv2_dw = mobilenet_conv_dw_relu(32, 64, 1)
        self.conv3_dw = mobilenet_conv_dw_relu(64, 128, 2)
        self.conv4_dw = mobilenet_conv_dw_relu(128, 128, 1)
        self.conv5_dw = mobilenet_conv_dw_relu(128, 256, 2)
        self.conv6_dw = mobilenet_conv_dw_relu(256, 256, 1)
        self.conv7_dw = mobilenet_conv_dw_relu(256, 512, 2)
        self.conv8_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv9_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv10_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv11_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv12_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv13_dw = mobilenet_conv_dw_relu(512, 1024, 2)
        self.conv14_dw = mobilenet_conv_dw_relu(1024, 1024, 1)
        # self.avg_pool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(1024, n_classes)

        self.classifier = nn.Conv2d(1024, self.n_classes, 1)

        if self.module_type=='16s' or self.module_type=='8s':
            self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        if self.module_type=='8s':
            self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        if pretrained:
            self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=False):
        model_checkpoint_path = os.path.expanduser('~/.torch/models/mobilenet_sgd_rmsprop_69.526.tar')
        if os.path.exists(model_checkpoint_path):
            model_checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
            pretrained_dict = model_checkpoint['state_dict']

            model_dict = self.state_dict()

            # print(model_dict.keys())
            # print(pretrained_dict.keys())
            model_dict_keys = model_dict.keys()

            new_dict = {}
            for dict_index, (k, v) in enumerate(pretrained_dict.items()):
                if k=='module.fc.weight':
                    break
                # print(dict_index)
                # print(k)
                new_k = model_dict_keys[dict_index]
                new_v = v
                new_dict[new_k] = new_v
            model_dict.update(new_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x_size = x.size()[2:]
        x_conv1 = self.conv1_bn(x)
        x_conv2 = self.conv2_dw(x_conv1)
        x_conv3 = self.conv3_dw(x_conv2)
        x_conv4 = self.conv4_dw(x_conv3)
        x_conv5 = self.conv5_dw(x_conv4)
        x_conv6 = self.conv6_dw(x_conv5)
        x_conv7 = self.conv7_dw(x_conv6)
        x_conv8 = self.conv8_dw(x_conv7)
        x_conv9 = self.conv9_dw(x_conv8)
        x_conv10 = self.conv10_dw(x_conv9)
        x_conv11 = self.conv11_dw(x_conv10)
        x_conv12 = self.conv12_dw(x_conv11)
        x_conv13 = self.conv13_dw(x_conv12)
        x = self.conv14_dw(x_conv13)

        # x = self.avg_pool(x)
        # x = x.view(-1, 1024)
        # x = self.fc(x)

        score = self.classifier(x)

        if self.module_type=='16s' or self.module_type=='8s':
            score_pool4 = self.score_pool4(x_conv12)
        if self.module_type=='8s':
            score_pool3 = self.score_pool3(x_conv6)

        if self.module_type=='16s' or self.module_type=='8s':
            score = F.upsample_bilinear(score, score_pool4.size()[2:])
            score += score_pool4
        if self.module_type=='8s':
            score = F.upsample_bilinear(score, score_pool3.size()[2:])
            score += score_pool3

        out = F.upsample_bilinear(score, x_size)

        return out


if __name__ == '__main__':
    batch_size = 1
    n_classes = 21
    model_fcn32s = fcn_MobileNet(module_type='32s', n_classes=n_classes, pretrained=True)
    model_fcn16s = fcn_MobileNet(module_type='16s', n_classes=n_classes, pretrained=True)
    model_fcn8s = fcn_MobileNet(module_type='8s', n_classes=n_classes, pretrained=True)
    x = Variable(torch.randn(batch_size, 3, 360, 480))
    y = Variable(torch.LongTensor(np.ones((batch_size, 360, 480), dtype=np.int)))

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
