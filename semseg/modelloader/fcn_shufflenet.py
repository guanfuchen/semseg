# -*- coding: utf-8 -*-
# !!!code is from [ShuffleNet-1g8-Pytorch](https://github.com/ericsun99/ShuffleNet-1g8-Pytorch)!!!
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from semseg.loss import cross_entropy2d


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    """3x3 convolution with padding
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    1x1卷积，groups==1那么正常卷积，否则grouped卷积
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups # 每一group的通道数

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width) # reshape为不同的groups

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3, grouped_conv=True, combine='add'):
        """
        :param in_channels: ShuffleUnit输入通道数
        :param out_channels: ShuffleUnit输出通道数
        :param groups: ShuffleUnit分组groups
        :param grouped_conv: 是否在第一个1x1卷积使用groued卷积
        :param combine: combine使用element wise add or concat
        """

        super(ShuffleUnit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4 # 输出通道数的1/4为ShuffleUnit的bottleneck通道数

        # define the type of ShuffleUnit
        # 不同stirde的ShuffleUnit使用不同的combine with path connection，stride=2的使用concat，stride=1的使用element wise add
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat
            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels # 确保concat输出和原始输出有相同的通道数，因此将out_channels-in_channels作为残差快的输出即可
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1 # 在stage2的第一个conv1x1不使用group卷积

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(self.in_channels, self.bottleneck_channels, self.first_1x1_groups, batch_norm=True, relu=True)

        # 3x3 depthwise convolution followed by batch normalization
        # 3x3 deptheise convolution with BN
        self.depthwise_conv3x3 = conv3x3(self.bottleneck_channels, self.bottleneck_channels, stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(self.bottleneck_channels, self.out_channels, self.groups, batch_norm=True, relu=False)

    @staticmethod
    def _add(x, out):
        # residual connection
        # 残差add连接，用于stride=1的ShuffleUnit
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        # concat连接，用于stride=2的ShuffleUnit
        return torch.cat((x, out), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, batch_norm=True, relu=False):

        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        # 是否在1x1卷积中增加BN
        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        # 是否在1x1卷积中增加ReLU
        if relu:
            modules['relu'] = nn.ReLU()

        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    def forward(self, x):
        # save for combining later with output
        residual = x

        # 如果是concat path connection那么使用平均池化操作，否则直接是x作为残差
        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)

        # 1x1 compress to out_channels//4的卷积
        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups) # channel shuffle操作，将out按照groups shuffle

        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)

        # 1x1 expand to out_channels的卷积
        out = self.g_conv_1x1_expand(out)

        # 最后和残差组合
        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNet(nn.Module):
    """ShuffleNet implementation.
    """

    def __init__(self, groups=3, in_channels=3, n_classes=1000, pretrained=False):
        """ShuffleNet constructor.
        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
                在ShuffeleUnit中使用的1x1卷积groups数量
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            n_classes (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.
        """
        super(ShuffleNet, self).__init__()

        self.groups = groups
        self.stage_repeats = [3, 7, 3] # stage 重复次数
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.pretrained = pretrained

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        # 总攻有3个stage，包括第一个Conv1和MaxPool
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        # Stage 1 always has 24 output channels
        # conv1+maxpool组成了stage1,2,3的第一组操作
        self.conv1 = conv3x3(self.in_channels, self.stage_out_channels[1], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        # stage2构建
        self.stage2 = self._make_stage(2)
        # Stage 3
        # stage3构建
        self.stage3 = self._make_stage(3)
        # Stage 4
        # stage4构建
        self.stage4 = self._make_stage(4)

        # Global pooling:
        # Undefined as PyTorch's functional API can be used for on-the-fly
        # shape inference if input size is not ImageNet's 224x224

        # Fully-connected classification layer
        num_inputs = self.stage_out_channels[-1]
        self.fc = nn.Linear(num_inputs, self.n_classes)

        if self.pretrained:
            self.init_weights()

    def init_weights(self):
        model_checkpoint_path = os.path.expanduser('~/.torch/models/ShuffleNet_1g8_Top1_67.408_Top5_87.258.pth.tar')
        if os.path.exists(model_checkpoint_path):
            pretrained_dict = torch.load(model_checkpoint_path, map_location='cpu')
            model_dict = self.state_dict()
            # new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            new_dict = {}
            for k, v in pretrained_dict.items():
                new_k = k[k.find('.') + 1:]
                new_v = v
                new_dict[new_k] = new_v

            # print(model_dict.keys()[:5])
            # print(pretrained_dict.keys()[:5])
            # print(new_dict.keys()[:5])
            model_dict.update(new_dict)
            self.load_state_dict(model_dict)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = "ShuffleUnit_Stage{}".format(stage) # 增加module name for convience

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2 # 第二个stage不使用grouped conv

        # 2. concatenation unit is always used.
        # 总是使用convcat单元在每一个stage的第一个
        first_module = ShuffleUnit(
            self.stage_out_channels[stage - 1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat'
        )
        modules[stage_name + "_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage - 2]):
            name = stage_name + "_{}".format(i + 1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add'
            )
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # global average pooling layer
        x = F.avg_pool2d(x, x.data.size()[-2:])

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # return F.log_softmax(x, dim=1)
        return x

class fcn_shufflenet(nn.Module):
    def __init__(self, module_type='32s', n_classes=21, pretrained=False, upsample_method='upsample_bilinear'):
        super(fcn_shufflenet, self).__init__()
        self.n_classes = n_classes
        self.module_type = module_type
        self.pretrained = pretrained
        self.upsample_method = upsample_method

        self.shufflenet = ShuffleNet(groups=8, in_channels=3, n_classes=1000, pretrained=self.pretrained)

        self.classifier = nn.Conv2d(self.shufflenet.stage_out_channels[4], self.n_classes, 1)

        if self.upsample_method == 'upsample_bilinear':
            pass
        elif self.upsample_method == 'ConvTranspose2d':
            self.upsample_1 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 3, stride=2, padding=1)
            self.upsample_2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 3, stride=2)
            # self.upsample_3 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 3, stride=2, padding=1)

        if self.module_type=='16s' or self.module_type=='8s':
            self.score_16s_conv = nn.Conv2d(self.shufflenet.stage_out_channels[3], self.n_classes, 1)
        if self.module_type=='8s':
            self.score_8s_conv = nn.Conv2d(self.shufflenet.stage_out_channels[2], self.n_classes, 1)


    def forward(self, x):
        x_size = x.size()[2:]
        features = []
        for name, module in self.shufflenet._modules.items()[:-1]:
            # print('name:', name)
            if name in ['stage3', 'stage4']:
                # pass
                # print('x.shape:', x.shape)
                features.append(x)
            # print('module:', module)
            x = module(x)
        score = self.classifier(x)
        # features.append(x)

        if self.module_type=='16s' or self.module_type=='8s':
            score_16s_out = self.score_16s_conv(features[1])
        if self.module_type=='8s':
            score_8s_out = self.score_8s_conv(features[0])

        if self.module_type=='16s' or self.module_type=='8s':
            if self.upsample_method == 'upsample_bilinear':
                score = F.upsample_bilinear(score, score_16s_out.size()[2:])
            elif self.upsample_method == 'ConvTranspose2d':
                score = self.upsample_1(score)
            score += score_16s_out
        if self.module_type=='8s':
            if self.upsample_method == 'upsample_bilinear':
                score = F.upsample_bilinear(score, score_8s_out.size()[2:])
            elif self.upsample_method == 'ConvTranspose2d':
                score = self.upsample_2(score)
            score += score_8s_out

        out = F.upsample_bilinear(score, x_size)

        return out

def fcn_shufflenet_32s(n_classes=21, pretrained=False):
    model = fcn_shufflenet(module_type='32s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_shufflenet_16s(n_classes=21, pretrained=False):
    model = fcn_shufflenet(module_type='16s', n_classes=n_classes, pretrained=pretrained)
    return model

def fcn_shufflenet_8s(n_classes=21, pretrained=False):
    model = fcn_shufflenet(module_type='8s', n_classes=n_classes, pretrained=pretrained)
    return model

if __name__ == '__main__':
    n_classes = 21
    model_fcn32s = fcn_shufflenet(module_type='32s', n_classes=n_classes, pretrained=True)
    model_fcn16s = fcn_shufflenet(module_type='16s', n_classes=n_classes, pretrained=True)
    model_fcn8s = fcn_shufflenet(module_type='8s', n_classes=n_classes, pretrained=True)
    # model.init_vgg16()
    x = Variable(torch.randn(1, 3, 360, 480))
    y = Variable(torch.LongTensor(np.ones((1, 360, 480), dtype=np.int)))
    # print(x.shape)

    # # ---------------------------fcn32s模型运行时间-----------------------
    start = time.time()
    pred = model_fcn32s(x)
    # print('pred.shape:', pred.shape)
    end = time.time()
    print(end-start)

    # ---------------------------fcn16s模型运行时间-----------------------
    start = time.time()
    pred = model_fcn16s(x)
    end = time.time()
    # print('pred.shape:', pred.shape)
    print(end-start)

    # ---------------------------fcn8s模型运行时间-----------------------
    start = time.time()
    pred = model_fcn8s(x)
    # print('pred.shape:', pred.shape)
    end = time.time()
    print(end-start)

    # print(pred.shape)
    loss = cross_entropy2d(pred, y)
    # print(loss)


