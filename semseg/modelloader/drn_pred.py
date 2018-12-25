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


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        if next(self.parameters()).is_cuda:
            return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(), Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)), Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1)):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation[1], bias=False, dilation=dilation[1])
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

class DRNPred_A(nn.Module):

    def __init__(self, block, layers, input_channel=3):
        self.inplanes = 64
        super(DRNPred_A, self).__init__()
        self.block = block
        self.layers = layers
        self.input_channel = input_channel

        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        self.layer5 = self._make_pred_layer(ASPP_Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], input_channel, in_channels=512*block.expansion)
        if self.layer5 is not None:
            self.out_dim = input_channel
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
            layers.append(block(self.inplanes, planes, dilation=(dilation, dilation)))

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


def drnpred_a_18(pretrained=False, **kwargs):
    model = DRNPred_A(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def drnpred_a_34(pretrained=False, **kwargs):
    model = DRNPred_A(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def drnpred_a_101(pretrained=False, **kwargs):
    model = DRNPred_A(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

# -------------------------semantic model----------------------------------
def drnsegpred_a_18(pretrained=False, n_classes=21, input_shape = (64, 64), input_channel=19):
    model = DRNSegPred(model_name='drnpred_a_18', pretrained=pretrained, input_channel=input_channel, input_shape=input_shape, n_classes=n_classes)
    return model

def drnsegpred_a_34(pretrained=False, n_classes=21, input_shape = (64, 64), input_channel=19):
    model = DRNSegPred(model_name='drnpred_a_34', pretrained=pretrained, input_channel=input_channel, input_shape=input_shape, n_classes=n_classes)
    return model

def drnsegpred_a_101(pretrained=False, n_classes=21, input_shape = (64, 64), input_channel=19):
    model = DRNSegPred(model_name='drnpred_a_101', pretrained=pretrained, input_channel=input_channel, input_shape=input_shape, n_classes=n_classes)
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
class DRNSegPred(nn.Module):
    def __init__(self, model_name, pretrained=False, use_torch_up=True, input_channel=19, input_shape=(64, 64), n_classes=21):
        super(DRNSegPred, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.input_channel = input_channel

        model = eval(model_name)(pretrained=pretrained, input_channel=input_channel*4)
        self.base = model
        self.seg = nn.Conv2d(model.out_dim, input_channel*4, kernel_size=1)
        m = self.seg

        # 初始化分割图最后的卷积weights和bias
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()

        # ----lstm seq----
        # self.lstm1 = nn.LSTM(8*8, 8*8, 2)
        # ----lstm seq----

        # ----conv lstm seq----
        self.lstm1 = ConvLSTM(input_size=(self.input_shape[0]//8, self.input_shape[1]//8), input_dim=self.input_channel, hidden_dim=[128, 128, self.input_channel], kernel_size=(3, 3), num_layers=3, batch_first=True, bias=True, return_all_layers=False)
        # ----conv lstm seq----

        self.out_conv = nn.Conv2d(self.input_channel*4, self.n_classes, kernel_size=1)

        if use_torch_up:
            # 使用pytorch双线性上采样
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            # 使用转置卷积上采样
            up = nn.ConvTranspose2d(input_channel*4, input_channel*4, 16, stride=8, padding=4, output_padding=0, groups=input_channel*4, bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        # 将分割图对应到分割类别数上
        x = self.seg(x)
        # print('x.shape:', x.shape)

        # ----lstm seq----
        # x = x.view(-1, 76, 8*8)
        # x, _ = self.lstm1(x)
        # x = x.view(-1, 76, 8, 8)
        # ----lstm seq----

        # ----conv lstm seq----
        # print('x.shape:', x.shape)
        # print('self.input_channel:', self.input_channel)
        x = x.view(-1, 4, self.input_channel, self.input_shape[0]//8, self.input_shape[1]//8)
        # print('x.shape:', x.shape)
        x, _ = self.lstm1(x)
        # print('x.shape:', x.shape)
        x = x.view(-1, 4*self.input_channel, self.input_shape[0]//8, self.input_shape[1]//8)
        # ----conv lstm seq----

        x = self.out_conv(x)
        # 使用双线性上采样或者转置卷积上采样8倍降采样率的分割图
        y = self.up(x)
        return y



if __name__ == '__main__':
    n_classes = 19
    # model = DRNSegPred(model_name='drnpred_a_18', pretrained=False, input_channel=n_classes*4)
    model = drnsegpred_a_18(pretrained=False, n_classes=n_classes)
    x = Variable(torch.randn(1, n_classes*4, 64, 64))
    y = Variable(torch.LongTensor(np.ones((1, 64, 64), dtype=np.int)))
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    print(pred.shape)
    # loss = cross_entropy2d(pred, y)
    # print(loss)
