# -*- coding: utf-8 -*-
# 代码!!部分参考!![gcn.py](https://github.com/ycszen/pytorch-segmentation/blob/master/gcn.py)
import torch
from torch import nn
from torchvision.models import vgg, resnet50, resnet18, resnet34, resnet101
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

from semseg.loss import cross_entropy2d


class gcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(self.kernel_size, 1), padding=(self.kernel_size//2, 0))
        self.conv1_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(1, self.kernel_size), padding=(0, self.kernel_size//2))

        self.conv2_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1, self.kernel_size), padding=(0, self.kernel_size//2))
        self.conv2_2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=(self.kernel_size, 1), padding=(self.kernel_size//2, 0))

    def forward(self, x):
        x_conv1_1 = self.conv1_1(x)
        x_conv1_2 = self.conv1_2(x_conv1_1)

        x_conv2_1 = self.conv2_1(x)
        x_conv2_2 = self.conv2_2(x_conv2_1)

        x_out = x_conv1_2 + x_conv2_2
        return x_out


class boundary_refine(nn.Module):
    def __init__(self, in_channels):
        super(boundary_refine, self).__init__()
        self.in_channels = in_channels

        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x_out = residual + x
        return x_out

def gcn_resnet18(n_classes=21, pretrained=False):
    return gcn_resnet(n_classes=n_classes, pretrained=pretrained, expansion=1, model='resnet18')

def gcn_resnet34(n_classes=21, pretrained=False):
    return gcn_resnet(n_classes=n_classes, pretrained=pretrained, expansion=1, model='resnet34')

def gcn_resnet50(n_classes=21, pretrained=False):
    return gcn_resnet(n_classes=n_classes, pretrained=pretrained, expansion=4, model='resnet50')

def gcn_resnet101(n_classes=21, pretrained=False):
    return gcn_resnet(n_classes=n_classes, pretrained=pretrained, expansion=4, model='resnet101')

class gcn_resnet(nn.Module):
    def __init__(self, n_classes=21, pretrained=False, expansion=4, model='resnet18'):
        super(gcn_resnet, self).__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.expansion = expansion
        backbone = eval(model)(self.pretrained)

        self.conv1= backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # ----global convolution network----
        self.gcn1 = gcn(in_channels=64*self.expansion, out_channels=self.n_classes)
        self.gcn2 = gcn(in_channels=128*self.expansion, out_channels=self.n_classes)
        self.gcn3 = gcn(in_channels=256*self.expansion, out_channels=self.n_classes)
        self.gcn4 = gcn(in_channels=512*self.expansion, out_channels=self.n_classes)
        # ----global convolution network----


        # ----boundary refinement----
        self.br1_1 = boundary_refine(in_channels=self.n_classes)
        self.br1_2 = boundary_refine(in_channels=self.n_classes)
        self.br1_3 = boundary_refine(in_channels=self.n_classes)
        self.br1_4 = boundary_refine(in_channels=self.n_classes)

        self.br2_1 = boundary_refine(in_channels=self.n_classes)
        self.br2_2 = boundary_refine(in_channels=self.n_classes)
        self.br2_3 = boundary_refine(in_channels=self.n_classes)

        self.br3_1 = boundary_refine(in_channels=self.n_classes)
        self.br3_2 = boundary_refine(in_channels=self.n_classes)
        # ----boundary refinement----


    def forward(self, x):
        # ----normal forward----
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        # ----normal forward----

        x_4 = self.gcn4(x_4)
        x_4 = self.br1_4(x_4)
        x_4_up = F.upsample_bilinear(x_4, x_3.size()[2:])

        x_3 = self.gcn3(x_3)
        x_3 = self.br1_3(x_3)
        x_3_skip = x_3 + x_4_up
        x_3_skip = self.br2_3(x_3_skip)
        x_3_up = F.upsample_bilinear(x_3_skip, x_2.size()[2:])

        x_2 = self.gcn2(x_2)
        x_2 = self.br1_2(x_2)
        x_2_skip = x_2 + x_3_up
        x_2_skip = self.br2_2(x_2_skip)
        x_2_up = F.upsample_bilinear(x_2_skip, x_1.size()[2:])

        x_1 = self.gcn1(x_1)
        x_1 = self.br1_1(x_1)
        x_1_skip = x_1 + x_2_up
        x_1_skip = self.br2_1(x_1_skip)
        x_1_up = F.upsample_bilinear(x_1_skip, scale_factor=2)

        x_out = self.br3_1(x_1_up)
        x_out = F.upsample_bilinear(x_out, scale_factor=2)
        x_out = self.br3_2(x_out)

        return x_out

if __name__ == '__main__':
    batch_size = 1
    n_classes = 21
    img_height, img_width = 360, 480
    # img_height, img_width = 1024, 512
    model = gcn_resnet18(n_classes=n_classes, pretrained=False)
    x = Variable(torch.randn(batch_size, 3, img_height, img_width))
    y = Variable(torch.LongTensor(np.ones((batch_size, img_height, img_width), dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    print(pred.shape)
    # print('pred.type:', pred.type)
    # loss = cross_entropy2d(pred, y)
    # print(loss)
