# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

# fcn32s模型
from semseg.loss import cross_entropy2d


class fcn32s(nn.Module):
    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        conv3 = self.conv3_block(conv2)
        conv4 = self.conv4_block(conv3)
        conv5 = self.conv5_block(conv4)
        score = self.classifier(conv5)
        out = F.upsample_bilinear(score, x.size()[2:])
        return out

    def __init__(self, n_classes=21):
        super(fcn32s, self).__init__()
        self.n_classes = n_classes

        # VGG16=2+2+3+3+3+3
        # VGG16网络的第一个模块是两个out_channel=64的卷积块
        self.conv1_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # VGG16网络的第二个模块是两个out_channel=128的卷积块
        self.conv2_block = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # VGG16网络的第三个模块是三个out_channel=256的卷积块
        self.conv3_block = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # VGG16网络的第四个模块是三个out_channel=512的卷积块
        self.conv4_block = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # VGG16网络的第五个模块是三个out_channel=512的卷积块
        self.conv5_block = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

    def init_vgg16(self):
        vgg16 = models.vgg16(pretrained=True)

        # -----------赋值前面2+2+3+3+3层feature的特征-------------
        # 由于vgg16的特征是Sequential，获得其中的子类通过children()
        vgg16_features = list(vgg16.features.children())

        conv_blocks = [self.conv1_block, self.conv2_block, self.conv3_block, self.conv4_block, self.conv5_block]
        conv_ids_vgg = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 30]]

        for conv_block_id, conv_block in enumerate(conv_blocks):
            # print(conv_block_id)
            conv_id_vgg = conv_ids_vgg[conv_block_id]
            for l1, l2 in zip(conv_block, vgg16_features[conv_id_vgg[0]:conv_id_vgg[1]]):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    # 赋值的是数据
                    l1.weight.data = l2.weight.data
                    l1.bias.data = l2.bias.data
                    # print(l1)
                    # print(l2)

        # -----------赋值后面3层classifier的特征-------------
        vgg16_classifier = list(vgg16.classifier.children())
        for l1, l2 in zip(self.classifier, vgg16_classifier[0:3]):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Linear):
                l1.weight.data = l2.weight.data.view(l1.weight.size())
                l1.bias.data = l2.bias.data.view(l1.bias.size())

        # -----赋值后面1层classifier的特征，由于类别不同，需要修改------
        l1 = self.classifier[6]
        l2 = vgg16_classifier[6]
        if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Linear):
            l1.weight.data = l2.weight.data[:n_classes, :].view(l1.weight.size())
            l1.bias.data = l2.bias.data[:n_classes].view(l1.bias.size())

if __name__ == '__main__':
    n_classes = 21
    model = fcn32s(n_classes=n_classes)
    # model.init_vgg16()
    x = Variable(torch.randn(1, 3, 360, 480))
    y = Variable(torch.LongTensor(np.ones((1, 360, 480), dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    # print(pred.shape)
    loss = cross_entropy2d(pred, y)
    # print(loss)
