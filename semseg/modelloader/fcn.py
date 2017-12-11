# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# fcn32s模型
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

if __name__ == '__main__':
    n_classes = 21
    model = fcn32s(n_classes=n_classes)
    x = Variable(torch.randn(1, 3, 360, 480))
    print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    print(pred.shape)
