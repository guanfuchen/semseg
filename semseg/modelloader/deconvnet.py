# -*- coding: utf-8 -*-
import torch
from torchvision import models


class deconv_vgg16(torch.nn.Module):
    def __init__(self, n_classes=21, pretrained=True):
        super(deconv_vgg16, self).__init__()
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.init_weights(pretrained=self.pretrained)

    def init_weights(self, pretrained=False):
        pass


    def forward(self, x):
        pass
