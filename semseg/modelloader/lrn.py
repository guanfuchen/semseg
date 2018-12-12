# -*- coding: utf-8 -*-
import torch
from torch import nn
from torchvision.models import vgg
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

from semseg.loss import cross_entropy2d

def conv3x3_bn_relu(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class LRNRefineUnit(nn.Module):
    def __init__(self, R_channel, M_channel):
        super(LRNRefineUnit, self).__init__()
        self.R_channel = R_channel
        self.M_channel = M_channel
        self.conv_M2m = conv3x3_bn_relu(in_planes=self.M_channel, out_planes=self.M_channel, stride=2)
        self.conv_R2r = nn.Conv2d(in_channels=self.R_channel+self.M_channel, out_channels=self.R_channel, kernel_size=3, padding=1)

    def forward(self, Rf, Mf):
        Mf_size = Mf.size()
        # print('Rf.shape:', Rf.shape)
        # print('Mf.shape:', Mf.shape)
        mf = self.conv_M2m(Mf)
        # print('mf.shape:', mf.shape)
        # print('Rf.shape:', Rf.shape)
        rf = torch.cat((mf[:, :, :Rf.shape[2], :Rf.shape[3]], Rf), 1)
        rf = self.conv_R2r(rf)
        out = F.upsample_bilinear(rf, Mf_size[2:])
        return out


class lrn_vgg16(nn.Module):
    def __init__(self, n_classes=21, pretrained=False):
        super(lrn_vgg16, self).__init__()
        self.n_classes = n_classes
        vgg16 = vgg.vgg16(pretrained=pretrained)
        self.encoder = vgg16.features
        self.out_conv = nn.Conv2d(in_channels=512, out_channels=self.n_classes, kernel_size=1)

        # ----decoder refine units----
        self.refine_units = []

        self.refine_1 = LRNRefineUnit(self.n_classes, 512)
        self.refine_units.append(self.refine_1)

        self.refine_2 = LRNRefineUnit(self.n_classes, 512)
        self.refine_units.append(self.refine_2)

        self.refine_3 = LRNRefineUnit(self.n_classes, 256)
        self.refine_units.append(self.refine_3)

        self.refine_4 = LRNRefineUnit(self.n_classes, 128)
        self.refine_units.append(self.refine_4)

        self.refine_5 = LRNRefineUnit(self.n_classes, 64)
        self.refine_units.append(self.refine_5)
        # ----decoder refine units----

    def forward(self, x):
        encoder_features = []
        for name, module in self.encoder._modules.items():
            # print('name:', name)
            # print('module:', module)
            x = module(x)
            if name in ['3', '8', '15', '22', '29']:
                # print('x.shape:', x.shape)
                encoder_features.append(x)
        x = self.out_conv(x)
        # print('x.shape:', x.shape)
        out_s = []
        out_s.append(x)
        encoder_features.reverse()
        for refine_id, encoder_feature in enumerate(encoder_features):
            x = self.refine_units[refine_id](x, encoder_feature)
            out_s.append(x)
            # print('x.shape:', x.shape)
            # break
        # return out_s
        return out_s[-1]

if __name__ == '__main__':
    batch_size = 1
    n_classes = 21
    img_height, img_width = 360, 480
    # img_height, img_width = 1024, 512
    model = lrn_vgg16(n_classes=n_classes, pretrained=True)
    x = Variable(torch.randn(batch_size, 3, img_height, img_width))
    y = Variable(torch.LongTensor(np.ones((batch_size, img_height, img_width), dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(end-start)
    # print(pred.shape)
    # print('pred.type:', pred.type)
    # loss = cross_entropy2d(pred, y)
    # print(loss)
