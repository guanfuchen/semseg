#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import collections
import torch
import scipy.misc as m
import matplotlib.pyplot as plt
from torch.utils import data
import glob
import numpy as np

class ade20kLoader(data.Dataset):
    def __init__(self, root, split="training", is_transform=False, img_size=512):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 13
        self.image_files = collections.defaultdict(list)
        self.label_files = collections.defaultdict(list)
        self.image_files[self.split] = glob.glob(os.path.join(root, 'images', self.split, '*', '*.jpg'))
        self.label_files[self.split] = glob.glob(os.path.join(root, 'images', self.split, '*', '*_seg.png'))
        # 文件名需相同
        self.image_files[self.split].sort()
        self.label_files[self.split].sort()
        # print(len(self.image_files[self.split]))
        # print(len(self.label_files[self.split]))
        assert len(self.image_files[self.split]) == len(self.label_files[self.split])

    def __len__(self):
        return len(self.image_files[self.split])

    def __getitem__(self, index):
        img_path = self.image_files[self.split][index]
        lbl_path = self.label_files[self.split][index]
        # print('img_path:', img_path)
        # print('lbl_path:', lbl_path)

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    # 转换HWC为CHW
    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(float) / 255.0
        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        lbl = self.encode_segmap(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest', mode='F')
        lbl = lbl.astype(int)
        # print(img)
        # print(lbl)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def encode_segmap(self, mask):
        # Refer : http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask = (mask[:, :, 0] / 10.0) * 256 + mask[:, :, 1]
        return np.array(label_mask, dtype=np.uint8)

    def decode_segmap(self, temp, plot=False):
        # TODO:(@meetshah1995)
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r / 255.0)
        rgb[:, :, 1] = (g / 255.0)
        rgb[:, :, 2] = (b / 255.0)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/ADE20K_2016_07_26')
    dst = ade20kLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, (imgs, labels) in enumerate(trainloader):
        # print(i)
        # print(imgs.shape)
        # print(labels.shape)
        if i == 0:
            imgs = imgs.numpy()
            img = imgs[0, :, :, :]
            # print(img.shape)
            img = np.transpose(img, (1, 2, 0))
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(dst.decode_segmap(labels.numpy()[0]))
            plt.show()
        break