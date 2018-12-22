# -*- coding: utf-8 -*-
import torch
import os
import collections
import random

import cv2
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torchvision import transforms
import glob
import torchfile

class movingmnistLoader(data.Dataset):

    def __init__(self, root, split="train", is_transform=False, is_augment=False, split_ratio=0.7):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.is_augment = is_augment
        self.split_ratio = split_ratio

        self.n_classes = 2
        colors = [
            [0, 0, 0],
            [255, 255, 255],
        ]
        self.label_colours = dict(zip(range(self.n_classes), colors))
        self.data = np.load(self.root).transpose(1, 0, 2, 3) # from TxSxHxW to SxTxHxW
        all_len = len(self.data[:, 0, 0, 0])
        split_index = int(all_len*split_ratio)
        if self.split=='train':
            self.data = self.data[:split_index]
        elif self.split=='val':
            self.data = self.data[split_index:]
        print('self.data.shape:', self.data.shape)

    def __len__(self):
        return len(self.data[:, 0, 0, 0])

    def __getitem__(self, index):
        img_np = self.data[index, ...]
        # img_np = (img_np-128)//128
        # print(img_np[0, 0, 0])
        img_np = np.clip(img_np, 0, 1)
        # print(img_np[0, 0, 0])
        # print(np.max(img_np))
        # print(np.min(img_np))
        # print(np.unique(img_np))
        # print('img_np.shape:', img_np.shape)
        img_past = img_np[:9]
        img_future = img_np[9]

        img_past = np.array(img_past, dtype=np.uint8)
        img_future = np.array(img_future, dtype=np.int32)

        img_past = torch.from_numpy(img_past).float()
        img_future = torch.from_numpy(img_future).long()

        # print('img_past.shape:', img_past.shape)
        # print('img_future.shape:', img_future.shape)

        return img_past, img_future

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
        # rgb[:, :, 0] = r / 255.0
        # rgb[:, :, 1] = g / 255.0
        # rgb[:, :, 2] = b / 255.0
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        return rgb

if __name__ == '__main__':
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/mnist_test_seq.npy')
    batch_size = 4
    dst = movingmnistLoader(local_path, is_transform=True, is_augment=False)
    trainloader = data.DataLoader(dst, batch_size=batch_size, shuffle=True)
    for i, (img_past, img_future) in enumerate(trainloader):
        # print(i)
        # print(img_past.shape)
        # print(img_future.shape)
        # if i == 0:
        image_list_len = img_past.shape[0]
        for image_list in range(image_list_len):
            pred_segment = img_future.numpy()[image_list]
            # print('img_future_onehot:', img_future_onehot)
            # print('img_future_onehot.shape:', img_future_onehot.shape)
            plt.imshow(dst.decode_segmap(pred_segment))
            # plt.imshow(pred_segment)
            # print('dst.decode_segmap(pred_segment):', dst.decode_segmap(pred_segment))
        plt.show()
        if i==0:
            break
