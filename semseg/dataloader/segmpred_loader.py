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


class segmpredLoader(data.Dataset):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, root, split="train", is_transform=False, is_augment=False):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.is_augment = is_augment

        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 19
        self.files = collections.defaultdict(list)

        file_list = glob.glob(root + '/' + split + '/*.t7')
        file_list.sort()
        # print('file_list:', file_list)
        self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        pass
        img_name = self.files[self.split][index]
        img_file_name = img_name[img_name.rfind('/')+1:img_name.rfind('.')]
        img_path = self.root + '/' + self.split + '/' + img_file_name + '.t7'
        # print('img_path:', img_path)
        img_pred = torchfile.load(img_path)
        import random
        random_batch_id = random.randint(0, 3)
        # print('random_batch_id:', random_batch_id)
        # random_batch_id = 0
        img_sequences = img_pred['R8s'][random_batch_id, ...]
        # print('img_sequences.shape:', img_sequences.shape)
        img_past = img_sequences[0:4, ...]
        img_future_onehot = img_sequences[4, ...]
        img_past = np.array(img_past, dtype=np.uint8)
        img_future_onehot = np.array(img_future_onehot, dtype=np.int32)

        img_past = torch.from_numpy(img_past).float()
        img_future_onehot = torch.from_numpy(img_future_onehot).long()

        img_past = img_past.view(-1, 64, 64)

        img_future = np.argmax(img_future_onehot, axis=0)
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
    local_path = os.path.join(HOME_PATH, 'Data/SegmPred')
    batch_size = 4
    dst = segmpredLoader(local_path, is_transform=True, is_augment=False)
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
            # print('dst.decode_segmap(pred_segment):', dst.decode_segmap(pred_segment))
        plt.show()
        if i==0:
            break
