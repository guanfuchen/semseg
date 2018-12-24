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

from semseg.dataloader.utils import Compose, RandomHorizontallyFlip, RandomRotate, RandomSized, RandomCrop


class camvidLRNLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, is_augment=False, img_sizes=[(11, 15), (22, 30), (45, 60), (90, 120,), (180, 240), (360, 480)]):
        self.root = root
        self.split = split
        self.img_size = (360, 480) # (h, w)
        self.img_sizes = img_sizes
        self.is_transform = is_transform
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 12
        self.files = collections.defaultdict(list)
        self.joint_augment_transform = None
        self.is_augment = is_augment
        if self.is_augment:
            self.joint_augment_transform = Compose([
                # RandomSized(int(480)),
                RandomRotate(degree=10),
                RandomHorizontallyFlip(),
            ])

        # file_list = os.listdir(root + '/' + split)
        file_list = glob.glob(root + '/' + split + '/*.png')
        file_list.sort()
        self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        imgs = []
        lbls = []
        for img_size in self.img_sizes:
            img_name = self.files[self.split][index]
            img_file_name = img_name[img_name.rfind('/') + 1:img_name.rfind('.')]
            # img_file_name = img_name[:img_name.rfind('.')]
            # print(img_file_name)
            img_path = self.root + '/' + self.split + '/' + img_file_name + '.png'
            lbl_path = self.root + '/' + self.split + 'annot/' + img_file_name + '.png'

            img = Image.open(img_path)
            lbl = Image.open(lbl_path)

            img = img.resize((img_size[1], img_size[0]))
            lbl = lbl.resize((img_size[1], img_size[0]))

            if self.is_augment:
                if self.joint_augment_transform is not None:
                    img, lbl = self.joint_augment_transform(img, lbl)

            img = np.array(img, dtype=np.uint8)
            lbl = np.array(lbl, dtype=np.int32)

            if self.is_transform:
                img, lbl = self.transform(img, lbl)
            imgs.append(img)
            lbls.append(lbl)
        return imgs, lbls

    # 转换HWC为CHW
    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, temp, plot=False):
        Sky = [128, 128, 128]
        Building = [128, 0, 0]
        Pole = [192, 192, 128]
        # Road_marking = [255, 69, 0]
        Road = [128, 64, 128]
        Pavement = [60, 40, 222]
        Tree = [128, 128, 0]
        SignSymbol = [192, 128, 128]
        Fence = [64, 64, 128]
        Car = [64, 0, 128]
        Pedestrian = [64, 64, 0]
        Bicyclist = [0, 128, 192]
        Unlabelled = [0, 0, 0]

        label_colours = np.array(
            [
                Sky,
                Building,
                Pole,
                Road,
                Pavement,
                Tree,
                SignSymbol,
                Fence,
                Car,
                Pedestrian,
                Bicyclist,
                Unlabelled,
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.float32)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

if __name__ == '__main__':
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    batch_size = 4
    dst = camvidLRNLoader(local_path, is_transform=True, is_augment=False)
    trainloader = data.DataLoader(dst, batch_size=batch_size, shuffle=True)
    for i, (imgs, labels) in enumerate(trainloader):
        print(i)
        print(len(imgs))
        print(imgs[-1].shape)
        # print(imgs.shape)
        # print(labels.shape)
        # if i == 0:
        image_list_len = imgs[-1].shape[0]
        for image_list in range(image_list_len):
            img = imgs[-1][image_list, :, :, :]
            # print('img.shape:', img.shape)
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.subplot(image_list_len, 2, 2 * image_list + 1)
            plt.imshow(img)
            plt.subplot(image_list_len, 2, 2 * image_list + 2)
            plt.imshow(dst.decode_segmap(labels[-1].numpy()[image_list]))
            # print(dst.decode_segmap(labels.numpy()[image_list])[0, 0, :])
        plt.show()
        if i==0:
            break
