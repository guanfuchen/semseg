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


class freespacepredLoader(data.Dataset):
    """
    freespace dataloader for my onw freespace dataset loader which contain free or not free two segment
    FreeSpaceDataset:
        - train
            - dji_caoping
                - *.png
                - ...
            - dji_huatan
            - dji_road_1
            - dji_road_2
        - trainannot
            - dji_caoping
                - *_mask.png
            - dji_huatan
            - dji_road_1
            - dji_road_2
    """
    def __init__(self, root, split="train", is_transform=False, is_augment=False):
        self.root = root
        self.split = split
        self.img_size = (360, 480) # (h, w)
        self.is_transform = is_transform
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 2
        self.files = collections.defaultdict(list)
        self.joint_augment_transform = None
        self.is_augment = is_augment
        if self.is_augment:
            self.joint_augment_transform = Compose([
                # RandomSized(int(min(self.img_size)/0.875)),
                # RandomCrop(self.img_size),
                RandomRotate(degree=10),
                RandomHorizontallyFlip(),
            ])

        # file_list = os.listdir(root + '/' + split)
        file_list = glob.glob(root + '/' + split + '/*/*.png')
        # print('file_list:', file_list)
        file_list.sort()
        self.files[split] = file_list

        self.input_shape = (64, 64)
        # if self.split == 'train':
        #     self.input_shape = (64, 64)
        # elif self.split == 'test':
        #     self.input_shape = (64, 64)

    def get_filename(self, index):
        return self.files[self.split][index]

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_past = []
        img_future = []

        index = np.clip(index, 0, self.__len__()-4) # for out of range
        for i in range(4):
            img_name = self.files[self.split][index+i]
            img_name_index = img_name.rfind('/')
            img_type_index = img_name.rfind('/', 0, img_name_index - 1)
            # print('img_name_index:', img_name_index)
            # print('img_type_index:', img_name_index)

            img_file_name = img_name[img_name_index + 1:img_name.rfind('.')]
            img_type_name = img_name[img_type_index + 1:img_name_index]
            # print('img_file_name:', img_file_name)
            # print('img_type_name:', img_type_name)
            # img_file_name = img_name[:img_name.rfind('.')]
            # print(img_file_name)
            img_path = self.root + '/' + self.split + '/' + img_type_name + '/' + img_file_name + '.png'
            lbl_path = self.root + '/' + self.split + 'annot/' + img_type_name + '/' + img_file_name + '_mask.png'

            img = Image.open(img_path)
            lbl = Image.open(lbl_path)

            img = img.resize((self.input_shape[1], self.input_shape[0]))
            lbl = lbl.resize((self.input_shape[1], self.input_shape[0]))

            if self.is_augment:
                if self.joint_augment_transform is not None:
                    img, lbl = self.joint_augment_transform(img, lbl)

            img = np.array(img, dtype=np.uint8)
            lbl = np.array(lbl, dtype=np.int32)

            if self.is_transform:
                img, lbl = self.transform(img, lbl)

            if i <= 3:
                img_past.append(lbl.float())
            img_future = lbl
        img_past = torch.stack(img_past)

        # print('img_past.shape:', img_past.shape)
        # print('img_future.shape:', img_future.shape)
        # return img, lbl
        return img_past, img_future

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
        FreeSpace = [255, 0, 0]
        Unlabelled = [0, 0, 0]

        label_colours = np.array(
            [
                Unlabelled,
                FreeSpace,
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
    local_path = os.path.join(HOME_PATH, 'Data/FreeSpacePredDataset')
    batch_size = 1
    dst = freespacepredLoader(local_path, is_transform=True, is_augment=False)
    trainloader = data.DataLoader(dst, batch_size=batch_size, shuffle=False)
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
