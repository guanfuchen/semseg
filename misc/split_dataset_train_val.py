# -*- coding: utf-8 -*-
import os
import torch
import random
from torch.utils import data

from semseg.dataloader.freespace_loader import freespaceLoader

if __name__ == '__main__':
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/FreeSpaceDataset')
    batch_size = 1
    dst = freespaceLoader(local_path, is_transform=True, is_augment=False)
    dst_len = len(dst)
    dst_ids = range(dst_len)
    random.shuffle(dst_ids)
    train_rate = 0.7
    train_num = int(dst_len*train_rate)
    train_ids = dst_ids[:train_num]
    test_ids = dst_ids[train_num:]
    # for train_id in train_ids:
    #     print(dst.get_filename(train_id))
    for test_id in test_ids:
        test_filename_src = dst.get_filename(test_id)
        test_annot_filename_src = test_filename_src.replace('train', 'trainannot').replace('.png', '_mask.png')

        test_filename_dst = test_filename_src.replace('train', 'test')
        test_annot_filename_dst = test_annot_filename_src.replace('train', 'test')

        test_dst_dir = os.path.dirname(test_filename_dst)
        if not os.path.exists(test_dst_dir):
            os.makedirs(test_dst_dir)

        test_annot_dst_dir = os.path.dirname(test_annot_filename_dst)
        if not os.path.exists(test_annot_dst_dir):
            os.makedirs(test_annot_dst_dir)

        os.rename(test_filename_src, test_filename_dst)
        os.rename(test_annot_filename_src, test_annot_filename_dst)
        # print('test_filename_src:', test_filename_src)
        # print('test_filename_dst:', test_filename_dst)
        # print('test_annot_filename_src:', test_annot_filename_src)
        # print('test_annot_filename_dst:', test_annot_filename_dst)
        # break
