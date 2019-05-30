# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import time
# import os.path as osp
import glob
import os, sys
# import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle
import msgpack
import tqdm
import pyarrow

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

from semseg.dataloader.camvid_loader import camvidLoader


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            # print('txn:', txn)
            self.length = pyarrow.deserialize(txn.get('__len__'))
            # print('length:', self.length)
            # self.keys = msgpack.loads(txn.get(b'__keys__'))
            self.keys = pyarrow.deserialize(txn.get('__keys__'))
            # print('keys:', self.keys)

        self.transform = transform

    def __getitem__(self, index):
        img = None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        img = pyarrow.deserialize(byteflow)

        # load image
        # imgbuf = unpacked[0]
        # print('imgbuf:', imgbuf)
        # print('imgbuf.shape:', imgbuf.shape)
        # cv2.imshow('img:', img)
        # cv2.waitKey(1)
        img = torch.FloatTensor(img)
        # buf = six.BytesIO()
        # buf.write(imgbuf)
        # buf.seek(0)
        # img = Image.open(buf).convert('RGB')
        # print('buf:', buf)

        # if self.transform is not None:
        #     img = self.transform(img)

        return img

    def __len__(self):
        return self.length
        # return 64

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pyarrow.serialize(obj).to_buffer()


def folder2lmdb(dpath, lmdb_path, write_frequency=5000):
    directory = os.path.expanduser(dpath)
    print("Loading dataset from %s" % directory)
    # dataset = ImageFolder(directory, loader=raw_reader)
    # data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)
    data_path_lists = glob.glob(os.path.join(directory, '*.png'))

    # lmdb_path = "{}.lmdb".format(time.time())
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)
    for idx, data_path in enumerate(data_path_lists):
        # print(type(data), data)
        image = cv2.imread(data_path)
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(image))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_path_lists)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        print('keys:', keys)
        print('len(keys):', len(keys))
        txn.put('__keys__', dumps_pyarrow(keys))
        txn.put('__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == "__main__":
    # lmdb_path = "{}.lmdb/".format(time.time())
    lmdb_path = "tmp.lmdb"
    # folder2lmdb("~/Data/CamVid/train/", lmdb_path)

    batch_size = 1

    dst = ImageFolderLMDB(lmdb_path, None)
    loader = DataLoader(dst, batch_size=batch_size, drop_last=True)

    time_start = time.time()
    for idx, data in enumerate(loader):
        pass
        # print("idx:", idx)
    time_end = time.time()
    print('load {} images cost time: {} sec'.format(len(dst), time_end-time_start))
    print('load {} images {} fps'.format(len(dst), len(dst)*1.0/(time_end-time_start)))

    local_path = os.path.join(os.path.expanduser('~/Data/CamVid'))
    dst = camvidLoader(local_path, is_transform=False, is_augment=False)
    loader = DataLoader(dst, batch_size=batch_size)
    time_start = time.time()
    for idx, data in enumerate(loader):
        pass
        # print("idx:", idx)
    time_end = time.time()
    print('load {} images cost time: {} sec'.format(len(dst), time_end-time_start))
    print('load {} images {} fps'.format(len(dst), len(dst)*1.0/(time_end-time_start)))
