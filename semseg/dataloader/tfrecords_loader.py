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
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
import random
import matplotlib.pyplot as plt


def write_tfrecords(start, end, labels, flower_dir, file):
    writer = tf.python_io.TFRecordWriter(file)
    for id in range(start, end):
        img = Image.open(flower_dir[id])
        # label = labels[id]
        label = 0
        width, height = img.size
        # h = 500
        # x = int((width - h) / 2)
        # y = int((height - h) / 2)
        # img_crop = img.crop([x, y, x + h, y + h])
        # img_500 = img_crop.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            # 'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_500])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            # 'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[500])),
            # 'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[500]))
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))
        }))
        # writer_500.write(example_500.SerializeToString())
        writer.write(example.SerializeToString())


def flower_preprocess(flower_folder='/Users/cgf/Data/tmp/tfrecords'):

    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
              1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    random.shuffle(labels)
    flower_dir = list()

    for img in os.listdir(flower_folder):
        flower_dir.append(os.path.join(flower_folder, img))
    flower_dir.sort()

    length = int(len(flower_dir)*0.7)
    file = "flower_train.tfrecords"
    write_tfrecords(0, length, labels, flower_dir, file)


class TFRecordsProxy():
    def __init__(self, filenames, data_size):
        self.filenames = filenames
        self.data_size = data_size

    def parse_data(self, example_proto):
        features = {'img': tf.FixedLenFeature([], tf.string, ''),
                    'label': tf.FixedLenFeature([], tf.int64, 0),
                    'width': tf.FixedLenFeature([], tf.int64, 0),
                    'height': tf.FixedLenFeature([], tf.int64, 0),
                    }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.decode_raw(parsed_features['img'], tf.uint8)
        label = tf.cast(parsed_features['label'], tf.int64)
        width = tf.cast(parsed_features['width'], tf.int64)
        height = tf.cast(parsed_features['height'], tf.int64)
        image = tf.reshape(image, tf.stack([height, width, 3]))
        # print('width:', width)
        # print('height:', height)
        # image = tf.reshape(image, [height, width, 3])
        # print('image:', image)
        return image, label
        # return width, label

    def my_input_fn(self, filenames, data_size):
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_data)
        # dataset = dataset.batch(data_size)

        # dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def main(self):
        features, labels = self.my_input_fn(self.filenames, self.data_size)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            img, label = sess.run([features, labels])

        return img, label


class TFRecordsDataset(torch.utils.data.Dataset):
    def __init__(self, filename, data_sz):
        self.data_tf, self.label_tf = TFRecordsProxy(filename, data_sz).main()

    def __getitem__(self, index):
        # data, label = self.data_tf[index], self.label_tf[index]
        data, label = self.data_tf, self.label_tf
        # print('data_tf_out:', data_tf_out)
        # data, label = self.data_tf[index, :, :, :], self.label_tf[index]
        # print('data:', data)
        # print('label:', label)

        return data, label

    def __len__(self):
        # return len(self.data_tf)
        return 20


if __name__ == "__main__":
    pass

    # flower_preprocess()

    # tf_fn = os.path.join(os.path.expanduser('~/GitHub/Quick/pytorch-tfrecords/flower_train.tfrecords'))
    tf_fn = os.path.join(os.path.expanduser('flower_train.tfrecords'))
    data_sz = 10
    dataset = TFRecordsDataset(tf_fn, data_sz)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=5, shuffle=True)

    for batch_idx, (data, target) in enumerate(data_loader):
        for i in range(0, len(data)):
            print(i)
            im = Image.fromarray(data[i, :, :, :].numpy())
            # print('im.size', im.size)
            plt.imshow(im)
            plt.show()
            # im.save(str(epoch) + "_train_" + str(batch_idx) + "_" + str(i) + ".jpeg")
