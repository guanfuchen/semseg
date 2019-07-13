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


def _dtype_feature(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:
        raise ValueError("The input should be numpy ndarray. \
                           Instaed got {}".format(ndarray.dtype))


def write_tfrecords_batch(start, end, labels, flower_dir, file):
    writer = tf.python_io.TFRecordWriter(file)
    batch = 2
    widths = [236, 256, 276]
    heights = [236, 256, 276]

    width_resize = widths[0]
    height_resize = heights[0]
    imgs = []
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
        img = img.resize((width_resize, height_resize))
        imgs.append(np.array(img).transpose(2, 1, 0))
        if (id+1)%batch==0 and id!=0:
            imgs = np.array(imgs, dtype=np.float)
            imgs = imgs.reshape(-1)
            print('imgs.shape:', imgs.shape)
            pass
            example = tf.train.Example(features=tf.train.Features(feature={
                # 'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                # 'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_500])),
                # 'img_batch': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                # 'img_batch': _dtype_feature(imgs),
                # tf.train.Features(feature={"bytes": _floats_feature(numpy_arr)})
                # 'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[500])),
                # 'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[500]))
                'imgs': tf.train.Feature(float_list=tf.train.FloatList(value=imgs)),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height_resize])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width_resize]))
            }))
            # writer_500.write(example_500.SerializeToString())
            # print(example)
            writer.write(example.SerializeToString())
            imgs = []
            resize_id = random.randint(0, 2)
            width_resize = widths[resize_id]
            height_resize = heights[resize_id]



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


def flower_preprocess(flower_folder='/Users/cgf/Data/tmp/tfrecords', tf_fn="flower_train.tfrecords"):

    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
              1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    random.shuffle(labels)
    flower_dir = list()

    for img in os.listdir(flower_folder):
        flower_dir.append(os.path.join(flower_folder, img))
    flower_dir.sort()

    length = int(len(flower_dir)*0.7)
    # file = "flower_train.tfrecords"
    # write_tfrecords(0, length, labels, flower_dir, tf_fn)
    write_tfrecords_batch(0, length, labels, flower_dir, tf_fn)


class TFRecordsProxy():
    def __init__(self, filenames, data_size):
        self.filenames = filenames
        self.data_size = data_size

    def parse_data(self, example_proto):
        features = {
                    # 'img': tf.FixedLenFeature([], tf.string, ''),
                    'imgs': tf.VarLenFeature(tf.float32),
                    # 'label': tf.FixedLenFeature([], tf.int64, 0),
                    # 'width': tf.FixedLenFeature([], tf.int64, 0),
                    # 'height': tf.FixedLenFeature([], tf.int64, 0),
                    }
        parsed_features = tf.parse_single_example(example_proto, features)
        # imgs = tf.decode_raw(parsed_features['imgs'], tf.uint8)
        label = 0
        imgs = tf.cast(parsed_features['imgs'], tf.float32)
        # width = tf.cast(parsed_features['width'], tf.int64)
        # height = tf.cast(parsed_features['height'], tf.int64)
        # image = tf.reshape(image, tf.stack([height, width, 3]))
        # print('width:', width)
        # print('height:', height)
        # image = tf.reshape(image, [height, width, 3])
        # print('image:', image)
        return imgs, label
        # return width, label

    def my_input_fn(self, filenames, data_size):
        reader = tf.TFRecordReader()

        filename_queue = tf.train.string_input_producer([filenames], num_epochs=1)
        _, serialized_example = reader.read(filename_queue)
        batch = tf.train.batch(tensors=[serialized_example], batch_size=1)
        features = {
                    'imgs': tf.VarLenFeature(tf.float32),
                    }
        key_parsed = tf.parse_example(batch, features)
        # print tf.contrib.learn.run_n(key_parsed)
        # dataset = tf.contrib.data.TFRecordDataset(filenames)
        # dataset = dataset.map(self.parse_data)
        # # dataset = dataset.batch(data_size)
        #
        # # dataset = dataset.repeat()
        # iterator = dataset.make_one_shot_iterator()
        # features, labels = iterator.get_next()
        # return features, labels
        return key_parsed, 1

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

    # flower_preprocess(tf_fn="flower_train_batch.tfrecords")

    tf_fn = os.path.join(os.path.expanduser('~/GitHub/Quick/pytorch-tfrecords/flower_train.tfrecords'))
    # tf_fn = os.path.join(os.path.expanduser('flower_train.tfrecords'))
    tf_fn = os.path.join(os.path.expanduser('flower_train_batch.tfrecords'))
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
