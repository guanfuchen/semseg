# -*- coding: utf-8 -*-
import torch
import os
import argparse

import numpy as np
from torch.autograd import Variable

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.loss import cross_entropy2d
from semseg.modelloader.fcn import fcn32s


def train(args):
    if args.dataset_path == '':
        HOME_PATH = os.path.expanduser('~')
        local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    else:
        local_path = args.dataset_path
    dst = camvidLoader(local_path, is_transform=True)
    trainloader = torch.utils.data.DataLoader(dst, batch_size=1)

    start_epoch = 0
    if args.resume_model != '':
        model = torch.load(args.resume_model)
        start_epoch_id1 = args.resume_model.rfind('_')
        start_epoch_id2 = args.resume_model.rfind('.')
        start_epoch = int(args.resume_model[start_epoch_id1+1:start_epoch_id2])
    else:
        model = fcn32s(n_classes=dst.n_classes)
        if args.init_vgg16:
            model.init_vgg16()

    print('start_epoch:', start_epoch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.99, weight_decay=5e-4)
    for epoch in range(start_epoch, 20000, 1):
        for i, (imgs, labels) in enumerate(trainloader):
            print(i)
            # print(labels.shape)
            # print(imgs.shape)
            imgs = Variable(imgs)
            labels = Variable(labels)
            pred = model(imgs)
            optimizer.zero_grad()

            loss = cross_entropy2d(pred, labels)
            print('loss:', loss)
            loss.backward()

            optimizer.step()
        if args.save_model:
            torch.save(model, 'fcn32s_camvid_{}.pkl'.format(epoch))


# best training: python train.py --resume_model fcn32s_camvid_9.pkl --save_model True --init_vgg16 True --dataset_path /home/cgf/Data/CamVid
if __name__=='__main__':
    print('train----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset_path', type=str, default='', help='train dataset path [ /home/cgf/Data/CamVid ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    print(args)
    train(args)
    print('train----out----')
