# -*- coding: utf-8 -*-
import torch
import os
import argparse

import numpy as np
import visdom
from torch.autograd import Variable

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.loss import cross_entropy2d
from semseg.modelloader.drn import drn_d_22, DRNSeg
from semseg.modelloader.duc_hdc import ResNetDUC
from semseg.modelloader.enet import ENet
from semseg.modelloader.fcn import fcn
from semseg.modelloader.pspnet import pspnet
from semseg.modelloader.segnet import segnet


def train(args):
    vis = visdom.Visdom()
    if args.dataset_path == '':
        HOME_PATH = os.path.expanduser('~')
        local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    else:
        local_path = args.dataset_path
    dst = camvidLoader(local_path, is_transform=True, is_augment=args.data_augment)
    trainloader = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True)

    start_epoch = 0
    if args.resume_model != '':
        model = torch.load(args.resume_model)
        start_epoch_id1 = args.resume_model.rfind('_')
        start_epoch_id2 = args.resume_model.rfind('.')
        start_epoch = int(args.resume_model[start_epoch_id1+1:start_epoch_id2])
    else:
        if args.structure == 'fcn32s':
            model = fcn(module_type='32s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn16s':
            model = fcn(module_type='16s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn8s':
            model = fcn(module_type='8s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'ResNetDUC':
            model = ResNetDUC(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'segnet':
            model = segnet(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'ENet':
            model = ENet(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'drn_d_22':
            model = DRNSeg(model_name='drn_d_22', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'drn_d_22':
            model = pspnet(n_classes=dst.n_classes, pretrained=args.init_vgg16, use_aux=False)
        if args.resume_model_state_dict != '':
            try:
                # fcn32s、fcn16s和fcn8s模型略有增加参数，互相赋值重新训练过程中会有KeyError，暂时捕捉异常处理
                model.load_state_dict(torch.load(args.resume_model_state_dict))
            except KeyError:
                print('missing key')



    print('start_epoch:', start_epoch)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    for epoch in range(start_epoch+1, 20000, 1):
        for i, (imgs, labels) in enumerate(trainloader):
            print(i)
            # print(labels.shape)
            # print(imgs.shape)

            imgs = Variable(imgs)
            labels = Variable(labels)

            outputs = model(imgs)

            if args.vis and i%50==0:
                pred_labels = outputs.data.max(1)[1].numpy()
                # print(pred_labels.shape)
                label_color = dst.decode_segmap(labels.data.numpy()[0]).transpose(2, 0, 1)
                # print(label_color.shape)
                pred_label_color = dst.decode_segmap(pred_labels[0]).transpose(2, 0, 1)
                # print(pred_label_color.shape)
                vis.image(label_color, win='label_color')
                vis.image(pred_label_color, win='pred_label_color')


            # print(outputs.size())
            # print(labels.size())
            optimizer.zero_grad()
            loss = cross_entropy2d(outputs, labels)
            print('loss:', loss.data.numpy()[0])
            loss.backward()

            optimizer.step()

            if args.vis:
                win = 'loss'
                win_res = vis.line(X=np.ones(1)*i, Y=loss.data.numpy(), win=win, update='append')
                if win_res != win:
                    vis.line(X=np.ones(1)*i, Y=loss.data.numpy(), win=win)
        if args.save_model and epoch%args.save_epoch==0:
            torch.save(model.state_dict(), '{}_camvid_{}.pt'.format(args.structure, epoch))


# best training: python train.py --resume_model fcn32s_camvid_9.pkl --save_model True
# --init_vgg16 True --dataset_path /home/cgf/Data/CamVid --batch_size 1 --vis True
if __name__=='__main__':
    # print('train----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='fcn32s', help='use the net structure to segment [ fcn32s ResNetDUC segnet ENet drn_d_22 ]')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--resume_model_state_dict', type=str, default='', help='resume model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--save_epoch', type=int, default=1, help='save model after epoch [ 1 ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset_path', type=str, default='', help='train dataset path [ /home/cgf/Data/CamVid ]')
    parser.add_argument('--data_augment', type=bool, default=False, help='enlarge the training data [ False ]')
    parser.add_argument('--batch_size', type=int, default=1, help='train dataset batch size [ 1 ]')
    parser.add_argument('--lr', type=float, default=1e-5, help='train learning rate [ 0.01 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    print(args)
    train(args)
    # print('train----out----')
