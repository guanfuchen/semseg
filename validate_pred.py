# -*- coding: utf-8 -*-
import torch
import os
import argparse

import cv2
import time
import numpy as np
import visdom
from torch.autograd import Variable
from scipy import misc

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.dataloader.cityscapes_loader import cityscapesLoader
from semseg.dataloader.freespace_loader import freespaceLoader
from semseg.dataloader.movingmnist_loader import movingmnistLoader
from semseg.dataloader.segmpred_loader import segmpredLoader
from semseg.loss import cross_entropy2d
from semseg.metrics import scores
from semseg.modelloader.EDANet import EDANet
from semseg.modelloader.bisenet import BiSeNet
from semseg.modelloader.deeplabv3 import Res_Deeplab_101, Res_Deeplab_50
from semseg.modelloader.drn import drn_d_22, DRNSeg, drn_a_asymmetric_18, drn_a_asymmetric_ibn_a_18, drnseg_a_50, drnseg_a_18, drnseg_a_34, drnseg_e_22, drnseg_a_asymmetric_18, drnseg_a_asymmetric_ibn_a_18, drnseg_d_22, drnseg_d_38
from semseg.modelloader.drn_a_irb import drnsegirb_a_18
from semseg.modelloader.drn_a_refine import drnsegrefine_a_18
from semseg.modelloader.drn_pred import drnsegpred_a_18
from semseg.modelloader.duc_hdc import ResNetDUC, ResNetDUCHDC
from semseg.modelloader.enet import ENet
from semseg.modelloader.enetv2 import ENetV2
from semseg.modelloader.erfnet import erfnet
from semseg.modelloader.fc_densenet import fcdensenet103, fcdensenet56, fcdensenet_tiny
from semseg.modelloader.fcn import fcn, fcn_32s, fcn_16s, fcn_8s
from semseg.modelloader.fcn_mobilenet import fcn_MobileNet, fcn_MobileNet_32s, fcn_MobileNet_16s, fcn_MobileNet_8s
from semseg.modelloader.fcn_resnet import fcn_resnet18, fcn_resnet34, fcn_resnet18_32s, fcn_resnet18_16s, \
    fcn_resnet18_8s, fcn_resnet34_32s, fcn_resnet34_16s, fcn_resnet34_8s, fcn_resnet50_32s, fcn_resnet50_16s, fcn_resnet50_8s
from semseg.modelloader.lrn import lrn_vgg16
from semseg.modelloader.segnet import segnet, segnet_squeeze, segnet_alignres, segnet_vgg19
from semseg.modelloader.segnet_unet import segnet_unet
from semseg.modelloader.sqnet import sqnet


def validate(args):
    init_time = str(int(time.time()))
    if args.vis:
        vis = visdom.Visdom()

    local_path = os.path.expanduser(args.dataset_path)
    if args.dataset == 'CamVid':
        dst = camvidLoader(local_path, is_transform=True, split=args.dataset_type)
    elif args.dataset == 'CityScapes':
        dst = cityscapesLoader(local_path, is_transform=True, split=args.dataset_type)
    elif args.dataset == 'SegmPred':
        dst = segmpredLoader(local_path, is_transform=True, split=args.dataset_type)
    elif args.dataset == 'MovingMNIST':
        dst = movingmnistLoader(local_path, is_transform=True, split=args.dataset_type)
    elif args.dataset == 'FreeSpace':
        dst = freespaceLoader(local_path, is_transform=True, split=args.dataset_type)
    else:
        pass
    val_loader = torch.utils.data.DataLoader(dst, batch_size=1, shuffle=False)

    # if os.path.isfile(args.validate_model):
    if args.validate_model != '':
        model = torch.load(args.validate_model)
    else:
        # ---------------for testing SegmPred---------------
        try:
            model = drnsegpred_a_18(n_classes=args.n_classes, pretrained=args.init_vgg16, input_shape=dst.input_shape)
        except:
            print('missing structure or not support')
            exit(0)
        if args.validate_model_state_dict != '':
            try:
                model.load_state_dict(torch.load(args.validate_model_state_dict, map_location='cpu'))
            except KeyError:
                print('missing key')
        # ---------------for testing SegmPred---------------
    if args.cuda:
        model.cuda()
    # some model load different mode different performance
    model.eval()
    # model.train()

    gts, preds, errors, imgs_name = [], [], [], []
    for i, (imgs, labels) in enumerate(val_loader):
        print(i)
        # if i==1:
        #     break
        img_path = dst.files[args.dataset_type][i]
        img_name = img_path[img_path.rfind('/')+1:]
        imgs_name.append(img_name)
        # print('img_path:', img_path)
        # print('img_name:', img_name)
        #  print(labels.shape)
        #  print(imgs.shape)
        # 将np变量转换为pytorch中的变量
        imgs = Variable(imgs, volatile=True)
        labels = Variable(labels, volatile=True)

        if args.cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()

        # print('imgs.shape', imgs.shape)
        # print('labels.shape', labels.shape)

        outputs = model(imgs)
        # print('outputs.shape', outputs.shape)
        loss = cross_entropy2d(outputs, labels)
        loss_np = loss.cpu().data.numpy()
        loss_np_float = float(loss_np)

        # print('loss_np_float:', loss_np_float)
        errors.append(loss_np_float)

        # 取axis=1中的最大值，outputs的shape为batch_size*n_classes*height*width，
        # 获取max后，返回两个数组，分别是最大值和相应的索引值，这里取索引值为label
        pred = outputs.cpu().data.max(1)[1].numpy()
        gt = labels.cpu().data.numpy()

        if args.save_result:
            if not os.path.exists('/tmp/'+init_time):
                os.mkdir('/tmp/'+init_time)
            pred_labels = outputs.cpu().data.max(1)[1].numpy()
            # print('pred_labels.shape:', pred_labels.shape)
            label_color = dst.decode_segmap(labels.cpu().data.numpy()[0]).transpose(2, 0, 1)
            pred_label_color = dst.decode_segmap(pred_labels[0]).transpose(2, 0, 1)
            # print('label_color.shape:', label_color.shape)
            # print('pred_label_color.shape:', pred_label_color.shape)

            label_color_cv2 = label_color.transpose(1, 2, 0)
            label_color_cv2 = cv2.cvtColor(label_color_cv2, cv2.COLOR_RGB2BGR)
            # print('label_color_cv2.shape:', label_color_cv2.shape)
            # print('label_color_cv2.dtype:', label_color_cv2.dtype)
            # cv2.imshow('label_color_cv2', label_color_cv2)
            # cv2.waitKey()
            cv2.imwrite('/tmp/'+init_time+'/gt_{}.png'.format(img_name), label_color_cv2)

            pred_label_color_cv2 = pred_label_color.transpose(1, 2, 0)
            pred_label_color_cv2 = cv2.cvtColor(pred_label_color_cv2, cv2.COLOR_RGB2BGR)
            cv2.imwrite('/tmp/'+init_time+'/pred_{}.png'.format(img_name), pred_label_color_cv2)

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    # print('errors:', errors)
    # print('imgs_name:', imgs_name)

    errors_indices = np.argsort(errors).tolist()
    print('errors_indices:', errors_indices)
    # for top_i in range(len(errors_indices)):
    for top_i in range(10):
        top_index = errors_indices.index(top_i)
        # print('top_index:', top_index)
        img_name_top = imgs_name[top_index]
        print('img_name_top:', img_name_top)

    score, class_iou = scores(gts, preds, n_class=dst.n_classes)
    for k, v in score.items():
        print(k, v)

    class_iou_list = []
    for i in range(dst.n_classes):
        class_iou_list.append(round(class_iou[i], 2))
        # print(i, round(class_iou[i], 2))
    print('classes:', range(dst.n_classes))
    print('class_iou_list:', class_iou_list)


# best validate: python validate.py --structure fcn32s --validate_model_state_dict fcn32s_camvid_9.pt
if __name__=='__main__':
    # print('validate----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='fcn32s', help='use the net structure to segment [ fcn32s ResNetDUC segnet ENet drn_d_22 ]')
    parser.add_argument('--validate_model', type=str, default='', help='validate model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--validate_model_state_dict', type=str, default='', help='validate model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset', type=str, default='CamVid', help='train dataset [ CamVid CityScapes FreeSpace SegmPred MovingMNIST ]')
    parser.add_argument('--dataset_path', type=str, default='~/Data/CamVid', help='train dataset path [ ~/Data/CamVid ~/Data/cityscapes ~/Data/FreeSpaceDataset ~/Data/SegmPred ~/Data/mnist_test_seq.npy]')
    parser.add_argument('--dataset_type', type=str, default='val', help='dataset type [ train val test ]')
    parser.add_argument('--n_classes', type=int, default=12, help='train class num [ 12 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda [ False ]')
    parser.add_argument('--save_result', type=bool, default=False, help='save the val dataset prediction result [ False True ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    print(args)
    validate(args)
    # print('validate----out----')
