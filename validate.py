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
from semseg.loss import cross_entropy2d
from semseg.metrics import scores
from semseg.modelloader.EDANet import EDANet
from semseg.modelloader.deeplabv3 import Res_Deeplab_101, Res_Deeplab_50
from semseg.modelloader.drn import drn_d_22, DRNSeg, drn_a_asymmetric_18, drnseg_a_50, drnseg_a_18, drnseg_e_22, \
    drnseg_a_asymmetric_18, drnseg_d_22
from semseg.modelloader.duc_hdc import ResNetDUC, ResNetDUCHDC
from semseg.modelloader.enet import ENet
from semseg.modelloader.enetv2 import ENetV2
from semseg.modelloader.erfnet import erfnet
from semseg.modelloader.fc_densenet import fcdensenet103, fcdensenet56, fcdensenet_tiny
from semseg.modelloader.fcn import fcn, fcn_32s, fcn_16s, fcn_8s
from semseg.modelloader.fcn_mobilenet import fcn_MobileNet, fcn_MobileNet_32s, fcn_MobileNet_16s, fcn_MobileNet_8s
from semseg.modelloader.fcn_resnet import fcn_resnet18, fcn_resnet34, fcn_resnet18_32s, fcn_resnet18_16s, \
    fcn_resnet18_8s, fcn_resnet34_32s, fcn_resnet34_16s, fcn_resnet34_8s, fcn_resnet50_32s, fcn_resnet50_16s, fcn_resnet50_8s
from semseg.modelloader.segnet import segnet, segnet_squeeze, segnet_alignres, segnet_vgg19
from semseg.modelloader.segnet_unet import segnet_unet
from semseg.modelloader.sqnet import sqnet


def validate(args):
    init_time = str(int(time.time()))
    if args.vis:
        vis = visdom.Visdom()
    if args.dataset_path == '':
        HOME_PATH = os.path.expanduser('~')
        local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    else:
        local_path = args.dataset_path
    local_path = os.path.expanduser(args.dataset_path)
    if args.dataset == 'CamVid':
        dst = camvidLoader(local_path, is_transform=True, split=args.dataset_type)
    elif args.dataset == 'CityScapes':
        dst = cityscapesLoader(local_path, is_transform=True)
    else:
        pass
    val_loader = torch.utils.data.DataLoader(dst, batch_size=1)

    # if os.path.isfile(args.validate_model):
    if args.validate_model != '':
        model = torch.load(args.validate_model)
    else:
        try:
            model = eval(args.structure)(n_classes=args.n_classes, pretrained=args.init_vgg16)
        except:
            print('missing structure or not support')
            exit(0)
        if args.validate_model_state_dict != '':
            try:
                model.load_state_dict(torch.load(args.validate_model_state_dict))
            except KeyError:
                print('missing key')
    if args.cuda:
        model.cuda()
    model.eval()

    gts, preds = [], []
    for i, (imgs, labels) in enumerate(val_loader):
        print(i)
        #  print(labels.shape)
        #  print(imgs.shape)
        # 将np变量转换为pytorch中的变量
        imgs = Variable(imgs)
        labels = Variable(labels)

        if args.cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()

        outputs = model(imgs)
        # 取axis=1中的最大值，outputs的shape为batch_size*n_classes*height*width，
        # 获取max后，返回两个数组，分别是最大值和相应的索引值，这里取索引值为label
        pred = outputs.cpu().data.max(1)[1].numpy()
        gt = labels.cpu().data.numpy()
        # print(pred.dtype)
        # print(gt.dtype)
        # print('pred.shape:', pred.shape)
        # print('gt.shape:', gt.shape)

        # if args.vis and i % 1 == 0:
        #     img = imgs.cpu().data.numpy()[0]
        #     # print(img.shape)
        #     label_color = dst.decode_segmap(gt[0]).transpose(2, 0, 1)
        #     # print(label_color.shape)
        #     pred_label_color = dst.decode_segmap(pred[0]).transpose(2, 0, 1)
        #     # print(pred_label_color.shape)
        #     # try:
        #     #     win = 'label_color'
        #     #     vis.image(label_color, win=win)
        #     #     win = 'pred_label_color'
        #     #     vis.image(pred_label_color, win=win)
        #     # except ConnectionError:
        #     #     print('ConnectionError')
        #
        #
        #     if args.blend:
        #         img_hwc = img.transpose(1, 2, 0)
        #         img_hwc = img_hwc*255.0
        #         img_hwc += np.array([104.00699, 116.66877, 122.67892])
        #         img_hwc = np.array(img_hwc, dtype=np.uint8)
        #         # label_color_hwc = label_color.transpose(1, 2, 0)
        #         pred_label_color_hwc = pred_label_color.transpose(1, 2, 0)
        #         pred_label_color_hwc = np.array(pred_label_color_hwc, dtype=np.uint8)
        #         # print(img_hwc.dtype)
        #         # print(pred_label_color_hwc.dtype)
        #         label_blend = img_hwc * 0.5 + pred_label_color_hwc * 0.5
        #         label_blend = np.array(label_blend, dtype=np.uint8)
        #
        #         if not os.path.exists('/tmp/' + init_time):
        #             os.mkdir('/tmp/' + init_time)
        #         time_str = str(int(time.time()))
        #
        #         misc.imsave('/tmp/'+init_time+'/'+time_str+'_label_blend.png', label_blend)

        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)


    score, class_iou = scores(gts, preds, n_class=dst.n_classes)
    for k, v in score.items():
        print(k, v)

    for i in range(dst.n_classes):
        print(i, class_iou[i])
    # else:
    #     print(args.validate_model, ' not exists')
# best validate: python validate.py --structure fcn32s --validate_model_state_dict fcn32s_camvid_9.pt
if __name__=='__main__':
    # print('validate----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='fcn32s', help='use the net structure to segment [ fcn32s ResNetDUC segnet ENet drn_d_22 ]')
    parser.add_argument('--validate_model', type=str, default='', help='validate model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--validate_model_state_dict', type=str, default='', help='validate model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset', type=str, default='CamVid', help='train dataset [ CamVid CityScapes ]')
    parser.add_argument('--dataset_path', type=str, default='~/Data/CamVid', help='train dataset path [ ~/Data/CamVid ~/Data/cityscapes ]')
    parser.add_argument('--dataset_type', type=str, default='val', help='dataset type [ train val test ]')
    parser.add_argument('--n_classes', type=int, default=12, help='train class num [ 12 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda [ False ]')
    parser.add_argument('--blend', type=bool, default=False, help='blend the result and the origin [ False ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    print(args)
    validate(args)
    # print('validate----out----')
