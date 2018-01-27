# -*- coding: utf-8 -*-
import argparse

import torch
import os

from torch.autograd import Variable
import visdom
from scipy import misc
from requests import ConnectionError
import numpy as np

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.metrics import scores
from semseg.modelloader.drn import DRNSeg
from semseg.modelloader.duc_hdc import ResNetDUC
from semseg.modelloader.enet import ENet
from semseg.modelloader.fcn import fcn
from semseg.modelloader.segnet import segnet


def validate(args):
    if args.vis:
        vis = visdom.Visdom()
    if args.dataset_path == '':
        HOME_PATH = os.path.expanduser('~')
        local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    else:
        local_path = args.dataset_path
    dst = camvidLoader(local_path, is_transform=True, split='val')
    valloader = torch.utils.data.DataLoader(dst, batch_size=1)

    # if os.path.isfile(args.validate_model):
    if args.validate_model != '':
        model = torch.load(args.validate_model)
    else:
        if args.structure == 'fcn32s':
            model = fcn(module_type='32s', n_classes=dst.n_classes)
        elif args.structure == 'fcn16s':
            model = fcn(module_type='16s', n_classes=dst.n_classes)
        elif args.structure == 'fcn8s':
            model = fcn(module_type='8s', n_classes=dst.n_classes)
        elif args.structure == 'ResNetDUC':
            model = ResNetDUC(n_classes=dst.n_classes)
        elif args.structure == 'segnet':
            model = segnet(n_classes=dst.n_classes)
        elif args.structure == 'ENet':
            model = ENet(n_classes=dst.n_classes)
        elif args.structure == 'drn_d_22':
            model = DRNSeg(model_name='drn_d_22', n_classes=dst.n_classes)
        elif args.structure == 'pspnet':
            model = pspnet(n_classes=dst.n_classes, use_aux=False)
        elif args.structure == 'erfnet':
            model = erfnet(n_classes=dst.n_classes)
        if args.validate_model_state_dict != '':
            try:
                model.load_state_dict(torch.load(args.validate_model_state_dict))
            except KeyError:
                print('missing key')
    model.eval()

    gts, preds = [], []
    for i, (imgs, labels) in enumerate(valloader):
        print(i)
        #  print(labels.shape)
        #  print(imgs.shape)
        # 将np变量转换为pytorch中的变量
        imgs = Variable(imgs)
        labels = Variable(labels)

        outputs = model(imgs)
        # 取axis=1中的最大值，outputs的shape为batch_size*n_classes*height*width，
        # 获取max后，返回两个数组，分别是最大值和相应的索引值，这里取索引值为label
        pred = outputs.data.max(1)[1].numpy()
        gt = labels.data.numpy()
        # print(pred.dtype)
        # print(gt.dtype)
        # print('pred.shape:', pred.shape)
        # print('gt.shape:', gt.shape)

        if args.vis and i % 50 == 0:
            img = imgs.data.numpy()[0]
            # print(img.shape)
            label_color = dst.decode_segmap(gt[0]).transpose(2, 0, 1)
            # print(label_color.shape)
            pred_label_color = dst.decode_segmap(pred[0]).transpose(2, 0, 1)
            # print(pred_label_color.shape)
            # try:
            #     win = 'label_color'
            #     vis.image(label_color, win=win)
            #     win = 'pred_label_color'
            #     vis.image(pred_label_color, win=win)
            # except ConnectionError:
            #     print('ConnectionError')


            if args.blend:
                img_hwc = img.transpose(1, 2, 0)
                img_hwc = img_hwc*255.0
                img_hwc += np.array([104.00699, 116.66877, 122.67892])
                img_hwc = np.array(img_hwc, dtype=np.uint8)
                # label_color_hwc = label_color.transpose(1, 2, 0)
                pred_label_color_hwc = pred_label_color.transpose(1, 2, 0)
                pred_label_color_hwc = np.array(pred_label_color_hwc, dtype=np.uint8)
                # print(img_hwc.dtype)
                # print(pred_label_color_hwc.dtype)
                label_blend = img_hwc * 0.5 + pred_label_color_hwc * 0.5
                label_blend = np.array(label_blend, dtype=np.uint8)
                misc.imsave('/tmp/label_blend.png', label_blend)

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

if __name__=='__main__':
    # print('validate----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='fcn32s', help='use the net structure to segment [ fcn32s ResNetDUC segnet ENet drn_d_22 ]')
    parser.add_argument('--validate_model', type=str, default='', help='validate model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--validate_model_state_dict', type=str, default='', help='validate model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--dataset_path', type=str, default='', help='train dataset path [ /home/cgf/Data/CamVid ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    parser.add_argument('--blend', type=bool, default=False, help='blend the result and the origin [ False ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    print(args)
    validate(args)
    # print('validate----out----')
