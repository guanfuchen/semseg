# -*- coding: utf-8 -*-
import argparse

import torch
import os

from torch.autograd import Variable

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.metrics import scores


def validate(args):
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    dst = camvidLoader(local_path, is_transform=True, split='val')
    valloader = torch.utils.data.DataLoader(dst, batch_size=1)

    if os.path.isfile(args.validate_model):
        model = torch.load(args.validate_model)
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
            for gt_, pred_ in zip(gt, pred):
                gts.append(gt_)
                preds.append(pred_)
        score, class_iou = scores(gts, preds, n_class=dst.n_classes)
        for k, v in score.items():
            print(k, v)

        for i in range(dst.n_classes):
            print(i, class_iou[i])
    else:
        print(args.validate_model, ' not exists')

if __name__=='__main__':
    print('validate----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--validate_model', type=str, default='', help='validate model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--dataset_path', type=str, default='', help='train dataset path [ /home/cgf/Data/CamVid ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    print(args)
    validate(args)
    print('validate----out----')
