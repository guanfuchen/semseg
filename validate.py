# -*- coding: utf-8 -*-
import torch
import os

from torch.autograd import Variable

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.metrics import scores


def validate():
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    dst = camvidLoader(local_path, is_transform=True, split='val')
    valloader = torch.utils.data.DataLoader(dst, batch_size=1)

    model = torch.load('fcn32s_camvid.pkl')
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


if __name__=='__main__':
    print('validate----in----')
    validate()
    print('validate----out----')
