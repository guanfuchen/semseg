# -*- coding: utf-8 -*-
import os
import numpy as np

iou_epoch_fp = open(os.path.expanduser('/Users/cgf/GitHub/Quick/master_thesis/杂/检测实验/DRNSegMT_A_small_YOLO_miou_epoch500_lr_time_12_12_1.txt'), 'rb')
# iou_epoch_fp = open(os.path.expanduser('/Users/cgf/GitHub/Quick/master_thesis/杂/检测实验/DRNSegMT0_A_resnet18_32s_miou_epoch500_lr_time_12_12_1.txt'), 'rb')
iou_epoch_content = iou_epoch_fp.readlines()

iou_expand_interval = 5

temp_fp = open('/tmp/tmp.txt', 'wb')
iou_max = -np.inf
for iou_epoch_content_item in iou_epoch_content:
    iou_epoch_content_item_split = iou_epoch_content_item.strip().split('\t')
    # print('iou_epoch_content_item_split:', iou_epoch_content_item_split)
    iou_epoch_id = int(float(iou_epoch_content_item_split[0]))
    iou_epoch_val = float(iou_epoch_content_item_split[1])
    # if iou_epoch_val>0.5:
    #     iou_epoch_val -= 0.01
    if iou_max<iou_epoch_val:
        iou_max = iou_epoch_val
    temp_fp.write('{}\n'.format(iou_epoch_val) * iou_expand_interval)
    # break

print('iou_max:', iou_max)
temp_fp.close()
