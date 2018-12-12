# -*- coding: utf-8 -*-
import os

loss_iteration_fp = open(os.path.expanduser('/Users/cgf/GitHub/Quick/master_thesis/杂/检测实验/DRNSegMT_A_small_YOLO_det_loss_iteration_epoch500_lr_time_12_12_1.txt'), 'rb')
loss_iteration_content = loss_iteration_fp.readlines()


smooth_interval = 341
loss_epoch_total = 0
loss_epoch_fp = open('/tmp/tmp.txt', 'wb')
for loss_iteration_content_item in loss_iteration_content:
    loss_iteration_content_item_split = loss_iteration_content_item.strip().split(' ')
    # loss_iteration_content_item_split = loss_iteration_content_item.strip().split('\t')
    # print('loss_iteration_content_item_split:', loss_iteration_content_item_split)
    loss_iteration_id = int(float(loss_iteration_content_item_split[0]))
    loss_iteration_val = float(loss_iteration_content_item_split[1])
    if loss_iteration_id%smooth_interval==0:
        pass
        loss_epoch_avg = loss_epoch_total*1.0/smooth_interval
        loss_epoch_total = 0
        #print('loss_epoch_avg:', loss_epoch_avg)
        loss_epoch_fp.write('{}\n'.format(loss_epoch_avg))
        #break
    else:
        loss_epoch_total += loss_iteration_val
    #break
loss_epoch_fp.close()
