# -*- coding: utf-8 -*-_resnet18_32s
import torch
import os
import argparse

import cv2
import time
import numpy as np
import visdom
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchvision import transforms

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.dataloader.cityscapes_loader import cityscapesLoader
from semseg.dataloader.yolodataset_loader import yoloDataset
from semseg.loss import cross_entropy2d
from semseg.metrics import scores
from semseg.modelloader.drn_a_mt import drnsegmt_a_18
from semseg.schedulers import ConstantLR, PolynomialLR
from semseg.utils.get_class_weights import median_frequency_balancing, ENet_weighing
from semseg.yoloLoss import yoloLoss

def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index)
                        # print('cls_index:', cls_index)
                        probs.append(contain_prob*max_prob)
    # print('boxes:', boxes)
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        # print('boxes.shape:', len(boxes))
        # print('probs.shape:', len(probs))
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.stack(cls_indexs,0) #(n,)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def validate(args):
    init_time = str(int(time.time()))
    if args.vis:
        # start visdom and close all window
        vis = visdom.Visdom()
        vis.close()

        # vis_text_usage = 'Operating in the text window<br>Press s to save data<br>'
        # callback_text_usage_window = vis.text(vis_text_usage)
        # vis.register_event_handler(type_callback, callback_text_usage_window)

    class_weight = None
    local_path = os.path.expanduser(args.dataset_path)
    train_dst = None
    val_dst = None
    if args.dataset == 'CamVid':
        train_dst = camvidLoader(local_path, is_transform=True, is_augment=args.data_augment, split='train')
        val_dst = camvidLoader(local_path, is_transform=True, is_augment=False, split='val')
    elif args.dataset == 'CityScapes':
        train_dst = cityscapesLoader(local_path, is_transform=True, split='train')
        val_dst = cityscapesLoader(local_path, is_transform=True, split='val')
    else:
        print('{} dataset does not implement'.format(args.dataset))
        exit(0)

    if args.cuda:
        if class_weight is not None:
            class_weight = class_weight.cuda()
    print('class_weight:', class_weight)

    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=1, shuffle=True)

    yolo_B = 2
    yolo_C = 2
    yolo_S = 7
    yolo_out_tensor_shape = yolo_B * 5 + yolo_C
    print('yolo_out_tensor_shape:', yolo_out_tensor_shape)

    det_file_root = os.path.expanduser('~/Data/CamVid/train/')
    det_train_dst = yoloDataset(root=det_file_root, list_file=['camvid_det.txt'], train=False, transform=[transforms.ToTensor()], yolo_out_tensor_shape=yolo_out_tensor_shape)
    det_train_loader = torch.utils.data.DataLoader(det_train_dst, batch_size=1, shuffle=False)

    model = drnsegmt_a_18(pretrained=args.init_vgg16, n_classes=args.n_classes, det_tensor_num=yolo_out_tensor_shape)
    if args.resume_model_state_dict != '':
        pretrained_dict = torch.load(args.resume_model_state_dict, map_location='cpu')
        model.load_state_dict(pretrained_dict)
    else:
        print('missing resume_model_state_dict')
        exit()

    if args.cuda:
        model.cuda()

    model.eval()
    for epoch in range(0, 1, 1):
        # ----for object detection----
        for det_i, (det_imgs, det_labels, det_imgs_ori) in enumerate(det_train_loader):
            print('det_imgs.shape:', det_imgs.shape)
            print('det_labels.shape:', det_labels.shape)
            # det_imgs_height = det_imgs.shape[2]
            # det_imgs_width = det_imgs.shape[3]
            # print('det_imgs_height:', det_imgs_height)
            # print('det_imgs_width:', det_imgs_width)


            det_imgs = Variable(det_imgs)
            det_labels = Variable(det_labels)

            if args.cuda:
                det_imgs = det_imgs.cuda()
                det_labels = det_labels.cuda()

            _, outputs_det = model(det_imgs)
            # print('outpust_det:', outputs_det.shape)

            # det_loss = det_criterion(outputs_det, det_labels)
            # det_loss_np = det_loss.cpu().data.numpy()
            outputs_det = outputs_det.cpu()
            det_boxes, det_cls_indexs, det_probs = decoder(outputs_det)

            image_ori = det_imgs_ori[0, ...].cpu().data.numpy()
            det_imgs_ori_height = image_ori.shape[0]
            det_imgs_ori_width = image_ori.shape[1]
            # image = image.transpose(1, 2, 0)
            for i, det_box in enumerate(det_boxes):
                x1 = int(det_box[0] * det_imgs_ori_width)
                x2 = int(det_box[2] * det_imgs_ori_width)
                y1 = int(det_box[1] * det_imgs_ori_height)
                y2 = int(det_box[3] * det_imgs_ori_height)
                det_cls_index = det_cls_indexs[i]
                det_cls_index = int(det_cls_index)  # convert LongTensor to int

                det_prob = det_probs[i]
                det_prob = float(det_prob)
                if x1<0 or x1>det_imgs_ori_width-1:
                    continue
                if x2<0 or x2>det_imgs_ori_width-1:
                    continue
                if y1<0 or y1>det_imgs_ori_height-1:
                    continue
                if y2<0 or y2>det_imgs_ori_height-1:
                    continue
                # x1 = np.clip(x1, 0, det_imgs_ori_width-1)
                # x2 = np.clip(x2, 0, det_imgs_ori_width-1)
                # y1 = np.clip(y1, 0, det_imgs_ori_height-1)
                # y2 = np.clip(y2, 0, det_imgs_ori_height-1)

                if det_prob>0:
                    print('(x1,y1)->(x2,y2):({},{})->({},{})'.format(x1, y1, x2, y2))
                    cv2.rectangle(image_ori, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.imshow('image_ori', image_ori)
            cv2.waitKey()

        # ----for object detection----

        # # ----for semantic segment----
        # for i, (imgs, labels) in enumerate(train_loader):
        #     # if i==1:
        #     #     break
        #     # model.train()
        #
        #     # 最后的几张图片可能不到batch_size的数量，比如batch_size=4，可能只剩3张
        #     imgs_batch = imgs.shape[0]
        #     if imgs_batch != args.batch_size:
        #         break
        #     # iteration_step += 1
        #
        #     imgs = Variable(imgs)
        #     labels = Variable(labels)
        #
        #     if args.cuda:
        #         imgs = imgs.cuda()
        #         labels = labels.cuda()
        #     outputs_sem, _ = model(imgs)
        #     # print('outputs_sem.shape:', outputs_sem.shape)
        #
        #     # print('outputs.size:', outputs.size())
        #     # print('labels.size:', labels.size())
        #
        #     loss = cross_entropy2d(outputs_sem, labels, weight=class_weight)
        #     loss_np = loss.cpu().data.numpy()
        #     loss_epoch += loss_np
        #
        #     if args.vis and i%50==0:
        #         pred_labels = outputs_sem.cpu().data.max(1)[1].numpy()
        #         label_color = train_dst.decode_segmap(labels.cpu().data.numpy()[0]).transpose(2, 0, 1)
        #         pred_label_color = train_dst.decode_segmap(pred_labels[0]).transpose(2, 0, 1)
        #         win = 'label_color'
        #         vis.image(label_color, win=win, opts=dict(title='Gt', caption='Ground Truth'))
        #         win = 'pred_label_color'
        #         vis.image(pred_label_color, win=win, opts=dict(title='Pred', caption='Prediction'))
        #
        #     # 显示一个周期的loss曲线
        #     if args.vis:
        #         win = 'loss_iteration'
        #         loss_np_expand = np.expand_dims(loss_np, axis=0)
        #         win_res = vis.line(X=np.ones(1)*(i+data_count*(epoch-1)+1), Y=loss_np_expand, win=win, update='append')
        #         if win_res != win:
        #             vis.line(X=np.ones(1)*(i+data_count*(epoch-1)+1), Y=loss_np_expand, win=win, opts=dict(title=win, xlabel='iteration', ylabel='loss'))
        # # ----for semantic segment----

# best training: python train.py --resume_model fcn32s_camvid_9.pkl --save_model True
# --init_vgg16 True --dataset_path /home/cgf/Data/CamVid --batch_size 1 --vis True
if __name__=='__main__':
    # print('train----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='ENetV2', help='use the net structure to segment [ fcn_32s ResNetDUC segnet ENet drn_d_22 ]')
    parser.add_argument('--solver', type=str, default='SGD', help='use the solver to optimizer net [ SGD ]')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--resume_model_state_dict', type=str, default='', help='resume model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--save_epoch', type=int, default=1, help='save model after epoch [ 1 ]')
    parser.add_argument('--training_epoch', type=int, default=500, help='training epoch end training model [ 30000 ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset', type=str, default='CamVid', help='train dataset [ CamVid CityScapes ]')
    parser.add_argument('--dataset_path', type=str, default='~/Data/CamVid', help='train dataset path [ ~/Data/CamVid ~/Data/cityscapes ]')
    parser.add_argument('--data_augment', type=bool, default=True, help='enlarge the training data [ True False ]')
    parser.add_argument('--class_weighting', type=str, default='MFB', help='weighting class [ MFB ENET ]')
    parser.add_argument('--batch_size', type=int, default=1, help='train dataset batch size [ 1 ]')
    parser.add_argument('--val_interval', type=int, default=-1, help='val dataset interval unit epoch [ 3 ]')
    parser.add_argument('--n_classes', type=int, default=12, help='train class num [ 12 ]')
    parser.add_argument('--lr', type=float, default=1e-4, help='train learning rate [ 0.00001 ]')
    parser.add_argument('--lr_policy', type=str, default='Polynomial', help='train learning policy [ Constant Polynomial ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda [ False ]')
    args = parser.parse_args()
    print(args)
    validate(args)
    # print('train----out----')
