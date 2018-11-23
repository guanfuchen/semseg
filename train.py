# -*- coding: utf-8 -*-_resnet18_32s
import torch
import os
import argparse

import cv2
import time
import numpy as np
import visdom
from torch.autograd import Variable

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.dataloader.cityscapes_loader import cityscapesLoader
from semseg.loss import cross_entropy2d
from semseg.modelloader.deeplabv3 import Res_Deeplab_101, Res_Deeplab_50
from semseg.modelloader.drn import drn_d_22, DRNSeg
from semseg.modelloader.duc_hdc import ResNetDUC, ResNetDUCHDC
from semseg.modelloader.enet import ENet
from semseg.modelloader.enetv2 import ENetV2
from semseg.modelloader.erfnet import erfnet
from semseg.modelloader.fc_densenet import fcdensenet103, fcdensenet56, fcdensenet_tiny
from semseg.modelloader.fcn import fcn
from semseg.modelloader.fcn_mobilenet import fcn_MobileNet
from semseg.modelloader.fcn_resnet import fcn_resnet18, fcn_resnet34
from semseg.modelloader.segnet import segnet, segnet_squeeze, segnet_alignres, segnet_vgg19
from semseg.modelloader.segnet_unet import segnet_unet
from semseg.modelloader.sqnet import sqnet




def train(args):
    def type_callback(event):
        # print('event_type:{}'.format(event['event_type']))
        if event['event_type'] == 'KeyPress':
            event_key = event['key']
            if event_key == 'Enter':
                pass
                # print('event_type:Enter')
            elif event_key == 'Backspace':
                pass
                # print('event_type:Backspace')
            elif event_key == 'Delete':
                pass
                # print('event_type:Delete')
            elif len(event_key) == 1:
                pass
                # print('event_key:{}'.format(event['key']))
                if event_key=='s':
                    import json
                    win = 'loss_iteration'
                    win_data = vis.get_window_data(win)
                    win_data_dict = json.loads(win_data)
                    win_data_content_dict = win_data_dict['content']
                    win_data_x = np.array(win_data_content_dict['data'][0]['x'])
                    win_data_y = np.array(win_data_content_dict['data'][0]['y'])

                    win_data_save_file = '/tmp/loss_iteration_{}.txt'.format(init_time)
                    with open(win_data_save_file, 'wb') as f:
                        for item_x, item_y in zip(win_data_x, win_data_y):
                            f.write("{} {}\n".format(item_x, item_y))
                    done_time = str(int(time.time()))
                    vis.text(vis_text_usage+'\n done at {} \n'.format(done_time), win=callback_text_usage_window)

    init_time = str(int(time.time()))
    if args.vis:
        vis = visdom.Visdom()
        vis_text_usage = 'Operating in the text window \n Press s to save data'

        callback_text_usage_window = vis.text(vis_text_usage)
        vis.register_event_handler(type_callback, callback_text_usage_window)

    # if args.dataset_path == '':
    #     HOME_PATH = os.path.expanduser('~')
    #     local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    # else:
    local_path = os.path.expanduser(args.dataset_path)
    if args.dataset == 'CamVid':
        dst = camvidLoader(local_path, is_transform=True, is_augment=args.data_augment)
    elif args.dataset == 'CityScapes':
        dst = cityscapesLoader(local_path, is_transform=True)
    else:
        pass

    # dst.n_classes = args.n_classes # 保证输入的class
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
        elif args.structure == 'fcn_resnet18_32s':
            model = fcn_resnet18(module_type='32s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn_resnet18_16s':
            model = fcn_resnet18(module_type='16s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn_resnet18_8s':
            model = fcn_resnet18(module_type='8s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn_resnet34_32s':
            model = fcn_resnet34(module_type='32s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn_resnet34_16s':
            model = fcn_resnet34(module_type='16s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn_resnet34_8s':
            model = fcn_resnet34(module_type='8s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn_MobileNet_32s':
            model = fcn_MobileNet(module_type='32s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn_MobileNet_16s':
            model = fcn_MobileNet(module_type='16s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'fcn_MobileNet_8s':
            model = fcn_MobileNet(module_type='8s', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'ResNetDUC':
            model = ResNetDUC(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'ResNetDUCHDC':
            model = ResNetDUCHDC(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'segnet':
            model = segnet(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'segnet_vgg19':
            model = segnet_vgg19(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'segnet_unet':
            model = segnet_unet(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'segnet_alignres':
            model = segnet_alignres(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'sqnet':
            model = sqnet(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'segnet_squeeze':
            model = segnet_squeeze(n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'ENet':
            model = ENet(n_classes=dst.n_classes)
        elif args.structure == 'ENetV2':
            model = ENetV2(n_classes=dst.n_classes)
        elif args.structure == 'drn_d_22':
            model = DRNSeg(model_name='drn_d_22', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'drn_a_50':
            model = DRNSeg(model_name='drn_a_50', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'drn_a_18':
            model = DRNSeg(model_name='drn_a_18', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'drn_e_22':
            model = DRNSeg(model_name='drn_e_22', n_classes=dst.n_classes, pretrained=args.init_vgg16)
        elif args.structure == 'erfnet':
            model = erfnet(n_classes=dst.n_classes)
        elif args.structure == 'fcdensenet103':
            model = fcdensenet103(n_classes=dst.n_classes)
        elif args.structure == 'fcdensenet56':
            model = fcdensenet56(n_classes=dst.n_classes)
        elif args.structure == 'fcdensenet_tiny':
            model = fcdensenet_tiny(n_classes=dst.n_classes)
        elif args.structure == 'Res_Deeplab_101':
            model = Res_Deeplab_101(n_classes=dst.n_classes, is_refine=False)
        elif args.structure == 'Res_Deeplab_50':
            model = Res_Deeplab_50(n_classes=dst.n_classes, is_refine=False)
        if args.resume_model_state_dict != '':
            try:
                # fcn32s、fcn16s和fcn8s模型略有增加参数，互相赋值重新训练过程中会有KeyError，暂时捕捉异常处理
                start_epoch_id1 = args.resume_model_state_dict.rfind('_')
                start_epoch_id2 = args.resume_model_state_dict.rfind('.')
                start_epoch = int(args.resume_model_state_dict[start_epoch_id1 + 1:start_epoch_id2])
                pretrained_dict = torch.load(args.resume_model_state_dict)
                # model_dict = model.state_dict()
                # for k, v in pretrained_dict.items():
                #     print(k)
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # model_dict.update(pretrained_dict)
                model.load_state_dict(pretrained_dict)
            except KeyError:
                print('missing key')



    if args.cuda:
        model.cuda()
    print('start_epoch:', start_epoch)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    data_count = int(dst.__len__() * 1.0 / args.batch_size)
    print('data_count:', data_count)
    for epoch in range(start_epoch+1, 20000, 1):
        loss_epoch = 0
        loss_avg_epoch = 0
        # data_count = 0
        # if args.vis:
        #     vis.text('epoch:{}'.format(epoch), win='epoch')
        for i, (imgs, labels) in enumerate(trainloader):

            # 最后的几张图片可能不到batch_size的数量，比如batch_size=4，可能只剩3张
            imgs_batch = imgs.shape[0]
            if imgs_batch != args.batch_size:
                break
            print(i)
            # data_count = i
            # print(labels.shape)
            # print(imgs.shape)

            imgs = Variable(imgs)
            labels = Variable(labels)

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            outputs = model(imgs)
            # print('type(outputs):', type(outputs))

            if args.vis and i%50==0:
                pred_labels = outputs.cpu().data.max(1)[1].numpy()
                # print(pred_labels.shape)
                label_color = dst.decode_segmap(labels.cpu().data.numpy()[0]).transpose(2, 0, 1)
                # print(label_color.shape)
                pred_label_color = dst.decode_segmap(pred_labels[0]).transpose(2, 0, 1)
                # print(pred_label_color.shape)
                win = 'label_color'
                vis.image(label_color, win=win)
                win = 'pred_label_color'
                vis.image(pred_label_color, win=win)

                # if epoch < 100:
                #     if not os.path.exists('/tmp/'+init_time):
                #         os.mkdir('/tmp/'+init_time)
                #     time_str = str(int(time.time()))
                #     print('label_color.transpose(2, 0, 1).shape:', label_color.transpose(1, 2, 0).shape)
                #     print('pred_label_color.transpose(2, 0, 1).shape:', pred_label_color.transpose(1, 2, 0).shape)
                #     cv2.imwrite('/tmp/'+init_time+'/'+time_str+'_label.png', label_color.transpose(1, 2, 0))
                #     cv2.imwrite('/tmp/'+init_time+'/'+time_str+'_pred_label.png', pred_label_color.transpose(1, 2, 0))


            # print(outputs.size())
            # print(labels.size())
            # 一次backward后如果不清零，梯度是累加的
            optimizer.zero_grad()

            loss = cross_entropy2d(outputs, labels)
            loss_np = loss.cpu().data.numpy()
            loss_epoch += loss_np
            print('loss:', loss_np)
            loss.backward()

            optimizer.step()

            # 显示一个周期的loss曲线
            if args.vis:
                win = 'loss_iteration'
                loss_np_expand = np.expand_dims(loss_np, axis=0)
                win_res = vis.line(X=np.ones(1)*(i+data_count*(epoch-1)+1), Y=loss_np_expand, win=win, update='append')
                if win_res != win:
                    vis.line(X=np.ones(1)*(i+1), Y=loss_np_expand, win=win)
                # if i+data_count*(epoch-1)==0:
                #     vis.register_event_handler(type_callback, win_res)
        # 关闭清空一个周期的loss，目标不清空
        # if args.vis:
        #     win = 'loss_iteration'
        #     vis.close(win)

        # 显示多个周期的loss曲线
        loss_avg_epoch = loss_epoch / (data_count * 1.0)
        # print(loss_avg_epoch)
        if args.vis:
            win = 'loss_epoch'
            loss_avg_epoch_expand = np.expand_dims(loss_avg_epoch, axis=0)
            win_res = vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win, update='append')
            if win_res != win:
                vis.line(X=np.ones(1)*epoch, Y=loss_avg_epoch_expand, win=win)

        if args.save_model and epoch%args.save_epoch==0:
            torch.save(model.state_dict(), '{}_camvid_class_{}_{}.pt'.format(args.structure, dst.n_classes, epoch))


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
    parser.add_argument('--dataset', type=str, default='CamVid', help='train dataset [ CamVid CityScapes ]')
    parser.add_argument('--dataset_path', type=str, default='~/Data/CamVid', help='train dataset path [ ~/Data/CamVid ~/Data/cityscapes ]')
    parser.add_argument('--data_augment', type=bool, default=False, help='enlarge the training data [ False ]')
    parser.add_argument('--batch_size', type=int, default=1, help='train dataset batch size [ 1 ]')
    # parser.add_argument('--n_classes', type=int, default=13, help='train class num [ 13 ]')
    parser.add_argument('--lr', type=float, default=1e-5, help='train learning rate [ 0.00001 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda [ False ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    print(args)
    train(args)
    # print('train----out----')
