# -*- coding: utf-8 -*-

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
from semseg.modelloader.EDANet import EDANet
from semseg.modelloader.deeplabv3 import Res_Deeplab_101, Res_Deeplab_50
from semseg.modelloader.drn import drn_d_22, DRNSeg, drn_a_asymmetric_18, drnseg_a_50, drnseg_a_18, drnseg_e_22, \
    drnseg_a_asymmetric_18, drnseg_d_22, drnseg_a_asymmetric_n, drnseg_a_n
from semseg.modelloader.duc_hdc import ResNetDUC, ResNetDUCHDC
from semseg.modelloader.enet import ENet
from semseg.modelloader.enetv2 import ENetV2
from semseg.modelloader.erfnet import erfnet
from semseg.modelloader.fc_densenet import fcdensenet103, fcdensenet56, fcdensenet_tiny
from semseg.modelloader.fcn import fcn, fcn_32s, fcn_16s, fcn_8s
from semseg.modelloader.fcn_mobilenet import fcn_MobileNet, fcn_MobileNet_32s, fcn_MobileNet_16s, fcn_MobileNet_8s
from semseg.modelloader.fcn_resnet import fcn_resnet18, fcn_resnet34, fcn_resnet18_32s, fcn_resnet18_16s, \
    fcn_resnet18_8s, fcn_resnet34_32s, fcn_resnet34_16s, fcn_resnet34_8s
from semseg.modelloader.segnet import segnet, segnet_squeeze, segnet_alignres, segnet_vgg19
from semseg.modelloader.segnet_unet import segnet_unet
from semseg.modelloader.sqnet import sqnet
from semseg.utils.flops_benchmark import add_flops_counting_methods




def performance_table(args):
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
        if args.structure == 'drnseg_a_asymmetric_n':
            model = eval(args.structure)(n_classes=dst.n_classes, pretrained=args.init_vgg16, depth_n=args.depth_n)
        elif args.structure == 'drnseg_a_n':
            model = eval(args.structure)(n_classes=dst.n_classes, pretrained=args.init_vgg16, depth_n=args.depth_n)
        else:
            model = eval(args.structure)(n_classes=dst.n_classes, pretrained=args.init_vgg16)
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

    model = add_flops_counting_methods(model)
    if args.cuda:
        model.cuda()
    model.train()
    model.start_flops_count()
    # print('start_epoch:', start_epoch)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.99, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    forward_time = 0
    backward_time = 0

    # 第一次warmup将GPU调用
    for i, (imgs, labels) in enumerate(trainloader):
        imgs_batch = imgs.shape[0]
        if imgs_batch != args.batch_size:
            break
        if args.cuda:
            imgs = imgs.cuda()
        model(imgs)
        break

    for epoch in range(0, 1, 1):
        for i, (imgs, labels) in enumerate(trainloader):

            # 最后的几张图片可能不到batch_size的数量，比如batch_size=4，可能只剩3张
            imgs_batch = imgs.shape[0]
            if imgs_batch != args.batch_size:
                break
            # print(i)
            # data_count = i
            # print(labels.shape)
            # print(imgs.shape)

            imgs = Variable(imgs)
            labels = Variable(labels)

            # imgs = Variable(torch.randn(1, 3, 360, 640))
            # labels = Variable(torch.LongTensor(np.ones((1, 360, 640), dtype=np.int)))

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            if args.cuda:
                torch.cuda.synchronize()
            start = time.time()
            outputs = model(imgs)
            if args.cuda:
                torch.cuda.synchronize()
            end = time.time()
            forward_time += (end - start)
            # print('forward time:', end - start)

            if args.cuda:
                torch.cuda.synchronize()
            start = time.time()
            # 一次backward后如果不清零，梯度是累加的
            optimizer.zero_grad()
            loss = cross_entropy2d(outputs, labels)
            loss.backward()
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()
            end = time.time()
            backward_time += (end - start)
            # print('backward time:', end - start)

            if i==args.iterations-1:
                break

    avg_forward_time = forward_time * 1.0 / args.iterations
    avg_backward_time = backward_time * 1.0 / args.iterations

    print('average forward time:', forward_time * 1.0 / args.iterations)
    print('average backward time:', backward_time * 1.0 / args.iterations)
    model_flops = model.compute_average_flops_cost() / 1e9 / 2
    print('model_flops:', model_flops)
    if args.save_model:
        torch.save(model.state_dict(), 'performance_{}_{}_class_{}.pt'.format(args.structure, args.dataset, args.n_classes))

    return avg_forward_time, avg_backward_time, model_flops


# best training: python performance_table.py --resume_model fcn32s_camvid_9.pkl --save_model True
# --init_vgg16 True --dataset_path /home/cgf/Data/CamVid --batch_size 1 --vis True
if __name__=='__main__':
    # print('train----in----')
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='ALL', help='use the net structure to segment [ fcn32s ResNetDUC segnet ENet drn_d_22 ]')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path [ fcn32s_camvid_9.pkl ]')
    parser.add_argument('--resume_model_state_dict', type=str, default='', help='resume model state dict path [ fcn32s_camvid_9.pt ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--save_epoch', type=int, default=1, help='save model after epoch [ 1 ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset', type=str, default='CamVid', help='train dataset [ CamVid CityScapes ]')
    parser.add_argument('--dataset_path', type=str, default='~/Data/CamVid', help='train dataset path [ ~/Data/CamVid ~/Data/cityscapes ]')
    parser.add_argument('--data_augment', type=bool, default=False, help='enlarge the training data [ False ]')
    parser.add_argument('--batch_size', type=int, default=1, help='train dataset batch size [ 1 ]')
    parser.add_argument('--iterations', type=int, default=1, help='train dataset iterations [ 1 ]')
    parser.add_argument('--n_classes', type=int, default=12, help='train class num [ 12 ]')
    parser.add_argument('--depth_n', type=int, default=18, help='just for testing  drnseg_a_asymmetric_n [ 18 ]')
    parser.add_argument('--lr', type=float, default=1e-5, help='train learning rate [ 0.00001 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda [ False ]')
    args = parser.parse_args()
    # print(args.resume_model)
    # print(args.save_model)
    structures = [
        'fcn_32s', 'fcn_16s', 'fcn_8s',
        'fcn_resnet18_32s', 'fcn_resnet18_16s', 'fcn_resnet18_8s',
        'fcn_resnet34_32s', 'fcn_resnet34_16s', 'fcn_resnet34_8s',
        'fcn_MobileNet_32s', 'fcn_MobileNet_16s', 'fcn_MobileNet_8s',
        'ResNetDUC', 'ResNetDUCHDC',
        'segnet', 'segnet_vgg19', 'segnet_unet', 'segnet_alignres',
        # 'sqnet',
        'segnet_squeeze',
        'ENet', 'ENetV2',
        'drnseg_d_22', 'drnseg_a_50', 'drnseg_a_18', 'drnseg_e_22', 'drnseg_a_asymmetric_18',
        'erfnet',
        # 'fcdensenet103', 'fcdensenet56', 'fcdensenet_tiny',
        'Res_Deeplab_101', 'Res_Deeplab_50',
        'EDANet'
    ]
    if args.structure == 'ALL':
        for structure in structures:
            print('-----------------------------------------------------------------------------')
            args.structure = structure
            print(args)
            print('structure:', args.structure)
            performance_table(args)
            print('-----------------------------------------------------------------------------')
    else:
        performance_table(args)

    # args.structure = 'drnseg_a_asymmetric_n'
    # args.structure = 'drnseg_a_n'
    # args.cuda = True
    #
    # for structure in ['drnseg_a_n', 'drnseg_a_asymmetric_n']:
    #     for cuda in [True, False]:
    #         if structure=='drnseg_a_asymmetric_n' and cuda==True:
    #             pass
    #         else:
    #             continue
    #         args.structure = structure
    #         args.cuda = cuda
    #         tmp_txt = open('/tmp/performance_speed_{}_gpu_{}.txt'.format(args.structure, args.cuda), 'wb')
    #         tmp_txt.write('depth_n avg_fps avg_forward_time avg_backward_time model_flops\n')
    #         try:
    #             for depth_n in range(18, 100, 1):
    #                 args.depth_n = depth_n
    #                 args.iterations = 10
    #                 avg_forward_time, avg_backward_time, model_flops = performance_table(args)
    #                 if args.cuda:
    #                     torch.cuda.empty_cache()
    #                 print('avg_forward_time:', avg_forward_time)
    #                 # if avg_forward_time > 1/25.0:
    #                 #     break
    #                 tmp_txt.write('{}\t{}\t{}\t{}\t{}\n'.format(depth_n, 1.0 / avg_forward_time, avg_forward_time,
    #                                                             avg_backward_time, model_flops))
    #         except:
    #             pass
    #         finally:
    #             tmp_txt.close()


    # print('train----out----')

