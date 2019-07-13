# -*- coding: utf-8 -*-
import random
import collections
import logging as log
import torch
import numpy as np
from PIL import Image, ImageOps
import time
import cv2
import matplotlib.pyplot as plt


def randomCropLetterboxPil(img):
    output_w, output_h = (1408, 768)
    jitter = 0.3
    fill_color = 127

    orig_w, orig_h = img.size
    img_np = np.array(img)
    channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
    dw = int(jitter * orig_w)
    dh = int(jitter * orig_h)
    new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
    scale = random.random() * (2 - 0.25) + 0.25
    if new_ar < 1:
        nh = int(scale * orig_h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * orig_w)
        nh = int(nw / new_ar)

    if output_w > nw:
        dx = random.randint(0, output_w - nw)
    else:
        dx = random.randint(output_w - nw, 0)

    if output_h > nh:
        dy = random.randint(0, output_h - nh)
    else:
        dy = random.randint(output_h - nh, 0)

    nxmin = max(0, -dx)
    nymin = max(0, -dy)
    nxmax = min(nw, -dx + output_w - 1)
    nymax = min(nh, -dy + output_h - 1)
    sx, sy = float(orig_w) / nw, float(orig_h) / nh
    orig_xmin = int(nxmin * sx)
    orig_ymin = int(nymin * sy)
    orig_xmax = int(nxmax * sx)
    orig_ymax = int(nymax * sy)
    orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
    orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
    output_img = Image.new(img.mode, (output_w, output_h), color=(fill_color,) * channels)
    output_img.paste(orig_crop_resize, (0, 0))
    return output_img


def randomCropLetterboxCv(img):
    output_w, output_h = (1408, 768)
    jitter = 0.3
    fill_color = 127

    # orig_w, orig_h = img.size
    orig_h, orig_w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) > 2 else 1
    dw = int(jitter * orig_w)
    dh = int(jitter * orig_h)
    new_ar = float(orig_w + random.randint(-dw, dw)) / (orig_h + random.randint(-dh, dh))
    scale = random.random() * (2 - 0.25) + 0.25
    if new_ar < 1:
        nh = int(scale * orig_h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * orig_w)
        nh = int(nw / new_ar)

    if output_w > nw:
        dx = random.randint(0, output_w - nw)
    else:
        dx = random.randint(output_w - nw, 0)

    if output_h > nh:
        dy = random.randint(0, output_h - nh)
    else:
        dy = random.randint(output_h - nh, 0)

    nxmin = max(0, -dx)
    nymin = max(0, -dy)
    nxmax = min(nw, -dx + output_w - 1)
    nymax = min(nh, -dy + output_h - 1)
    sx, sy = float(orig_w) / nw, float(orig_h) / nh
    orig_xmin = int(nxmin * sx)
    orig_ymin = int(nymin * sy)
    orig_xmax = int(nxmax * sx)
    orig_ymax = int(nymax * sy)
    # orig_crop = img.crop((orig_xmin, orig_ymin, orig_xmax, orig_ymax))
    orig_crop = img[orig_ymin:orig_ymax, orig_xmin:orig_xmax, :]
    orig_crop_resize = cv2.resize(orig_crop, (nxmax - nxmin, nymax - nymin), interpolation=cv2.INTER_CUBIC)
    output_img = np.ones((output_h, output_w, channels), dtype=np.uint8) * fill_color

    y_lim = int(min(output_img.shape[0], orig_crop_resize.shape[0]))
    x_lim = int(min(output_img.shape[1], orig_crop_resize.shape[1]))
    output_img[:y_lim, :x_lim, :] = orig_crop_resize[:y_lim, :x_lim, :]
    # orig_crop_resize = orig_crop.resize((nxmax - nxmin, nymax - nymin))
    # output_img = Image.new(img.mode, (output_w, output_h), color=(fill_color,) * channels)
    # output_img.paste(orig_crop_resize, (0, 0))
    return output_img


if __name__ == '__main__':
    # test_num = 10
    # img = Image.open('../data/0006R0_f00930.png')
    # start_time = time.time()
    # for _ in range(test_num):
    #     out_img = randomCropLetterboxPil(img)
    #     plt.imshow(out_img)
    #     plt.show()
    # end_time = time.time()
    # print('cost time:', (end_time-start_time)/test_num)

    test_num = 1000
    img = cv2.imread('../data/0006R0_f00930.png')
    # img = cv2.resize(img, (1024, 720), interpolation=cv2.INTER_CUBIC)
    print('img.shape:', img.shape)
    start_time = time.time()
    for _ in range(test_num):
        out_img = randomCropLetterboxCv(img)
        cv2.imshow('img', out_img)
        cv2.waitKey(0)
    end_time = time.time()
    print('cost time:', (end_time-start_time)/test_num)
