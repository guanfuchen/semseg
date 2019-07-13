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


def randomFlipPil(img):
    """ Randomly flip image """
    flip = random.random() < 0.5
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def randomFlipCv(img):
    """ Randomly flip image """
    flip = random.random() < 0.5
    if flip:
        img = cv2.flip(img, 1)
    return img


def hsvShiftPil(img):
    hue = 0.1
    sat = 1.5
    val = 1.5

    dh = random.uniform(-hue, hue)
    ds = random.uniform(1, sat)
    if random.random() < 0.5:
        ds = 1 / ds
    dv = random.uniform(1, val)
    if random.random() < 0.5:
        dv = 1 / dv

    img = img.convert('HSV')
    channels = list(img.split())

    def change_hue(x):
        x += int(dh * 255)
        if x > 255:
            x -= 255
        elif x < 0:
            x += 0
        return x

    channels[0] = channels[0].point(change_hue)
    channels[1] = channels[1].point(lambda i: min(255, max(0, int(i*ds))))
    channels[2] = channels[2].point(lambda i: min(255, max(0, int(i*dv))))

    img = Image.merge(img.mode, tuple(channels))
    img = img.convert('RGB')
    return img


def hsvShiftCv(img):
    hue = 0.1
    sat = 1.5
    val = 1.5

    dh = random.uniform(-hue, hue)
    ds = random.uniform(1, sat)
    if random.random() < 0.5:
        ds = 1 / ds
    dv = random.uniform(1, val)
    if random.random() < 0.5:
        dv = 1 / dv

    img = img.astype(np.float32) / 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    def wrap_hue(x):
        x[x >= 360.0] -= 360.0
        x[x < 0.0] += 360.0
        return x

    img[:, :, 0] = wrap_hue(img[:, :, 0] + (360.0 * dh))
    img[:, :, 1] = np.clip(ds * img[:, :, 1], 0.0, 1.0)
    img[:, :, 2] = np.clip(dv * img[:, :, 2], 0.0, 1.0)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = (img * 255).astype(np.uint8)
    return img


if __name__ == '__main__':
    test_num = 100
    img = Image.open('../data/0006R0_f00930.png')
    img = img.resize((1024, 720))
    out_img = img
    start_time = time.time()
    for _ in range(test_num):
        out_img = randomCropLetterboxPil(out_img)
        out_img = randomFlipPil(out_img)
        out_img = hsvShiftPil(out_img)
        # plt.imshow(out_img)
        # plt.show()
    end_time = time.time()
    print('cost time:', (end_time-start_time)/test_num)

    test_num = 100
    img = cv2.imread('../data/0006R0_f00930.png')
    img = cv2.resize(img, (1024, 720), interpolation=cv2.INTER_CUBIC)
    out_img = img
    # print('img.shape:', img.shape)
    start_time = time.time()
    for _ in range(test_num):
        out_img = randomCropLetterboxCv(out_img)
        out_img = randomFlipCv(out_img)
        out_img = hsvShiftCv(out_img)
        # cv2.imshow('img', out_img)
        # cv2.waitKey(0)
    end_time = time.time()
    print('cost time:', (end_time-start_time)/test_num)
