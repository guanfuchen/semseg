# -*- coding: utf-8 -*-
import json
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import glob
import numpy as np
from scipy import misc

if __name__ == '__main__':

    # img = cv2.imread('../data/0006R0_f00930.png')
    # bbox_path = '../data/0006R0_f00930.xml'

    root = os.path.expanduser('~/Data/cityscapes')
    split = 'train'
    images_base = os.path.join(root, "leftImg8bit", split)
    annotations_base = os.path.join(root, "gtFine", split)
    print('root:', root)
    print('images_base:', images_base)
    print('annotations_base:', annotations_base)
    file_list = glob.glob(images_base + "/*/*.png")
    file_list.sort()
    # print(file_list)

    save_obj_fp = open('/tmp/cityscapes_det.txt', 'wb')
    right_obj_names = ['car', 'person']
    for img_index, img_name in enumerate(file_list):
        # img_path = file_list[split][img_index].rstrip()
        # img = misc.imread(img_name)
        img = cv2.imread(img_name)
        img = np.array(img, dtype=np.uint8)
        polygons_path = os.path.join(
            annotations_base,
            img_name.split(os.sep)[-2],
            os.path.basename(img_name)[:-15] + "gtFine_polygons.json",
        )
        polygons_json = json.load(open(polygons_path, 'rb'))
        object_det_bboxes = []
        # print(polygons_json['objects'])
        polygons_json_objects = polygons_json['objects']
        for polygons_json_object in polygons_json_objects:
            polygons_json_object_polygon = polygons_json_object['polygon']
            polygons_json_object_label = polygons_json_object['label']
            if polygons_json_object_label in right_obj_names:
                # print(polygons_json_object_polygon)
                polygons_json_object_polygon_np = np.array(polygons_json_object_polygon)
                # print(polygons_json_object_polygon_np)
                # print(polygons_json_object_polygon_np.shape)
                polygons_json_object_polygon_np_x = polygons_json_object_polygon_np[:, 0]
                polygons_json_object_polygon_np_y = polygons_json_object_polygon_np[:, 1]
                x1 = min(polygons_json_object_polygon_np_x)
                x2 = max(polygons_json_object_polygon_np_x)
                y1 = min(polygons_json_object_polygon_np_y)
                y2 = max(polygons_json_object_polygon_np_y)
                object_cls = right_obj_names.index(polygons_json_object_label)
                object_det_bboxes.append([x1, y1, x2, y2, object_cls])

        if object_det_bboxes:
            save_obj_line = img_name[img_name.index(split)+len(split)+1:]
            for object_det_bbox in object_det_bboxes:
                import cv2

                x1 = object_det_bbox[0]
                y1 = object_det_bbox[1]
                x2 = object_det_bbox[2]
                y2 = object_det_bbox[3]
                bbox_obj_name_cls = object_det_bbox[4]
                cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=5)
                save_obj_line += ' {} {} {} {} {}'.format(x1, y1, x2, y2, bbox_obj_name_cls)
            save_obj_line += '\n'
            save_obj_fp.write(save_obj_line)

        # cv2.imshow('img', img)
        # cv2.waitKey()
    save_obj_fp.close()
