# -*- coding: utf-8 -*-
import glob
import os

if __name__ == '__main__':

    root = os.path.expanduser('~/Data/CamVid')
    split = 'train'
    file_list = glob.glob(root + '/' + split + 'bbox/*.xml')
    file_list.sort()
    # print(file_list)

    img_file_count = 6690
    for bbox_path_name in file_list:
        img_file_name = bbox_path_name[bbox_path_name.rfind('/')+1:bbox_path_name.rfind('.')]
        if '001TP' in img_file_name:
            img_file_new_name = '0001TP_{:06d}'.format(img_file_count)
            img_file_count += 30
            print(bbox_path_name)
            print(img_file_name)
            print(img_file_new_name)
            bbox_path_new_name = root + '/' + split + 'bbox/{}.xml'.format(img_file_new_name)
            print(bbox_path_new_name)
            os.rename(bbox_path_name, bbox_path_new_name)

