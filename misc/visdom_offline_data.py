# -*- coding: utf-8 -*-

import numpy as np
import visdom
import json
import time

init_time = str(int(time.time()))
vis = visdom.Visdom()
win = 'det_loss_iteration'
win_data = vis.get_window_data(win)
win_data_dict = json.loads(win_data)
win_data_content_dict = win_data_dict['content']
win_data_x = np.array(win_data_content_dict['data'][0]['x'])
win_data_y = np.array(win_data_content_dict['data'][0]['y'])

win_data_save_file = '/tmp/loss_iteration_{}.txt'.format(init_time)
with open(win_data_save_file, 'wb') as f:
    for item_x, item_y in zip(win_data_x, win_data_y):
        f.write("{}\t{}\n".format(item_x, item_y))
done_time = str(int(time.time()))
