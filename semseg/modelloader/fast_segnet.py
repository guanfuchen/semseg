# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

# class fast_segnet(nn.Module):
#     def __init__(self):
#         super(fast_segnet, self).__init__()
#
#     def forward(self, x):
#         pass