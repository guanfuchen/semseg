import torch
import os

import numpy as np
from torch.autograd import Variable

from semseg.dataloader.camvid_loader import camvidLoader
from semseg.loss import cross_entropy2d
from semseg.modelloader.fcn import fcn32s


def train():
    HOME_PATH = os.path.expanduser('~')
    local_path = os.path.join(HOME_PATH, 'Data/CamVid')
    dst = camvidLoader(local_path, is_transform=True)
    trainloader = torch.utils.data.DataLoader(dst, batch_size=1)
    model = fcn32s(n_classes=dst.n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.99, weight_decay=5e-4)
    for epoch in range(20000):
        for i, (imgs, labels) in enumerate(trainloader):
            print(i)
            #  print(labels.shape)
            #  print(imgs.shape)
            labels = labels[np.newaxis, :]
            imgs = Variable(imgs)
            labels = Variable(labels)
            pred = model(imgs)
            optimizer.zero_grad()

            loss = cross_entropy2d(pred, labels)
            print('loss:', loss)
            loss.backward()

            optimizer.step()
        torch.save(model, 'fcn32s_camvid_{}.pkl'.format(epoch))



if __name__=='__main__':
    print('train----in----')
    train()
    print('train----out----')
