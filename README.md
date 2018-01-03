# semseg

---
## semantic segmentation algorithms

这个仓库旨在实现常用的语义分割算法，主要参考如下：
- [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)
- [Semantic Segmentation using Fully Convolutional Networks over the years](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets)
- [Fully Convolutional Networks for Semantic Segmentation](doc/fcn_understanding.md)
- [dataset_loaders](https://github.com/fvisin/dataset_loaders) 实现了主要的数据集的加载，包括视频，图像
- [semantic-segmentation-pytorch](https://github.com/hangzhaomit/semantic-segmentation-pytorch) ade20k数据集常用模型评估，其中具有caffe，pytorch和torch代码的实现

---
### 网络实现

- FCN，参考[fcn_understanding](doc/fcn_understanding.md)
- RefineNet，参考[refinenet_understanging](doc/refinenet_understanging.md)
- DUC，参考[duc_understanding](doc/duc_understanding.md)
- DRN
- PSPNet，参考[pspnet_understanding](doc/pspnet_understanding.md)
- ENet
- EefNet
- ...

---
### 数据集实现

- CamVid
- PASCAL VOC
- CityScapes
- ADE20K，参考[ade20k数据集相关](doc/ade20k_dataset.md)
- ...

---
### 数据集增加

通过仿射变换来实现数据集增加的方法扩充语义分割数据集。

- [imgaug](https://github.com/aleju/imgaug)
- [Augmentor](https://github.com/mdbloice/Augmentor)
- [joint_transforms.py](https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py) 使用这个脚本实现数据集增加

---
### 依赖

- pytorch
- ...

---
### 数据

- CamVid，参考[camvid_dataset](doc/camvid_dataset.md)
- PASCAL VOC，参考[pascal_voc_dataset](doc/pascal_voc_dataset.md)
- CityScapes，参考[cityscapes_dataset](doc/cityscapes.md)
- ...

---
### 用法

**可视化**

[visdom](https://github.com/facebookresearch/visdom)
[开发相关问题](doc/visdom_problem.md)

```bash
# 在tmux或者另一个终端中开启可视化服务器visdom
python -m visdom.server
# 然后在浏览器中查看127.0.0.1:9097
```

**训练**
```bash
# 训练模型
python train.py
```

**校验**
```bash
# 校验模型
python validate.py
```

**测试**
```bash
# 测试模型
python test.py
```

