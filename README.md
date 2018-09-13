# semseg

语义图像分割，为图像中的每个像素分配语义标签（例如“道路”，“天空”，“人”，“狗”）的任务使得能够实现许多新应用，例如Pixel 2和Pixel 2 XL智能手机的纵向模式中提供的合成浅景深效果和移动实时视频分割。
> 引用自[Semantic Image Segmentation with DeepLab in TensorFlow](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html)

---
## semantic segmentation algorithms

这个仓库旨在实现常用的语义分割算法，主要参考如下：
- [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)
- [Semantic Segmentation using Fully Convolutional Networks over the years](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets)
- [Fully Convolutional Networks for Semantic Segmentation](doc/fcn_understanding.md)
- [dataset_loaders](https://github.com/fvisin/dataset_loaders) 实现了主要的数据集的加载，包括视频，图像
- [semantic-segmentation-pytorch](https://github.com/hangzhaomit/semantic-segmentation-pytorch) ade20k数据集常用模型评估，其中具有caffe，pytorch和torch代码的实现
- [Weakly Supervised Instance Segmentation using Class Peak Response](https://arxiv.org/pdf/1804.00880.pdf) [代码PRM](https://github.com/ZhouYanzhao/PRM)
- [Learning to Segment Every Thing](https://arxiv.org/pdf/1711.10370.pdf) [代码seg_every_thing](https://github.com/ronghanghu/seg_every_thing)
- [Video Object Segmentation with Re-identification](https://arxiv.org/abs/1708.00197) [VS-ReID代码](https://github.com/lxx1991/VS-ReID)
- [segmentation-networks-benchmark](https://github.com/BloodAxe/segmentation-networks-benchmark) 评估了常用的语义分割网路的精度。

---
## 相关论文
- [Adversarial Learning for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/1802.07934) 代码[AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](http://liangchiehchen.com/projects/DeepLab.html) DeepLab v1，v2和v3系列论文，增加DeepLab系列论文总结，参考[deeplab实现理解](./doc/deeplab_understanding.md)
- [DenseASPP for Semantic Segmentation in Street Scenes](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790)，相应代码[LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax)。
- [Video Object Segmentation with Re-identification](https://arxiv.org/abs/1708.00197) [VS-ReID代码](https://github.com/lxx1991/VS-ReID)
- Unified Perceptual Parsing for Scene Understanding
- ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation，速度非常快，在TITAN上能达到112fps。

---
### 网络实现

- FCN，参考[fcn_understanding](doc/fcn_understanding.md)
- RefineNet，参考[refinenet_understanging](doc/refinenet_understanging.md)
- DUC，参考[duc_understanding](doc/duc_understanding.md)
- DRN
- PSPNet，参考[pspnet_understanding](doc/pspnet_understanding.md)
- ENet
- ErfNet
- LinkNet，参考[pytorch-linknet](https://github.com/e-lab/pytorch-linknet)
- FC-DenseNet，参考[fcdensenet_understanding](doc/fcdensenet_understanding.md)
- ...

---
### 数据集实现

- CamVid
- PASCAL VOC
- CityScapes
- ADE20K，参考[ade20k数据集相关](doc/ade20k_dataset.md)
- Mapillary Vistas Dataset，参考[Mapillary Vistas Dataset数据集相关](doc/mapillary_vistas_dataset.md)
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

