# semseg

语义图像分割，为图像中的每个像素分配语义标签（例如“道路”，“天空”，“人”，“狗”）的任务使得能够实现许多新应用，例如Pixel 2和Pixel 2 XL智能手机的纵向模式中提供的合成浅景深效果和移动实时视频分割。
> 引用自[Semantic Image Segmentation with DeepLab in TensorFlow](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html)

---
## semantic segmentation algorithms

这个仓库旨在实现常用的语义分割算法，主要参考如下：
- [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)
- [awesome-semantic-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)，语义分割的awesome系列。
- [Semantic Segmentation using Fully Convolutional Networks over the years](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets)
- [Fully Convolutional Networks for Semantic Segmentation](doc/fcn_understanding.md)
- [dataset_loaders](https://github.com/fvisin/dataset_loaders) 实现了主要的数据集的加载，包括视频，图像
- [semantic-segmentation-pytorch](https://github.com/hangzhaomit/semantic-segmentation-pytorch) ade20k数据集常用模型评估，其中具有caffe，pytorch和torch代码的实现
- [Weakly Supervised Instance Segmentation using Class Peak Response](https://arxiv.org/pdf/1804.00880.pdf) [代码PRM](https://github.com/ZhouYanzhao/PRM)
- [Learning to Segment Every Thing](https://arxiv.org/pdf/1711.10370.pdf) [代码seg_every_thing](https://github.com/ronghanghu/seg_every_thing)
- [Video Object Segmentation with Re-identification](https://arxiv.org/abs/1708.00197) [VS-ReID代码](https://github.com/lxx1991/VS-ReID)
- [segmentation-networks-benchmark](https://github.com/BloodAxe/segmentation-networks-benchmark) 评估了常用的语义分割网路的精度。
- [SemanticSegmentation_DL](https://github.com/tangzhenyu/SemanticSegmentation_DL)，该仓库整理了常用的DL语义分割论文资料。
- [2015-10-09-segmentation.md](https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-segmentation.md) handong对于语义分割的论文收集总结，参考较多。

---
## 相关论文
- [Adversarial Learning for Semi-supervised Semantic Segmentation](https://arxiv.org/abs/1802.07934) 代码[AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](http://liangchiehchen.com/projects/DeepLab.html) DeepLab v1，v2和v3系列论文，增加DeepLab系列论文总结，参考[deeplab实现理解](./doc/deeplab_understanding.md)
- [DenseASPP for Semantic Segmentation in Street Scenes](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790)，相应代码[LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax)。
- [Video Object Segmentation with Re-identification](https://arxiv.org/abs/1708.00197) [VS-ReID代码](https://github.com/lxx1991/VS-ReID)
- Unified Perceptual Parsing for Scene Understanding
- ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation，速度非常快，在TITAN上能达到112fps。
- OCNet: Object Context Network for Scene Parsing
- PSANet: Point-wise Spatial Attention Network for Scene Parsing
- Recent progress in semantic image segmentation，简单地综述性文章，可以作为科普文章查看，意义不太大。
- Not All Pixels Are Equal: Difficulty-Aware Semantic Segmentation via Deep Layer Cascade
- Wider or Deeper: Revisiting the ResNet Model for Visual Recognition
- High-performance Semantic Segmentation Using Very Deep Fully Convolutional Networks
- Dilated Residual Networks
- Fully Convolutional Instance-aware Semantic Segmentation
- Light-Weight RefineNet for Real-Time Semantic Segmentation
- Dark Model Adaptation: Semantic Image Segmentation from Daytime to Nighttime，从白天到黑夜自适应的语义图像分割。
- Label Refinement Network for Coarse-to-Fine Semantic Segmentation
- Smoothed Dilated Convolutions for Improved Dense Prediction，非常好的一篇对于空洞卷积理解的论文，值得阅读，这篇文章将空洞卷积理解为3个分解操作，然后通过建立不同group之间的联系去除网格伪影进行。
- Characterizing Adversarial Examples Based on Spatial Consistency Information for Semantic Segmentation
- UOLO - automatic object detection and segmentation in biomedical images，联合目标检测和语义分割的框架，可以参考借鉴。
- 弱监督语义分割，参考如下。

---
## 弱监督语义分割

- Generating Self-Guided Dense Annotations for Weakly Supervised Semantic Segmentation

---
## 实例分割

目前暂且收集相关实例分割到语义分割目录中，待综述完成单独分离。

- Semantic Instance Segmentation with a Discriminative Loss Function

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

以下是相关语义分割论文粗略整理。

---
## ShuffleSeg: Real-time Semantic Segmentation Network

| 摘要 |
| ---- |
| **Real-time semantic segmentation** is of significant importance for mobile and robotics related applications. We propose a computationally efficient segmentation network which we term as **ShuffleSeg**. **The proposed architecture is based on grouped convolution and channel shuffling in its encoder for improving the performance.** An ablation study of different decoding methods is compared including Skip architecture, UNet, and Dilation Frontend. Interesting insights on the speed and accuracy tradeoff is discussed. It is shown that skip architecture in the decoding method provides the best compromise for the goal of real-time performance, while it provides adequate accuracy by utilizing higher resolution feature maps for a more accurate segmentation. ShuffleSeg is evaluated on CityScapes and compared against the state of the art real-time segmentation networks. It achieves 2x GFLOPs reduction, while it provides on par mean intersection over union of **58.3%** on CityScapes test set. ShuffleSeg runs at **15.7 frames** per second on NVIDIA Jetson TX2, which makes it of great potential for real-time applications. |

| 会议／期刊 | 作者 | 论文 | 代码 |
| ---- | ---- | ---- | ---- |
| arXiv: 1803.03816 | Mostafa Gamal, **Mennatullah Siam**, Moemen Abdel-Razek | ShuffleSeg: Real-time Semantic Segmentation Network | [TFSegmentation](https://github.com/MSiam/TFSegmentation) |

本文提出了一种基于ShuffleNet的实时语义分割网络，通过在编码器中使用grouped convolution和channle shuffling（ShuffleNet基本结构），同时用不同的解码方法，包括Skip架构，UNet和Dilation前端探索了精度和速度的平衡。

主要动机是：
> It was shown in [4][2][3] that depthwise separable convolution or grouped convolution reduce the computational cost, while maintaining good representation capability.

训练的trciks：充分利用CityScapes数据集，将其中粗略标注的图像作为网络预训练，然后基于精细标注的图像作为网络微调。

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/shuffle_seg_1.png)

---
## RTSeg: Real-time Semantic Segmentation Comparative Study

| 摘要 |
| ---- |
| Semantic segmentation benefits robotics related applications especially autonomous driving. Most of the research on semantic segmentation is only on increasing the accuracy of segmentation models with little attention to computationally efficient solutions. **The few work conducted in this direction does not provide principled methods to evaluate the different design choices for segmentation.** In this paper, we address this gap by presenting a real-time semantic segmentation benchmarking framework with a decoupled design for feature extraction and decoding methods. **The framework is comprised of different network architectures for feature extraction such as VGG16, Resnet18, MobileNet, and ShuffleNet.** It is also comprised of multiple meta-architectures for segmentation that define the decoding methodology. These include SkipNet, UNet, and Dilation Frontend. Experimental results are presented on the Cityscapes dataset for urban scenes. The modular design allows novel architectures to emerge, that lead to 143x GFLOPs reduction in comparison to SegNet. |

| 会议／期刊 | 作者 | 论文 | 代码 |
| ---- | ---- | ---- | ---- |
| arXiv: 1803.02758 | **Mennatullah Siam**, Mostafa Gamal, Moemen Abdel-Razek, Senthil Yogamani, Martin Jagersand | RTSeg: Real-time Semantic Segmentation Comparative Study | [TFSegmentation](https://github.com/MSiam/TFSegmentation) |

和ShuffleSeg: Real-time Semantic Segmentation Network同一作者。

本文整体思路和ShuffleSeg类同，只不过更加抽象了编码器解码器，这里的编码器不再仅仅是ShuffleNet，而是增加了VGG16，Resnet18，MobileNet，方便了后期不同基础网络性能的比较。

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/rtseg_1.png)

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/rtseg_2.png)