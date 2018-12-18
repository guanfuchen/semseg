# semseg

语义图像分割，为图像中的每个像素分配语义标签（例如“道路”，“天空”，“人”，“狗”）的任务使得能够实现许多新应用，例如Pixel 2和Pixel 2 XL智能手机的纵向模式中提供的合成浅景深效果和移动实时视频分割。
> 引用自[Semantic Image Segmentation with DeepLab in TensorFlow](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html)

下面将近期主要的论文整理表格以供后面进一步总结。

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/semantic_survey_total.png)

---
### 网络实现

- FCN（VGG和ResNet的骨干网络），已实现，参考[fcn_understanding](doc/fcn_understanding.md)
- RefineNet，已实现，参考[refinenet_understanging](doc/refinenet_understanging.md)
- DUC，参考[duc_understanding](doc/duc_understanding.md)
- DRN，已实现
- PSPNet，参考[pspnet_understanding](doc/pspnet_understanding.md)
- ENet，已实现
- ErfNet，已实现
- EDANet，已实现
- LinkNet，已实现，参考[pytorch-linknet](https://github.com/e-lab/pytorch-linknet)
- FC-DenseNet，已实现，参考[fcdensenet_understanding](doc/fcdensenet_understanding.md)
- LRN，已实现，但是没有增加多分辨率loss训练，后期增加。
- BiSeNet，已实现，主要是ResNet-18和ResNet-101，其余类似。
- FRRN，已实现，FRRN A和FRRN B。
- 增加YOLO-V1多任务学习，还未完全测试。
- ...
- 
---
## semantic segmentation algorithms

这个仓库旨在实现常用的语义分割算法，主要参考如下：
- [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)
- [Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)，和pytorch对应的tensorflow语义分割框架实现。
- [awesome-semantic-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)，语义分割的awesome系列。
- [segmentation_models](https://github.com/qubvel/segmentation_models)，作者引入了不同backbones的不同分割模型组合，包括VGG16，VGG19，ResNet18，ResNet34，ResNet50，ResNet101，ResNet152和ResNeXt系列，DenseNet和Inception变种同UNet、LinkNet和PSPNet等结合，**进一步开发将包含多种的backbones**。
- [SemanticSegPaperCollection](https://github.com/shawnyuen/SemanticSegPaperCollection)
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
- [SemanticSegPaperCollection](https://github.com/shawnyuen/SemanticSegPaperCollection)，相关论文收集。
- [real-time-network](https://github.com/wpf535236337/real-time-network)，包含了相关实时语义分割的论文。

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
- CGNet: A Light-weight Context Guided Network for Semantic Segmentation

---
## 弱监督语义分割

- Generating Self-Guided Dense Annotations for Weakly Supervised Semantic Segmentation

---
## 实例分割

目前暂且收集相关实例分割到语义分割目录中，待综述完成单独分离。

- Semantic Instance Segmentation with a Discriminative Loss Function

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

- 可视化
[visdom](https://github.com/facebookresearch/visdom)
[开发相关问题](doc/visdom_problem.md)
```bash
# 在tmux或者另一个终端中开启可视化服务器visdom
python -m visdom.server
# 然后在浏览器中查看127.0.0.1:9097
```
- 训练
```bash
# 训练模型
python train.py
```
- 校验
```bash
# 校验模型
python validate.py
```

![ENet可视化结果](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/enet_data_11_1.png)





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

---
## SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling

| 摘要 |
| ---- |
| We propose a novel deep architecture, SegNet, for semantic pixel wise image labelling. SegNet has several attractive properties; (i) it only requires forward evaluation of a fully learnt function to obtain smooth label predictions, (ii) with increasing depth, a larger context is considered for pixel labelling which improves accuracy, and (iii) it is easy to visualise the effect of feature activation(s) in the pixel label space at any depth.
SegNet is composed of a stack of encoders followed by a corresponding decoder stack which feeds into a soft-max classification layer. The decoders help map low resolution feature maps at the output of the encoder stack to full input image size feature maps. This addresses an important drawback of recent deep learning approaches which have adopted networks designed for object categorization for pixel wise labelling. **These methods lack a mechanism to map deep layer feature maps to input dimensions.** They resort to **ad hoc** methods to upsample features, e.g. by replication. This results in noisy predictions and also restricts the number of pooling layers in order to avoid too much upsampling and thus reduces spatial context. SegNet overcomes these problems by learning to map encoder outputs to image pixel labels. We test the performance of SegNet on outdoor RGB scenes from CamVid, KITTI and indoor scenes from the NYU dataset. Our results show that SegNet achieves state-of-the-art performance even without use of additional cues such as depth, video frames or post-processing with CRF models. |

| 会议／期刊 | 作者 | 论文 | 代码 |
| ---- | ---- | ---- | ---- |
| arXiv: 1505.07293 | Vijay Badrinarayanan, Ankur Handa, Roberto Cipolla | SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling | [caffe-segnet](https://github.com/alexgkendall/caffe-segnet) |

本文为SegNet-Basic，基本思路就是编码器-解码器架构，指出当前语义分割方法都缺少一个机制将深度特征图map到输入维度的机制，基本都是特定的上采样特征方法，比如复制。

---
## Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding

| 摘要 |
| ---- |
| We present a deep learning framework for **probabilistic pixel-wise semantic segmentation**, which we term **Bayesian SegNet**. Semantic segmentation is an important tool for visual scene understanding and a meaningful measure of uncertainty is essential for decision making. **Our contribution is a practical system which is able to predict pixelwise class labels with a measure of model uncertainty.** We achieve this by Monte Carlo sampling with dropout at test time to generate a posterior distribution of pixel class labels. In addition, we show that modelling uncertainty improves segmentation performance by 2-3% across a number of state of the art architectures such as SegNet, FCN and Dilation Network, with no additional parametrisation. We also observe a significant improvement in performance for smaller datasets where modelling uncertainty is more effective. We benchmark Bayesian SegNet on the indoor SUN Scene Understanding and outdoor CamVid driving scenes datasets. |

| 会议／期刊 | 作者 | 论文 | 代码 |
| ---- | ---- | ---- | ---- |
| arXiv: 1511.02680 | **Alex Kendall**, Vijay Badrinarayanan, Roberto Cipolla | Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understandin | [caffe-segnet](https://github.com/alexgkendall/caffe-segnet) |

本文主要提出了一种基于概率的像素级语义分割框架Bayesian SegNet，通过建模模型不确定性能够在许多网络中都提升2-3%性能，如SegNet，FCN和Dilation网络。

---
## SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation

| 摘要 |
| ---- |
| We present a novel and practical deep fully convolutional neural network architecture for semantic pixel-wise segmentation termed **SegNet**. This core trainable segmentation engine consists of an encoder network, a corresponding decoder network followed by a pixel-wise classification layer. **The architecture of the encoder network is topologically identical to the 13 convolutional layers in the VGG16 network.** The role of the decoder network is to map the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification. The novelty of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature map(s). **Specifically, the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling.** This eliminates the need for learning to upsample. The upsampled maps are sparse and are then convolved with trainable filters to produce dense feature maps. We compare our proposed architecture with the widely adopted FCN and also with the well known DeepLab-LargeFOV, DeconvNet architectures. This comparison reveals **the memory versus accuracy** trade-off involved in achieving good segmentation performance.
SegNet was primarily motivated by scene understanding applications. Hence, it is designed to be efficient both in terms of memory and computational time during inference. It is also significantly smaller in the number of trainable parameters than other competing architectures and can be trained end-to-end using stochastic gradient descent. We also performed a controlled benchmark of SegNet and other architectures on both road scenes and SUN RGB-D indoor scene segmentation tasks. These quantitative assessments show that SegNet provides good performance with competitive inference time and most efficient inference memory-wise as compared to other architectures. We also provide a Caffe implementation of SegNet and a web demo at http://mi.eng.cam.ac.uk/projects/segnet/. |

| 会议／期刊 | 作者 | 论文 | 代码 |
| ---- | ---- | ---- | ---- |
| arXiv: 1511.00561 | Vijay Badrinarayanan, **Alex Kendall**, Roberto Cipolla | SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation | [caffe-segnet](https://github.com/alexgkendall/caffe-segnet) |

本文提出的SegNet是应用最为广泛的架构，其中SegNet-VGG16在性能和精度上都获得了较大的提升，主要指出了解码器使用的反池化操作。

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/segnet_1.png)

---
## U-Net: Convolutional Networks for Biomedical Image Segmentation

| 摘要 |
| ---- |
| There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, **we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently.** The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffee) and the trained networks are available at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net. |

| 会议／期刊 | 作者 | 论文 | 代码 |
| ---- | ---- | ---- | ---- |
| arXiv: 1505.04597 | Olaf Ronneberger, Philipp Fischer, Thomas Brox | U-Net: Convolutional Networks for Biomedical Image Segmentation | [unet第三方](https://github.com/zhixuhao/unet) |

本文提出的U-Net网络能够有效利用标注样本，通过a symmetric expanding path提升分割精度。

![](https://chenguanfuqq.gitee.io/tuquan2/img_2018_5/unetseg_1.png)