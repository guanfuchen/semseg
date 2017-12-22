# CityScapes数据集

这个大型数据集包含了来自50个不同城市的街景场景中记录的多样化的双目视频序列，除了20000个弱注释帧以外，还有5000帧的高质量像素级注释。

主要参考如下：
- [cityscapesScripts](https://github.com/mcordts/cityscapesScripts)
- [fastSceneUnderstanding](https://github.com/DavyNeven/fastSceneUnderstanding)

该数据集由gtFine和leftImg8bit这两个目录组成，结构如下所示，其中aachen等表示拍摄场景的城市名：
```
├── gtFine
│   ├── train
│   │   ├── aachen
│   │   ├── bochum
│   │   └── bremen
│   └── val
│       └── frankfurt
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   ├── bochum
    │   └── bremen
    └── val
        └── frankfurt
```
这个数据集包含语义分割，实例分割，深度估计等标签数据，对应的训练标签如下所示：
```
cd $CITYSCAPES_ROOT
# 训练和校准对应的数据集
ls leftImg8bit/train/*/*.png > trainImages.txt
ls leftImg8bit/val/*/*.png > valImages.txt

# 训练和校准标签对应的数据集
ls gtFine/train/*/*labelIds.png > trainLabels.txt
ls gtFine/val/*/*labelIds.png.png > valLabels.txt

# 训练和校准实例标签对应的数据集
ls gtFine/train/*/*instanceIds.png > trainInstances.txt
ls gtFine/val/*/*instanceIds.png.png > valInstances.txt

# 训练和校准深度标签对应的数据集
ls disparity/train/*/*.png > trainDepth.txt
ls disparity/val/*/*.png.png > valDepth.txt
```