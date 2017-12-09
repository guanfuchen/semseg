# CamVid数据集

![CamVid原始数据示例1](http://chenguanfuqq.oschina.io/tuquan/img_2017_12/2017_12_9_17_37_54.png)
![CamVid标注数据示例1](http://chenguanfuqq.oschina.io/tuquan/img_2017_12/2017_12_9_17_37_14.png)

参考如下：
- [SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)
- [camvid full label](https://github.com/mostafaizz/camvid)

测试集233张，训练集367张，校准集101张，总共233+367+101=701张，图像大小为360*480，语义类别为13类，包括以下类别

```bash
天空 Sky = [128, 128, 128]
建筑物 Building = [128, 0, 0]
路灯 Pole = [192, 192, 128]
道路标记 Road_marking = [255, 69, 0]
道路 Road = [128, 64, 128]
人行道 Pavement = [60, 40, 222]
树木 Tree = [128, 128, 0]
交通信号灯 SignSymbol = [192, 128, 128]
栏 Fence = [64, 64, 128]
汽车 Car = [64, 0, 128]
行人 Pedestrian = [64, 64, 0]
自行车手 Bicyclist = [0, 128, 192]
未标注 Unlabelled = [0, 0, 0]
```

在SegNet-Tutorial中CamVid文件下的目录如下所示：
- test/ 测试集原始图像，其中共有233张
- testannot/ 测试集标注图像
- train/ 训练集原始图像，其中共有367张
- trainannot/ 训练集标注图像
- val/ 校准集原始图像，其中共有101张
- valannot/ 校准集标注图像
- test.txt 测试集图像对
```
/SegNet/CamVid/test/0001TP_008550.png /SegNet/CamVid/testannot/0001TP_008550.png
```
- train.txt 训练集图像对
```
/SegNet/CamVid/train/0001TP_008550.png /SegNet/CamVid/trainannot/0001TP_008550.png
```
- val.txt 校准集图像对
```
/SegNet/CamVid/val/0001TP_008550.png /SegNet/CamVid/valannot/0001TP_008550.png
```
