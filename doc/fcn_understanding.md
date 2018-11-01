# FCN Understaning

## 参考代码

- [fcn.berkeleyvision.org]()https://github.com/shelhamer/fcn.berkeleyvision.org) 作者论文实现代码，使用caffe
- [fcn](https://github.com/wkentaro/fcn) 使用chainer实现的fcn
- [pytorch-fcn](https://github.com/wkentaro/pytorch-fcn) 使用pytorch实现的fcn，其中很多训练相关的指标都借鉴于此，该代码给了原论文的结果。
- [FCN-pytorch](https://github.com/pochih/FCN-pytorch) 该版本在CamVid和Cityscapes上成功训练。

## FCN32s
主架构由VGG16组成，第一个卷积层的padding为100，最后的线性分类器修改为全卷积网络，这样的网络输出大小和原始图像大小不同，使用上采样将输出结果上采样和原始分辨率相同。

## FCN16s

## FCN8s


## 运行时间
测量FCN32s的forward时间大约为7s。

## 精度
fcn8s模型在CamVid数据集上运行两天后的精度，可以看出道路、天空、建筑物、人行道等精度较高
```bash
('FreqW Acc : \t', 0.78559525867563329)
('Overall Acc: \t', 0.8673878689952329)
('Mean Acc : \t', 0.574542563552356)
('Mean IoU : \t', 0.46866009473705755)
(0, 0.87483980790630644)
(1, 0.80011354499720377)
(2, 0.00033405881459735792)
(3, 0.91004779965501004)
(4, 0.74199948119548453)
(5, 0.86258904694843108)
(6, 0.062677848865206984)
(7, 0.37419241967934219)
(8, 0.50016352116938967)
(9, 0.16664317987651819)
(10, 0.18685390165119567)
(11, 0.14346652608600419)
(12, nan)
```