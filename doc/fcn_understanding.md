# FCN Understaning

## FCN32s
主架构由VGG16组成，第一个卷积层的padding为100，最后的线性分类器修改为全卷积网络，这样的网络输出大小和原始图像大小不同，使用上采样将输出结果上采样和原始分辨率相同。

## FCN16s

## FCN8s


## 运行时间
测量FCN32s的forward时间大约为7s。
