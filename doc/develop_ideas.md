# 开发过程

---
## feature
- 增加评估函数```python validate.py```，输出语义分割指标
- 增加网络图显示函数```python graph_show.py```，输出网络的架构细节图
- 增加各个模型对应的VGG，ResNet和DenseNet等衍生版本，比如FCN-VGG、FCN-ResNet以及FCN-DenseNet
- 其他模型是否也可以像pspnet一样增加辅助训练，比如fcn8s中将fcn32s和fcn16s中的训练指标加入到总的训练中
- 训练策略，比如在[0-150]lr=0.1，[150-250]lr=0.01，[250-350]lr=0.01等
- 增加crf后处理
- 当前仅仅是按照时间存储模型，设计一种能在校验集中精度提升的储存思路
- 将预测结果和原图blend更为精确的显示
- 增加多个类似每一个数据集，每一个模型，Mean IU和Pixel Accuracy等评估性能，可参考[segmentation-experiments](https://github.com/wondervictor/segmentation-experiments)。