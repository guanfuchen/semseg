# 开发过程

---
## feature
- 增加评估函数```python validate.py```，输出语义分割指标
- 增加网络图显示函数```python graph_show.py```，输出网络的架构细节图
- 增加各个模型对应的VGG，ResNet和DenseNet等衍生版本，比如FCN-VGG、FCN-ResNet以及FCN-DenseNet
- 其他模型是否也可以像pspnet一样增加辅助训练，比如fcn8s中将fcn32s和fcn16s中的训练指标加入到总的训练中