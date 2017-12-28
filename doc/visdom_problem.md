# 可视化开发相关问题


---
## 画线增量的情况下

```python
if args.vis:
    win = 'loss'
    win_res = vis.line(X=np.ones(1)*i, Y=loss.data.numpy(), win=win, update='append')
    if win_res != win:
        vis.line(X=np.ones(1)*i, Y=loss.data.numpy(), win=win)
```

在win没有出现的情况下使用update选项会提示win not exists

[Is there near plan to add more documents?](https://github.com/facebookresearch/visdom/issues/44)
