#!/usr/bin/python
# -*- coding: UTF-8 -*-

from graphviz import Digraph
import torch
from torch.autograd import Variable
import numpy as np

from semseg.modelloader.fcn import fcn32s

# 网络模型绘图，生成pytorch autograd图表示，蓝色节点表示要求grad梯度的变量，黄色表示在反向传播的张量
def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(var_grad):
        if var_grad not in seen:
            if torch.is_tensor(var_grad):
                dot.node(str(id(var_grad)), size_to_str(var_grad.size()), fillcolor='orange')
            elif hasattr(var_grad, 'variable'):
                u = var_grad.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var_grad)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var_grad)), str(type(var_grad).__name__))
            seen.add(var_grad)
            if hasattr(var_grad, 'next_functions'):
                for u in var_grad.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var_grad)))
                        add_nodes(u[0])
            if hasattr(var_grad, 'saved_tensors'):
                for t in var_grad.saved_tensors:
                    dot.edge(str(id(t)), str(id(var_grad)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

def main():
    n_classes = 21
    model = fcn32s(n_classes=n_classes)
    x = Variable(torch.randn(1, 3, 360, 480))
    pred = model(x)
    g = make_dot(pred)
    # print(g)
    # g.render('model_vis.gv', view=True)


if __name__ == '__main__':
    main()
