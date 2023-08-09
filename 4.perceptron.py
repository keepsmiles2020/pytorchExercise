#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorchExercise
@File ：4.perceptron.py
@Author ：月欣
@Date ：2023/6/19 21:58
@Contact: panyuexin654@163.com
'''
import torch
from torch.nn import functional as F
a = torch.randn(1,10)
# tensor([[-0.0985,  0.3252, -0.9185,  0.8419, -0.0533, -1.8077,  0.0364,  0.3448,
#           2.5360,  0.6876]])
w = torch.randn(1,10,requires_grad=True )
o = F.sigmoid(a@w.t())
loss = F.mse_loss(torch.ones(1,1),o)
loss.backward()
print(w.grad)

# mult-layer perceptron
a = torch.randn(1,10)
w = torch.randn(2,10,requires_grad=True )
o = F.sigmoid(a@w.t())
loss = F.mse_loss(torch.ones(1,2),o)
loss.backward()
print(w.grad)



