#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorchExercise
@File ：1.statistic.py
@Author ：月欣
@Date ：2023/5/26 22:56
@Contact: panyuexin654@163.com
'''
import torch

# 正则化
a = torch.full([8],fill_value=1,dtype=torch.float32)
b = a.view(2,4)
c = a.view(2,2,2)
print(torch.norm(a,1))
print(torch.norm(b,1,dim=1))

# 求张量中的最小值和最大值以及相乘
a = torch.arange(8).reshape(2,4).float()
a = torch.randn([4,10])
print(a)
print(a.argmax(1,keepdim=True))
print(a[:,0])
# Top-k/k-th
a = torch.randn(4,10)
print('---------')
print(a)
print(a.topk(3,dim=1))
print('=========')
print(a.topk(3,dim=1).indices)

