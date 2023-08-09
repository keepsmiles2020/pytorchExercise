#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorchExercise
@File ：2.tensor_advanced.py
@Author ：月欣
@Date ：2023/6/3 21:50
@Contact: panyuexin654@163.com
'''
import torch
cond = torch.rand(2,2)
print(cond)
a = torch.zeros(2,2).float()
b = torch.ones(2,2).float()
print(a)
print(b)
print(torch.where(cond>0.5,a,b))

# gather
prob = torch.randn(4,10)
print('prob:\n',prob)
idx = prob.topk(dim=1,k=3)
print('idx:\n',idx)
print('idx[0]:\n',idx[0])
print('idx[1]:\n',idx[1])