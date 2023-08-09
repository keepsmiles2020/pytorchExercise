#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorchExercise
@File ：5.chain_rule.py
@Author ：月欣
@Date ：2023/6/19 22:13
@Contact: panyuexin654@163.com
'''
import torch
from torch.nn import functional as F

x = torch.tensor(1.)
w1 = torch.tensor(2.,requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2.,requires_grad=True)
b2 = torch.tensor(1.)
y1 = x*w1+b1
y1.requires_grad_()
y2 = y1*w2+b2
dy2_dy1 = torch.autograd.grad(y2,[y1],retain_graph=True)[0]
dy1_dw1 = torch.autograd.grad(y1,[w1],retain_graph=True)[0]
dy2_dw1=torch.autograd.grad(y2,[w1],retain_graph=True)[0]

print(dy2_dy1*dy1_dw1)
print(dy2_dw1)
