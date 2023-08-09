#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorchExercise
@File ：3.activation_and_loss_gradient.py
@Author ：月欣
@Date ：2023/6/4 8:42
@Contact: panyuexin654@163.com
'''
import torch
from torch.nn import functional as F

a = torch.linspace(-100,100,10)

print(torch.sigmoid(a))

a = torch.linspace(-1,1,10)
print(torch.tanh(a))

# Relu
a = torch.linspace(-1,1,10)
print(F.relu(a))

# gradient loss
x = torch.ones(1)
w = torch.full([1],fill_value=2).float()
w.requires_grad_()
mse = F.mse_loss(x*w,torch.ones(1))
print(torch.autograd.grad(mse,[w]))
print(mse)
# print(torch.autograd.grad(mse,[w],requires_grad=True))

mse = F.mse_loss(x*w,torch.ones(1))
mse.backward()
print(w.grad)
# cross entropy
a = torch.rand(3)
a.requires_grad_()
p = F.softmax(a,dim=0)
print(torch.autograd.grad(p[1],p[0],retain_graph=True))

a=torch.rand(3)
a.requires_grad_()
p=F.softmax(a,dim=0)
print(torch.autograd.grad(p[1],[a],retain_graph=True))

x = torch.ones(2,requires_grad=True)
z = x +2
z.sum().backward()
print(x.grad)