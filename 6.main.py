#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorchExercise
@File ：6.main.py
@Author ：月欣
@Date ：2023/6/20 22:11
@Contact: panyuexin654@163.com
'''
import numpy as np
import torch
from torch.nn import functional as F
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


# 定义函数
def himmelblau(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
print("x,y shape:",x.shape,y.shape)
X,Y = np.meshgrid(x,y)
print('X,Y shape:',X.shape,Y.shape)
Z = himmelblau([X,Y])
# print(Z)
fig = plt.figure('himmelblau')
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# [1,0] ,[-4,0],[4,0]
x = torch.tensor([-4.,0.],requires_grad=True )
# y = torch.tensor([0.],requires_grad=True )
optimizer = torch.optim.Adam([x],lr=1e-3)
for step in range(20000):
    pred = himmelblau(x)
    optimizer.zero_grad()
    pred.backward()
    optimizer.step()
    if step % 2000  ==0:
        print("step {}: x = {}  f(x,y) = {}".format(step,x.tolist(),pred.item()))