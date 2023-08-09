#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorchExercise
@File ：11.main_splited.py
@Author ：月欣
@Date ：2023/7/4 21:55
@Contact: panyuexin654@163.com
'''
'''
1.加载数据
先划分train 和test
2.定义网络
3.train_loader 来训练迭代，定义loss，输出
4.用val_loader来测试，挑选最好参数
5.将最好的参数模型加载进来，用test_loader来验收结果
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
batch_size = 64
learning_rate = 0.01
epoches = 10

# 划分数据集 train 和test
train_db = datasets.MNIST('./data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))]
                          ))
data_loader = torch.utils.data.DataLoader(
    train_db, batch_size=batch_size, shuffle=True
)
test_db = datasets.MNIST('./data', train=False, transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081))]
))
test_loader = torch.utils.data.DataLoader(
    test_db, batch_size=batch_size, shuffle=True
)
# 将train数据集进行train_loader与val_loader(5:1)
print('train:', len(train_db), 'test:', len(test_db))

train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
print('db1:', len(train_db), 'db2:', len(val_db))
train_loader = torch.utils.data.DataLoader(
    train_db, batch_size=batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_db, batch_size=batch_size, shuffle=True
)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epoches):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)

        logit = net(data)
        loss = criteon(logit,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 ==0:
            print('Train Epoech:{}[{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.item()
            ))

