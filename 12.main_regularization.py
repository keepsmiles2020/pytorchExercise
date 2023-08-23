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
from  visdom import Visdom
batch_size = 200
learning_rate = 0.01
epoches = 10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose(
                       [
                           transforms.ToTensor(),
                       ]
                   )
                   ), batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor
    ]))
)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(748, 200),
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
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)
criteon = nn.CrossEntropyLoss().to(device)

viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([0.0, 0.0], [0.0], win='test', opts=dict(
    title='test loss&acc.', legend=['loss', 'acc']))

global_step = 0

for epoch in range(epoches):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data,target = data.to(device),target.to(device)
        logit = net(data)
        loss = criteon(logit,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step +=1
        viz.line([loss.item()],[global_step],win='train_loss',update='append')
        if batch_idx %  100 ==0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()
            ))
    test_loss = 0
    correct = 0
    for data,target in test_loader:
        data = data.view(-1,28*28)
        data,target = data.to(device),target.to(device)
        logit = net(data)
        test_loss += criteon(logit, target).item()
        pre = logit.argmax(dim=1)
        correct += pre.eq(target).float().sum().item()

    viz.line([[test_loss,correct/len(test_loader.dataset)]],
             [global_step],win='test',update='append')
    viz.images(data.view(-1,1,28,28),win='x')
    viz.text(str(pre.detach().cpu().numpy()),win='pre',opts=dict(title='pre'))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
