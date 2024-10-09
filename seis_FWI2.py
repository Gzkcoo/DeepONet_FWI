# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:25:57 2023

deeponet for FlatVel-A dataset

@author: Ye Li
"""

import numpy as np
import struct
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy
import time
from timeit import default_timer
from torch.nn.parameter import Parameter
import random
from math import ceil
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

seismic_data = np.load('data1.npy')
seismic_data = seismic_data/40
rho_data = np.load('model1.npy')
rho_data = rho_data/4000
X = np.linspace(0, 1, 70)
T = np.linspace(0, 1, 1000)
X1, T1 = np.meshgrid(X, T)
X2, Y2 = np.meshgrid(X, X)

# =============================================================================
# fig, ax = plt.subplots()
# cs = plt.contourf(X1, T1, seismic_data[0,0], 100, cmap='RdBu_r', zorder=1)
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# cbar = fig.colorbar(cs)
# plt.show()
# fig, ax = plt.subplots()
# cs = plt.contourf(X2, Y2, rho_data[0,0], 100, cmap='RdBu_r', zorder=1)
# ax.set_xlabel('x')
# ax.set_ylabel('t')
# cbar = fig.colorbar(cs)
# plt.show()
# =============================================================================




class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        x = x.squeeze()   # (None, 512)
        return x


class Net_smooth(nn.Module):
    def __init__(self, layer_sizes):
        super(Net_smooth, self).__init__()

        self.Wz = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)
    
    def forward(self, x):
        X = x
        H = self.Wz[0](X)
        for linear in self.Wz[1:-1]:
            H = torch.relu(linear(H))
        H = self.Wz[-1](H)

        return H

class DeepONet(nn.Module):
    def __init__(self, t_dim, layer_sizes_t):
        super(DeepONet, self).__init__()
        self.t_dim = t_dim
        self.layer_sizes_t = layer_sizes_t
        
        self.branch1 = InversionNet()
        
        self.trunk = Net_smooth(self.layer_sizes_t)
        
        self.b = Parameter(torch.zeros(1))

        
    def forward(self, x1, lxy):
        x = self.branch1(x1)
        l = self.trunk(lxy)
        
        res = torch.einsum("bi,bki->bk", x, l)
        res = res.unsqueeze(-1) + self.b
        return res


N_src = 500
N_loc = 70 * 70
grid_vec = np.zeros((N_loc,2))  # (4900, 2)
u1_train = np.zeros((N_src,1))  # (500, 1)
loc_train = np.zeros((N_src,N_loc,2))  # (500, 4900, 2)
y_train = np.zeros((N_src,N_loc,1))  # (500, 4900, 1)
for i in range(70):
    for j in range(70):
        k = 70*i + j
        grid_vec[k] = [X2[i,j],Y2[i,j]]   # (4900, 2)
for i in range(N_src):
    u1_train[i,0] = i
    loc_train[i] = grid_vec
    y_train[i] = rho_data[i,0].reshape(N_loc,1)
u1_train = torch.Tensor(u1_train).to(device)
loc_train = torch.Tensor(loc_train).to(device)
y_train = torch.Tensor(y_train).to(device)

learning_rate = 1e-3
epochs = 1000
step_size = 100  # 每隔step_size个epoch时调整学习率为当前学习率乘gamma
gamma = 0.5
batch_size = 4
t_size = [2,512,512]
model = DeepONet(2, t_size).to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(u1_train, loc_train, y_train), batch_size=batch_size, shuffle=True)

optimizer = optim.Adam([
    {'params':model.parameters()}],lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_history_mse = []
loss_func = nn.MSELoss()

for i in range(epochs):
    t1 = default_timer()
    model.train()
    train_mse = 0
    for u1, loc, y in train_loader:
        optimizer.zero_grad()  # 梯度清零
        u_rho = np.zeros((u1.shape[0],5,1000,70))
        for n in range(u1.shape[0]):
            u_rho[n] = seismic_data[int(u1[n,0])]
        u_rho = torch.Tensor(u_rho).to(device)
        out = model(u_rho,loc)
        loss = loss_func(out,y) 
        loss.backward()  # 计算梯度
        optimizer.step()
        train_mse += loss.item()
    scheduler.step()
    train_mse /= len(train_loader)
    t2 = default_timer()
    loss_history_mse.append(train_mse)
    if i%100==0:
        print('epoch {:d} , MSE = {:.6f}, using {:.6f}s'.format(i, train_mse, t2 - t1))
        
y_pred = model(torch.Tensor(seismic_data[0:2]).to(device),loc_train[0:2]).detach().cpu().numpy()
y_p = y_pred[0].reshape(70,70)
fig, ax = plt.subplots()
cs = plt.contourf(X2, Y2, y_p, 100, cmap='RdBu_r', zorder=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
cbar = fig.colorbar(cs)
plt.show()
l2_rel = np.linalg.norm(rho_data[0,0]-y_p) / np.linalg.norm(rho_data[0,0])
print('relative l2 error:',l2_rel)