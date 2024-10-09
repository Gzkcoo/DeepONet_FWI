# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:40:28 2024

@author: Ye Li
"""

import numpy as np
import torch
import torch.nn as nn
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from timeit import default_timer
import cv2
import json
import sys
import transforms as T
from torchvision.transforms import Compose
from dataset import FWIDataset

# Load colormap for velocity map visualization
rainbow_cmap = ListedColormap(np.load('rainbow256.npy'))

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class Trunk_Net_Discon(nn.Module):
    def __init__(self, layer_sizes, eps_num):
        super(Trunk_Net_Discon, self).__init__()
        self.values = torch.tensor([0.]).to(device)
        self.Wz = nn.ModuleList()
        self.eps_num = eps_num
        self.activation = nn.LeakyReLU(0.2)
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)

        self.epsilon1 = nn.Parameter(torch.randn(1, self.eps_num))
        self.epsilon2 = nn.Parameter(torch.randn(1, self.eps_num))

    def forward(self, x):
        H = x
        H = self.activation(self.Wz[0](H))
        H = self.activation(self.Wz[1](H))
        H = self.activation(self.Wz[2](H))
        H = self.activation(self.Wz[3](H)) + self.epsilon1 * torch.heaviside(self.Wz[3](H), values=self.values)
        H = self.activation(self.Wz[4](H)) + self.epsilon2 * torch.heaviside(self.Wz[4](H), values=self.values)
        H = self.activation(self.Wz[5](H))
        H = self.Wz[6](H)

        return H
    
    
class DDeeponet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, deepth=70, length=70, sample_spatial=1.0, **kwargs):
        super(DDeeponet, self).__init__()
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

        # Coordinate
        xc = torch.arange(1, deepth+1).to(device)
        yc = torch.arange(1, length+1).to(device)
        xm, ym = torch.meshgrid(xc, yc)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        self.xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
        self.xy_coordinate = self.xy_coordinate / 70  

        layer_sizes = [2, 256, 256, 256, 128, 128, 256, 512]
        self.trunk = Trunk_Net_Discon(layer_sizes, 128)    #

        self.layer1 = nn.Linear(512, 1)
        self.layer2 = nn.Linear(512, 2)  

        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.constant_(self.layer1.bias, 0.)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
        nn.init.constant_(self.layer2.bias, 0.)


    def forward(self, x):
        # Branch net
        x = self.convblock1(x)  # (None, 32, 500, 70)
        x = self.convblock2_1(x)  # (None, 64, 250, 70)
        x = self.convblock2_2(x)  # (None, 64, 250, 70)
        x = self.convblock3_1(x)  # (None, 64, 125, 70)
        x = self.convblock3_2(x)  # (None, 64, 125, 70)
        x = self.convblock4_1(x)  # (None, 128, 63, 70)
        x = self.convblock4_2(x)  # (None, 128, 63, 70)
        x = self.convblock5_1(x)  # (None, 128, 32, 35)
        x = self.convblock5_2(x)  # (None, 128, 32, 35)
        x = self.convblock6_1(x)  # (None, 256, 16, 18)
        x = self.convblock6_2(x)  # (None, 256, 16, 18)
        x = self.convblock7_1(x)  # (None, 256, 8, 9)
        x = self.convblock7_2(x)  # (None, 256, 8, 9)
        x = self.convblock8(x)  # (None, 512, 1, 1)

        # Trunk net
        t_out = self.trunk(self.xy_coordinate)  # (4900, 512)

        # Multiplication
        x = torch.squeeze(x)  # (None, 512)
        x = x.unsqueeze(1)  # (None, 1, 512)
        x = x.repeat(1, 4900, 1)  # (None, 4900, 512)
        x = x * t_out  # (None, 4900, 512)

        x1 = self.layer1(x)  # ææè¾åºçå®çéåº¦å¾  (None, 4900, 1)
        x2 = self.layer2(x)  # è¾åºè½®å»  (None, 4900, 2)

        x1 = x1.reshape((-1, 1, 70, 70))  # (None, 1, 70, 70)
        x2 = x2.reshape((-1, 2))  

        return x1, x2


def extract_contours(para_image):
    '''
    Use Canny to extract contour features

    :param image:       Velocity model (numpy)
    :return:            Binary contour structure of the velocity model (numpy)
    '''
    image = para_image
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image_to_255 = norm_image * 255
    norm_image_to_255 = norm_image_to_255.astype(np.uint8)
    canny = cv2.Canny(norm_image_to_255, 10, 15)
    bool_canny = np.clip(canny, 0, 1)
    return bool_canny
    
#未归一化
#data_brunch = np.load('.\FWIOpenData\FlatVel_A\data\data1.npy')
#data_label = np.load('.\FWIOpenData\FlatVel_A\model\model1.npy')

with open('dataset_config.json') as f:
    try:
        ctx = json.load(f)['flatvel-a']
    except KeyError:
        print('Unsupported dataset.')
        sys.exit()
ctx['file_size'] = 500
# Normalize data and label to [-1, 1]
transform_data = Compose([
    T.LogTransform(k=1),
    T.MinMaxNormalize(T.log_transform(ctx['data_min'], k=1), T.log_transform(ctx['data_max'], k=1))
])
transform_label = Compose([
    T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
])
dataset_train = FWIDataset(
    './split_files/flatvel_a_train.txt',
    preload=True,
    sample_ratio=1,
    file_size=ctx['file_size'],
    transform_data=transform_data,
    transform_label=transform_label
)


data_brunch = np.zeros((len(dataset_train),5,1000,70))
data_label = np.zeros((len(dataset_train),1,70,70))
for i in range(len(dataset_train)):
    data_brunch[i] = dataset_train[i][0]
    data_label[i]  = dataset_train[i][1]
conlabels = np.zeros(data_label.shape)
for i in range(data_label.shape[0]):
    for j in range(data_label.shape[1]):
        conlabels[i, j, ...] = extract_contours(data_label[i, j, ...])
data_label = torch.tensor(data_label).to(device)
data_brunch = torch.tensor(data_brunch).to(device)
conlabels = torch.tensor(conlabels).to(device)


learning_rate = 1e-3
epochs = 120
step_size = 40  # 每隔step_size个epoch时调整学习率为当前学习率乘gamma
gamma = 0.1
batch_size = 64
model = DDeeponet().to(device)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_brunch, data_label, conlabels), batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_history_ce = []
loss_history_l1 = []
loss_history_mse = []
l1loss = nn.L1Loss()
l2loss = nn.MSELoss()
l3loss = nn.CrossEntropyLoss()

for i in range(epochs):
    model.train()
    train_l1 = 0
    train_mse = 0
    train_ce = 0
    t1 = default_timer()
    for b_brunch, b_label, b_conlabels in train_loader:
        out1, out2 = model(b_brunch.float())
        optimizer.zero_grad()
        b_conlabels = b_conlabels.reshape(-1, 1)
        loss1 = l1loss(out1, b_label)
        loss2 = l2loss(out1.float(), b_label.float())
        loss3 = l3loss(out2, b_conlabels.squeeze().long())
        loss  = 1.*loss1 + 0.*loss2 + 0.1 *loss3
        loss.backward()
        optimizer.step()
        train_l1 += loss1.item()
        train_mse += loss2.item()
        train_ce += loss3.item()
    scheduler.step()
    train_l1 /= len(train_loader)
    train_mse /= len(train_loader)
    train_ce /= len(train_loader)
    t2 = default_timer()
    loss_history_l1.append(train_l1)
    loss_history_mse.append(train_mse)
    loss_history_ce.append(train_ce)
    print('epoch {:d} , CrossEntropy = {:.6f}, L1Loss = {:.6f}, MSELoss = {:.6f}, using {:.6f}s'.format(i, train_ce, train_l1, train_mse, t2 - t1))
        
        

softmax = nn.Softmax(dim=1)
conpred = softmax(out2[:,1,:,:]).detach().cpu().numpy()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
label = conpred[0]
vmax, vmin = np.max(label), np.min(label)
im = ax.matshow(label, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax, shrink=1.0, label='pred contour')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
label = b_conlabels[0,0].cpu().numpy()
vmax, vmin = np.max(label), np.min(label)
im = ax.matshow(label, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax, shrink=1.0, label='true contour')
plt.show()






