import os
import sys
import time
import datetime
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from torch import nn
from torch.utils.data import RandomSampler, DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

import utils
from vis import *
import network
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as T


# discontinuous network
class Trunk_Net_Discon(nn.Module):
    def __init__(self):
        super(Trunk_Net_Discon, self).__init__()
        self.values = torch.tensor([0.]).to('cuda')
        self.layer1 = nn.Linear(2, 256)
        self.layer2 = nn.Linear(256, 8)
        self.layer3 = nn.Linear(8, 8)
        self.layer4 = nn.Linear(8, 256)
        self.layer5 = nn.Linear(256, 1)
        # self.leakrelu = nn.LeakyReLU(0.2)

        # 连续激活函数 + epsilon * heaviside
        # epsilon 是可学习的向量
        self.epsilon1 = nn.Parameter(torch.randn(1, 8))   # 8为不连续层的节点个数
        self.epsilon2 = nn.Parameter(torch.randn(1, 8))

        nn.init.xavier_normal_(self.epsilon1, gain=1)
        nn.init.xavier_normal_(self.epsilon2, gain=1)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.constant_(self.layer1.bias, 0.)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
        nn.init.constant_(self.layer2.bias, 0.)
        nn.init.xavier_normal_(self.layer3.weight, gain=1)
        nn.init.constant_(self.layer3.bias, 0.)
        nn.init.xavier_normal_(self.layer4.weight, gain=1)
        nn.init.constant_(self.layer4.bias, 0.)
        nn.init.xavier_normal_(self.layer5.weight, gain=1)
        nn.init.constant_(self.layer5.bias, 0.)


    def forward(self, x):
        X = x
        H = torch.relu(self.layer1(X))
        H = torch.relu(self.layer2(H)) + self.epsilon1 * torch.heaviside(self.layer2(H), values=self.values)
        H = torch.relu(self.layer3(H)) + self.epsilon2 * torch.heaviside(self.layer3(H), values=self.values)
        H = torch.relu(self.layer4(H))
        H = self.layer5(H)
        return H


# continuous network
class Trunk_Net_Fcl(nn.Module):
    def __init__(self, layer_sizes):
        super(Trunk_Net_Fcl, self).__init__()
        self.values = torch.tensor([0.]).to('cuda')
        self.Wz = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)
        # self.layer1 = nn.Linear(2, 256)
        # self.layer2 = nn.Linear(256, 512)
        # self.layer3 = nn.Linear(512, 256)
        # # self.layer4 = nn.Linear(256, 256)
        # self.layer5 = nn.Linear(256, 1)
        # # self.leakrelu = nn.LeakyReLU(0.2)
        #
        # nn.init.xavier_normal_(self.layer1.weight, gain=1)
        # nn.init.constant_(self.layer1.bias, 0.)
        # nn.init.xavier_normal_(self.layer2.weight, gain=1)
        # nn.init.constant_(self.layer2.bias, 0.)
        # nn.init.xavier_normal_(self.layer3.weight, gain=1)
        # nn.init.constant_(self.layer3.bias, 0.)
        # # nn.init.xavier_normal_(self.layer4.weight, gain=1)
        # # nn.init.constant_(self.layer4.bias, 0.)
        # nn.init.xavier_normal_(self.layer5.weight, gain=1)
        # nn.init.constant_(self.layer5.bias, 0.)

    def forward(self, x):
        H = x
        for linear in self.Wz[0:-1]:
            H = torch.relu(linear(H))
        # H = torch.relu(self.layer4(H))
        H = self.Wz[-1](H)
        return H


class Fcl_Net(nn.Module):
    def __init__(self, layer_sizes, is_last_bias):
        super(Fcl_Net,self).__init__()
        self.Wz = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            m = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)
        last_layer = nn.Linear(layer_sizes[len(layer_sizes) - 2], layer_sizes[len(layer_sizes) - 1], bias=is_last_bias)
        nn.init.xavier_normal_(last_layer.weight, gain=1)
        if is_last_bias:
            nn.init.constant_(last_layer.bias, 0.)
        self.Wz.append(last_layer)

    def forward(self, x):
        X = x
        H = torch.relu(self.Wz[0](X))
        for linear in self.Wz[1:-1]:
                H = torch.relu(linear(H))
        ones = torch.ones_like(H)
        zero = torch.zeros_like(H)
        R = torch.where(H > 0, ones, zero)
        H = self.Wz[-1](H)
        return H, R


class Bias_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Bias_Net, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 512)
        self.last_layer = nn.Linear(512, output_size)

    def forward(self, x):
        X = x;
        H = torch.tanh(self.hidden_layer(X))
        H = self.last_layer(H)
        return H


# DeLU
class DeLU(nn.Module):
    def __init__(self, layer_sizes):
        super(DeLU, self).__init__()
        self.fcl_net = Fcl_Net(layer_sizes, False)  # false为最后一层不加bias
        bias_input_size = layer_sizes[len(layer_sizes) - 2]
        bias_output_size = layer_sizes[len(layer_sizes) - 1]
        self.bias_net = Bias_Net(bias_input_size, bias_output_size)


    def forward(self, x):
        X = x
        H1, R = self.fcl_net(X)
        H2 = self.bias_net(R)
        return H1 + H2

class MyNet(nn.Module):
    def __init__(self, layer_sizes):
        super(MyNet, self).__init__()
        self.values = torch.tensor([0.]).to('cuda')
        self.fcl_net = Fcl_Net(layer_sizes, True)
        self.bias_layer1 = nn.Linear(layer_sizes[0], 256)
        self.bias_layer2 = nn.Linear(256, 256)
        self.bias_layer3 = nn.Linear(256, layer_sizes[-1])
        self.epsilon1 = nn.Parameter(torch.randn(1, 8))  # 8为不连续层的节点个数
        # self.epsilon2 = nn.Parameter(torch.randn(1, 8))  # 8为不连续层的节点个数

        nn.init.xavier_normal_(self.epsilon1, gain=1)
        # nn.init.xavier_normal_(self.epsilon2, gain=1)
        nn.init.xavier_normal_(self.bias_layer1.weight, gain=1)
        nn.init.constant_(self.bias_layer1.bias, 0.)
        nn.init.xavier_normal_(self.bias_layer2.weight, gain=1)
        nn.init.constant_(self.bias_layer2.bias, 0.)
        nn.init.xavier_normal_(self.bias_layer3.weight, gain=1)
        nn.init.constant_(self.bias_layer3.bias, 0.)



    def forward(self, x):
        X = x
        H1,_ = self.fcl_net(X)
        H2 = torch.relu(self.bias_layer1(X))    # + self.epsilon1 * torch.heaviside(self.bias_layer1(X), values=self.values)
        # H2 = torch.relu(self.bias_layer2(H2)) + self.epsilon2 * torch.heaviside(self.bias_layer2(H2), values=self.values)
        H2 = torch.relu(self.bias_layer2(H2))
        H2 = self.bias_layer3(H2)
        return H1 + H2

# class Fcl_Net(nn.Module):
#     def __init__(self, layer_sizes, is_last_bias):
#         super(Fcl_Net,self).__init__()
#         self.Wz = nn.ModuleList()
#         for i in range(len(layer_sizes) - 2):
#             m = nn.Linear(layer_sizes[i], layer_sizes[i+1])
#             nn.init.xavier_normal_(m.weight, gain=1)
#             nn.init.constant_(m.bias, 0.)
#             self.Wz.append(m)
#         last_layer = nn.Linear(layer_sizes[len(layer_sizes) - 2], layer_sizes[len(layer_sizes) - 1], bias=is_last_bias)
#         nn.init.xavier_normal_(last_layer.weight, gain=1)
#         self.Wz.append(last_layer)
#
#     def forward(self, x):
#         X = x
#         H = torch.relu(self.Wz[0](X))
#         ones = torch.ones_like(H)
#         zero = torch.zeros_like(H)
#         R = torch.where(H > 0, ones, zero)
#         for linear in self.Wz[1:-1]:
#             H = torch.relu(linear(H))
#             ones = torch.ones_like(H)
#             zero = torch.zeros_like(H)
#             r = torch.where(H > 0, ones, zero)
#             R = torch.cat((R, r), 1)      # 将每一层的0,1向量 按列拼接
#         H = self.Wz[-1](H)
#         return H, R
#
#
# class Bias_Net(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Bias_Net, self).__init__()
#         self.hidden_layer = nn.Linear(input_size, 512)
#         self.last_layer = nn.Linear(512, output_size)
#
#     def forward(self, x):
#         X = x;
#         H = torch.tanh(self.hidden_layer(X))
#         H = self.last_layer(H)
#         return H
#
#
# # DeLU
# class DeLU(nn.Module):
#     def __init__(self, layer_sizes):
#         super(DeLU, self).__init__()
#         self.fcl_net = Fcl_Net(layer_sizes, False)  # false为最后一层不加bias
#         bias_input_size = sum(layer_sizes) - layer_sizes[0] - layer_sizes[-1]   # 所有隐藏层节点大小和
#         bias_output_size = layer_sizes[-1]
#         self.bias_net = Bias_Net(bias_input_size, bias_output_size)
#
#     def forward(self, x):
#         X = x
#         H1, R = self.fcl_net(X)
#         H2 = self.bias_net(R)
#         return H1 + H2




device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

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

# dataset_train2 = FWIDataset(
#     './split_files/flatvel_a_train.txt',
#     preload=True,
#     sample_ratio=1,
#     file_size=ctx['file_size']
# )


data_train, data_label = dataset_train[6]   # 测试第一张velocity图
data_label = torch.tensor(data_label).reshape(-1, 1)  # (4900,1)

# Coordinate
xc = torch.arange(1, 70+1)
yc = torch.arange(1, 70+1)
xm, ym = torch.meshgrid(xc, yc)
x = xm.reshape(-1, 1)
y = ym.reshape(-1, 1)
xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)

# 设置超参数
epochs = 50
learning_rate = 1e-3
batch_size = 256
lambda_g1v = 1
lambda_g2v = 0
layer_sizes = [2, 256, 512, 256, 1]



l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

model = Trunk_Net_Discon().to(device)  # 不连续神经网络
#model = Trunk_Net_Fcl(layer_sizes).to(device)  # 全连接神经网络
#model = DeLU(layer_sizes).to(device)
# model = MyNet(layer_sizes).to(device)
dataLoader = DataLoader(TensorDataset(xy_coordinate, data_label), batch_size=batch_size, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)


loss_history = []
for i in range(1, epochs + 1):
    model.train()
    for _, (coordinate, label) in enumerate(dataLoader):
        coordinate = coordinate.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        loss_g1v = l1loss(model(coordinate), label)
        loss_g2v = l2loss(model(coordinate), label)
        loss = lambda_g1v * loss_g1v + lambda_g2v * loss_g2v
        loss.backward()
        optimizer.step()
    loss_history.append(loss.item())
    print('epoch[' + str(i) + '] loss:' + str(loss.item()) + ' L1loss:' + str(loss_g1v.item()) + ' L2loss:' + str(loss_g2v.item()))


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

# Visualization
utils.mkdir('output')
with torch.no_grad():
    pred = model(xy_coordinate.to(device)).cpu()
    # label_np = T.tonumpy_denormalize(data_label, ctx['label_min'], ctx['label_max'], exp=False).reshape(70, 70)
    # pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False).reshape(70, 70)
    pred_np = pred.numpy().reshape(70, 70)
    label_np = data_label.numpy().reshape(70, 70)
    plot_velocity(pred_np, label_np, f'./output/vis.png')

print('--------------------pred_np-------------------')
print(pred_np)
print('--------------------con_pred-------------------')
con_pred = extract_contours(pred_np) * 255
con_label = extract_contours(label_np) * 255
print(con_pred)

img1 = Image.fromarray(con_pred)
img2 = Image.fromarray(con_label)
img1.show()
img2.show()


fig_1 = plt.figure(1)
plt.subplot(1, 1, 1)
plt.plot(loss_history, 'r', label='loss')
plt.xlabel('steps')
plt.ylabel('loss')
plt.legend()
plt.savefig('./output/loss.png')