# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from collections import OrderedDict
from torchvision import models
import fno_model

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

# Replace the key names in the checkpoint in which legacy network building blocks are used 
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
              .replace('Conv2DwithBN_Tanh', 'layers')
              .replace('Deconv2DwithBN', 'layers')
              .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=0.2, dropout=None):
        super(Conv2DwithBN,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)
 
class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(Conv2DwithBN_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

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


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



# FlatFault/CurveFault
# 1000, 70 -> 70, 70
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
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
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
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

class FCN4_Deep_Resize_2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2, self).__init__()
        self.convblock1 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(70 * ratio / 8)), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
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
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x

class Conv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=None, stride=None, padding=None, **kwargs):
        super(Conv_HPGNN, self).__init__()
        layers = [
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8),
        ]
        if kernel_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Deconv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size, **kwargs):
        super(Deconv_HPGNN, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_fea, in_fea, kernel_size=kernel_size, stride=2, padding=0),
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# DeLU
class DeLU(nn.Module):
    def __init__(self, layer_sizes, bias_sizes):
        super(DeLU, self).__init__()
        self.fcl_net = Fcl_Net(layer_sizes, False)  # false为最后一层不加bias
        self.bias_net = Bias_Net(bias_sizes)


    def forward(self, x):
        X = x
        H1, R = self.fcl_net(X)
        H2 = self.bias_net(R)
        return H1 + H2

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
    def __init__(self, bias_sizes):
        super(Bias_Net, self).__init__()
        self.Wz = nn.ModuleList()
        for i in range(len(bias_sizes) -1):
            m = nn.Linear(bias_sizes[i], bias_sizes[i+1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)

    def forward(self, x):
        H = x
        for linear in self.Wz[0:-1]:
            H = torch.tanh(linear(H))
        H = self.Wz[-1](H)
        return H



# continuous network
class Smooth_Net(nn.Module):
    def __init__(self, layer_sizes):
        super(Smooth_Net, self).__init__()
        self.Wz = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)


    def forward(self, x):
        H = x
        for linear in self.Wz[0:-1]:
            H = torch.relu(linear(H))
        H = self.Wz[-1](H)
        return H

# 不连续神经网络
class Trunk_Net_Discon(nn.Module):
    def __init__(self, layer_sizes, eps_num):
        super(Trunk_Net_Discon, self).__init__()
        self.values = torch.tensor([0.]).to('cuda')
        self.Wz = nn.ModuleList()
        self.eps_num = eps_num
        self.activation = nn.LeakyReLU(0.2)
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)

        # self.epsilon1 = nn.Parameter(torch.ones(1))
        # self.epsilon2 = nn.Parameter(torch.ones(1))

        # 连续激活函数 + epsilon * heaviside
        # epsilon 是可学习的向量
        self.epsilon1 = nn.Parameter(torch.randn(1, self.eps_num))
        self.epsilon2 = nn.Parameter(torch.randn(1, self.eps_num))

        # nn.init.xavier_normal_(self.epsilon1, gain=1)
        # nn.init.xavier_normal_(self.epsilon2, gain=1)



    def forward(self, x):
        H = x
        H = self.activation(self.Wz[0](H))
        H = self.activation(self.Wz[1](H)) + self.epsilon1 * torch.heaviside(self.Wz[1](H), values=self.values)
        H = self.activation(self.Wz[2](H)) + self.epsilon2 * torch.heaviside(self.Wz[2](H), values=self.values)
        H = self.activation(self.Wz[3](H))
        H = self.Wz[4](H)

        return H


class FWIDeeponet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, deepth=70, length=70, sample_spatial=1.0, layer_sizes_t=[2, 256,256, 256, 512], **kwargs):
        super(FWIDeeponet, self).__init__()
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

        self.length = length
        self.deepth = deepth

        self.b = nn.Parameter(torch.zeros(1))

        # Coordinate
        xc = torch.arange(1, deepth+1)
        yc = torch.arange(1, length+1)
        xm, ym = torch.meshgrid(xc, yc)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        self.xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
        self.xy_coordinate = self.xy_coordinate / length  # 输入转换为（0，1）之间
        self.xy_coordinate = self.xy_coordinate.to('cuda')

        self.layer_sizes_t = [2, 256,256, 256, 512]
        self.trunk = Smooth_Net(self.layer_sizes_t)

        # layer_sizes = [2, 256, 256, 256, 64, 64, 256, 512]
        # self.trunk = Trunk_Net_Discon(layer_sizes, 64)    # 不连续神经网络

        # layer_sizes = [2, 256, 256, 512, 512]
        # bias_sizes = [512, 512, 512]
        # self.trunk = DeLU(layer_sizes, bias_sizes)   # DeLU

        # layer_sizes = [2, 256, 256, 512, 512]
        # self.trunk = Trunk_Net_Fcl(layer_sizes)       # 全连接神经网络

        #
        # self.last_layer = nn.Linear(512, 128)
        # nn.init.xavier_normal_(self.last_layer.weight, gain=1)
        # nn.init.constant_(self.last_layer.bias, 0.)





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

        x = torch.squeeze(x)  # (None, 512)
        # x = self.last_layer(x)  # (None, 128)

        # Trunk net
        t_out = self.trunk(self.xy_coordinate)  # (4900, 512)

        # Multiplication

        # x = x.unsqueeze(1)  # (None, 1, 512)
        # x = x.repeat(1, 4900, 1)  # (None, 4900, 512)
        # x = x * t_out  # (None, 4900, 512)
        # x = self.last_layer(x)  # (None, 4900, 1)

        # x = torch.sum(x, dim=2)  # (None, 4900)

        x = torch.einsum('bi,li->bl', x, t_out)
        x = x.unsqueeze(-1) + self.b
        x = x.reshape((-1, 1, self.deepth, self.length))  # (None, 1, 70, 70)
        # 图像转置
        # x = torch.transpose(x, 2, 3)
        return x


class FWIDeeponet2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, deepth=70, length=70, sample_spatial=1.0, **kwargs):
        super(FWIDeeponet2, self).__init__()
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

        self.length = length
        self.deepth = deepth

        self.b = nn.Parameter(torch.zeros(1))

        self.values = torch.tensor([0.]).to('cuda')

        # Coordinate
        xc = torch.arange(1, deepth+1)
        yc = torch.arange(1, length+1)
        xm, ym = torch.meshgrid(xc, yc)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        self.xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
        self.xy_coordinate = self.xy_coordinate / length  # 输入转换为（0，1）之间
        self.xy_coordinate = self.xy_coordinate.to('cuda')

        layer_sizes = [2, 256, 256, 256, 64, 64, 256, 512]
        self.trunk = Trunk_Net_Discon(layer_sizes, 64)    # 不连续神经网络

        # layer_sizes = [2, 256, 256, 512, 512]
        # bias_sizes = [512, 512, 512]
        # self.trunk = DeLU(layer_sizes, bias_sizes)   # DeLU

        # layer_sizes = [2, 256, 256, 512, 512]
        # self.trunk = Trunk_Net_Fcl(layer_sizes)       # 全连接神经网络


        # self.last_layer = nn.Linear(512, 128)
        # nn.init.xavier_normal_(self.last_layer.weight, gain=1)
        # nn.init.constant_(self.last_layer.bias, 0.)


        # DeLU
        self.de_layer1 = nn.Linear(2, 256)
        self.de_layer2 = nn.Linear(256, 256)
        self.de_layer3 = nn.Linear(256, 1)


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

        x = torch.squeeze(x)  # (None, 512)
        # x = self.last_layer(x)  # (None, 128)

        # Trunk net
        t_out = self.trunk(self.xy_coordinate)  # (4900, 512)

        x = torch.einsum('bi,li->bl', x, t_out)
        x = x.unsqueeze(-1) + self.b  # (None, 4900, 1)


        # DeLU
        d_x = torch.tanh(self.de_layer1(self.xy_coordinate))
        d_x = torch.heaviside(self.de_layer2(d_x), values=self.values)  # (4900, 256)
        d_x = self.de_layer3(d_x)  # (4900, 1)

        x = x + d_x


        x = x.reshape((-1, 1, self.deepth, self.length))  # (None, 1, 70, 70)

        return x


# continuous network  with BatchNorm1d
class BN_Smooth_Net(nn.Module):
    def __init__(self, layer_sizes):
        super(BN_Smooth_Net, self).__init__()
        self.Wz = nn.ModuleList()
        self.Bn = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            bn = nn.BatchNorm1d(layer_sizes[i + 1])
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)
            self.Bn.append(bn)


    def forward(self, x):
        H = x
        for i, linear in enumerate(self.Wz[0:-1]):
            H = self.Bn[i](torch.relu(linear(H)))
        H = self.Wz[-1](H)
        return H


class FWIDeeponet3(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, deepth=70, length=70, sample_spatial=1.0, **kwargs):
        super(FWIDeeponet3, self).__init__()
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

        self.length = length
        self.deepth = deepth

        self.values = torch.tensor([0.]).to('cuda')

        # Coordinate
        xc = torch.arange(1, deepth+1)
        yc = torch.arange(1, length+1)
        xm, ym = torch.meshgrid(xc, yc)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        self.xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
        self.xy_coordinate = self.xy_coordinate / length  # 输入转换为（0，1）之间
        self.xy_coordinate = self.xy_coordinate.to('cuda')

        layer_sizes = [2, 256, 256, 256, 256, 256, 256, 64]
        self.trunk = BN_Smooth_Net(layer_sizes)

        # layer_sizes = [2, 256, 256, 512, 512]
        # bias_sizes = [512, 512, 512]
        # self.trunk = DeLU(layer_sizes, bias_sizes)   # DeLU

        # layer_sizes = [2, 256, 256, 512, 512]
        # self.trunk = Trunk_Net_Fcl(layer_sizes)       # 全连接神经网络


        self.last_layer = nn.Linear(512, 64)
        nn.init.xavier_normal_(self.last_layer.weight, gain=1)
        nn.init.constant_(self.last_layer.bias, 0.)


        # DeLU
        self.de_layer1 = nn.Linear(2, 64)
        self.de_layer2 = nn.Linear(64, 128)
        self.de_layer3 = nn.Linear(128, 256)
        self.de_layer4 = nn.Linear(256, 64)

        nn.init.xavier_normal_(self.de_layer1.weight, gain=1)
        nn.init.constant_(self.de_layer1.bias, 0.)
        nn.init.xavier_normal_(self.de_layer2.weight, gain=1)
        nn.init.constant_(self.de_layer2.bias, 0.)
        nn.init.xavier_normal_(self.de_layer3.weight, gain=1)
        nn.init.constant_(self.de_layer3.bias, 0.)
        nn.init.xavier_normal_(self.de_layer4.weight, gain=1)
        nn.init.constant_(self.de_layer4.bias, 0.)


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

        x = torch.squeeze(x)  # (None, 512)
        x = self.last_layer(x)  # (None, 64)

        # Trunk net
        t_out = self.trunk(self.xy_coordinate)  # (4900, 64)

        # (x, y) 划分
        d_x = torch.tanh(self.de_layer1(self.xy_coordinate))
        d_x = torch.heaviside(self.de_layer2(d_x), values=self.values)  # (4900, 256)
        d_x = torch.tanh(self.de_layer3(d_x))
        d_x = self.de_layer4(d_x)  # (4900, 64)

        t_out = t_out + d_x

        x = torch.einsum('bi,li->bl', x, t_out)
        x = x.unsqueeze(-1)   # (None, 4900, 1)

        x = x + d_x


        x = x.reshape((-1, 1, self.deepth, self.length))  # (None, 1, 70, 70)

        return x



class InversionNet_Encoder(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet_Encoder, self).__init__()
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

    def forward(self, x):
        # Encoder Part
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
        x = x.squeeze()  # (None, 512)
        return x

class InversionNet_Decoder(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet_Decoder, self).__init__()
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)

    def forward(self, x):
        # Decoder Part
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x)  # (None, 1, 70, 70)
        return x


class InversionNet_Decoder2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet_Decoder2, self).__init__()
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6_1 = ConvBlock_Tanh(dim1, 1)
        self.deconv6_2 = ConvBlock_Tanh(dim1, 2)

    def forward(self, x):
        # Decoder Part
        x = self.deconv1_1(x)  # (None, 512, 5, 5)
        x = self.deconv1_2(x)  # (None, 512, 5, 5)
        x = self.deconv2_1(x)  # (None, 256, 10, 10)
        x = self.deconv2_2(x)  # (None, 256, 10, 10)
        x = self.deconv3_1(x)  # (None, 128, 20, 20)
        x = self.deconv3_2(x)  # (None, 128, 20, 20)
        x = self.deconv4_1(x)  # (None, 64, 40, 40)
        x = self.deconv4_2(x)  # (None, 64, 40, 40)
        x = self.deconv5_1(x)  # (None, 32, 80, 80)
        x = self.deconv5_2(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70) 125, 100
        x1 = self.deconv6_1(x)  # (None, 1, 70, 70)
        x2 = self.deconv6_2(x)  # (None, 2, 70, 70)
        return x1, x2

class DeepONet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(DeepONet, self).__init__()

        self.layer_sizes_t = [2, 256, 256, 256, 128, 128, 256, 512]

        self.branch1 = InversionNet_Encoder(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, dim5=dim5, sample_spatial=sample_spatial)

        # 全连接神经网络
        # self.trunk = Smooth_Net(self.layer_sizes_t)

        #不连续神经网络
        self.trunk = Trunk_Net_Discon(self.layer_sizes_t, 128)

        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x1, lxy):
        x = self.branch1(x1)
        l = self.trunk(lxy)

        res = torch.einsum("bi,bki->bk", x, l)
        res = res.unsqueeze(-1) + self.b
        return res


class FWIEnDeepOnet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, deepth=70, length=70, sample_spatial=1.0, **kwargs):
        super(FWIEnDeepOnet, self).__init__()
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
        xc = torch.arange(1, deepth + 1)
        yc = torch.arange(1, length + 1)
        xm, ym = torch.meshgrid(xc, yc)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        self.xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
        self.xy_coordinate = self.xy_coordinate / length   # 输入转换为（0，1）之间
        self.xy_coordinate = self.xy_coordinate.to('cuda')

        # layer_sizes = [2, 256, 256, 256, 32, 32, 256, 512]
        # self.trunk = Trunk_Net_Discon(layer_sizes, 32)  # 不连续神经网络

        # layer_sizes = [2, 256, 256, 512, 512]
        # bias_sizes = [512, 512, 512]
        # self.trunk = DeLU(layer_sizes, bias_sizes)   # DeLU

        layer_sizes = [2, 256, 256, 256, 512, 512]
        self.trunk = Smooth_Net(layer_sizes)       # 全连接神经网络

        # RootNet
        self.add_net = nn.Linear(512, 1, bias=False)
        self.substract_net = nn.Linear(512, 1, bias=False)
        self.multiply_net = nn.Linear(512, 1, bias=False)

        nn.init.xavier_normal_(self.add_net.weight, gain=1)
        nn.init.xavier_normal_(self.substract_net.weight, gain=1)
        nn.init.xavier_normal_(self.multiply_net.weight, gain=1)





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

        # add   subtract   Multiply
        x = torch.squeeze(x)  # (None, 512)
        x = x.unsqueeze(1)  # (None, 1, 512)
        x = x.repeat(1, 4900, 1)  # (None, 4900, 512)

        x_multiply = x * t_out  # (None, 4900, 512)
        x_multiply = self.multiply_net(x_multiply)  # (None, 4900, 1)

        x_add = x + t_out    # (None, 4900, 512)
        x_add = self.add_net(x_add)  # (None, 4900, 1)

        x_subtract = x - t_out   # (None, 4900, 512)
        x_subtract = self.substract_net(x_subtract)   # (None, 4900, 1)

        x = x_subtract + x_add + x_multiply   # (None, 4900, 1)

        x = x.reshape((-1, 1, 70, 70))  # (None, 1, 70, 70)


        return x

# deeponet两个输出  其中轮廓输出只有一个通道
class DDeeponet2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, deepth=70, length=70, sample_spatial=1.0, **kwargs):
        super(DDeeponet2, self).__init__()
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
        xc = torch.arange(1, deepth+1)
        yc = torch.arange(1, length+1)
        xm, ym = torch.meshgrid(xc, yc)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        self.xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
        self.xy_coordinate = self.xy_coordinate / 70  # 输入转换为（0，1）之间
        self.xy_coordinate = self.xy_coordinate.to('cuda')

        layer_sizes = [2, 256, 256, 256, 128, 128, 256, 512]
        self.trunk = Trunk_Net_Discon(layer_sizes, 128)    # 不连续神经网络

        # layer_sizes = [2, 256, 256, 512, 512]
        # bias_sizes = [512, 512, 512]
        # self.trunk = DeLU(layer_sizes, bias_sizes)   # DeLU

        # layer_sizes = [2, 256, 256, 512, 512]
        # self.trunk = Trunk_Net_Fcl(layer_sizes)       # 全连接神经网络


        # 全连接层
        self.layer1 = nn.Linear(512, 1)
        self.layer2 = nn.Linear(512, 2)  # 两个输出

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

        x1 = self.layer1(x)  # 期望输出真实的速度图  (None, 4900, 1)
        x2 = self.layer2(x)  # 输出轮廓  (None, 4900, 2)

        x1 = x1.reshape((-1, 1, 70, 70))  # (None, 1, 70, 70)
        x2 = x2.reshape((-1, 2))  # (None, 2)

        return x1, x2


# deeponet两个输出   其中轮廓输出只有一个通道
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
        xc = torch.arange(1, deepth+1)
        yc = torch.arange(1, length+1)
        xm, ym = torch.meshgrid(xc, yc)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        self.xy_coordinate = torch.torch.cat([x, y], dim=1).float()  # (4900, 2)
        self.xy_coordinate = self.xy_coordinate / 70  # 输入转换为（0，1）之间
        self.xy_coordinate = self.xy_coordinate.to('cuda')

        layer_sizes = [2, 256, 256, 256, 128, 128, 256, 512]
        self.trunk = Trunk_Net_Discon(layer_sizes, 128)    # 不连续神经网络

        # layer_sizes = [2, 256, 256, 512, 512]
        # bias_sizes = [512, 512, 512]
        # self.trunk = DeLU(layer_sizes, bias_sizes)   # DeLU

        # layer_sizes = [2, 256, 256, 512, 512]
        # self.trunk = Trunk_Net_Fcl(layer_sizes)       # 全连接神经网络


        # 全连接层
        self.layer1 = nn.Linear(512, 1)
        self.layer2 = nn.Linear(512, 1)  # 轮廓输出

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

        x1 = self.layer1(x)  # 期望输出真实的速度图  (None, 4900, 1)
        x2 = self.layer2(x)  # 输出轮廓  (None, 4900, 1)

        x1 = x1.reshape((-1, 1, 70, 70))  # (None, 1, 70, 70)
        x2 = x2.reshape((-1, 1))  # (None, 1)

        return x1, x2


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# 感知损失
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 加载预训练的VGG模型
        self.vgg = models.vgg16(pretrained=False).features
        self.vgg[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.vgg.load_state_dict(torch.load("vgg16-2.pth"))
        self.vgg.eval()
        self.vgg.requires_grad_(False)

    def forward(self, input, target):

        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return nn.L1Loss()(input_features, target_features)


class InversionDeepOnet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, layer_sizes_t=[5, 256 ,256, 256, 512], **kwargs):
        super(InversionDeepOnet, self).__init__()

        self.branch = InversionNet_Encoder(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, dim5=dim5,
                                            sample_spatial=sample_spatial)

        # 全连接神经网络
        self.layer_sizes_t = layer_sizes_t
        self.trunk = Smooth_Net(self.layer_sizes_t)

        # 不连续神经网络
        # self.layer_sizes_t = [5, 128, 32, 32, 256, 512]
        # self.trunk = Trunk_Net_Discon(self.layer_sizes_t, 32)

        # 解码器
        self.decoder = InversionNet_Decoder(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, dim5=dim5,
                                            sample_spatial=sample_spatial)

        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, loc):

        x_branch = self.branch(x)   # (None, 512)
        x_trunk = self.trunk(loc)    # (None, 512)
        x_dot = x_branch * x_trunk   # (None, 512)

        x_dot = x_dot.reshape(-1, 512, 1, 1)
        result = self.decoder(x_dot)     # (None, 1, 70, 70)

        return result


class InversionDeepOnet2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, layer_sizes_t=[5, 256,256, 256, 512], **kwargs):
        super(InversionDeepOnet2, self).__init__()

        self.branch = InversionNet_Encoder(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, dim5=dim5,
                                            sample_spatial=sample_spatial)

        # 全连接神经网络
        self.layer_sizes_t = layer_sizes_t
        self.trunk = Smooth_Net(self.layer_sizes_t)

        # 不连续神经网络
        # self.layer_sizes_t = [5, 128, 32, 32, 256, 512]
        # self.trunk = Trunk_Net_Discon(self.layer_sizes_t, 32)

        # 解码器
        self.decoder = InversionNet_Decoder2(dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, dim5=dim5,
                                            sample_spatial=sample_spatial)

        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x, loc):

        x_branch = self.branch(x)   # (None, 512)
        x_trunk = self.trunk(loc)    # (None, 512)
        x_dot = x_branch * x_trunk   # (None, 512)

        x_dot = x_dot.reshape(-1, 512, 1, 1)
        result1, result2 = self.decoder(x_dot)     # (None, 1, 70, 70)

        return result1, result2

class SeismicRecordDownSampling(nn.Module):
    '''
    Downsampling module for seismic records
    '''
    def __init__(self, shot_num):
        super().__init__()

        self.pre_dim_reducer1 = ConvBlock(shot_num, 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer2 = ConvBlock(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer3 = ConvBlock(8, 16, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer4 = ConvBlock(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pre_dim_reducer5 = ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.pre_dim_reducer6 = ConvBlock(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):

        width = x.shape[3]
        new_size = [width * 8, width]
        dimred0 = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

        dimred1 = self.pre_dim_reducer1(dimred0)
        dimred2 = self.pre_dim_reducer2(dimred1)
        dimred3 = self.pre_dim_reducer3(dimred2)
        dimred4 = self.pre_dim_reducer4(dimred3)
        dimred5 = self.pre_dim_reducer5(dimred4)
        dimred6 = self.pre_dim_reducer6(dimred5)

        return dimred6

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Conventional Network Unit
        (The red arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_batchnorm: Whether to use BN
        :param activ_fuc:    Activation function
        '''
        super(unetConv2, self).__init__()
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       nn.BatchNorm2d(out_size),
                                       activ_fuc)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       activ_fuc)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       activ_fuc)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetDown(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, activ_fuc = nn.ReLU(inplace=True)):
        '''
        Downsampling Unit
        (The purple arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_batchnorm: Whether to use BN
        :param activ_fuc:    Activation function
        '''
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, is_batchnorm, activ_fuc)
        self.down = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, inputs):
        skip_output = self.conv(inputs)
        outputs = self.down(skip_output)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv, activ_fuc=nn.ReLU(inplace=True)):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(unetUp, self).__init__()
        self.output_lim = output_lim
        self.conv = unetConv2(in_size, out_size, True, activ_fuc)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input1, input2):
        input2 = self.up(input2)
        input2 = F.interpolate(input2, size=self.output_lim, mode='bilinear', align_corners=False)
        return self.conv(torch.cat([input1, input2], 1))

class netUp(nn.Module):
    def __init__(self, in_size, out_size, output_lim, is_deconv):
        '''
        Upsampling Unit
        (The yellow arrow shown in Fig 1 of the paper)

        :param in_size:      Number of channels of input
        :param out_size:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        :param activ_fuc:    Activation function
        '''
        super(netUp, self).__init__()
        self.output_lim = output_lim
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, input):
        input = self.up(input)
        output = F.interpolate(input, size=self.output_lim, mode='bilinear', align_corners=False)
        return output

class DDNet70Model(nn.Module):
    def __init__(self, n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, **kwargs):
        '''
        DD-Net70 Architecture

        :param n_classes:    Number of channels of output (any single decoder)
        :param in_channels:  Number of channels of network input
        :param is_deconv:    Whether to use deconvolution
        :param is_batchnorm: Whether to use BN
        '''
        super(DDNet70Model, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)

        # Intrinsic UNet section
        self.down3 = unetDown(32, 64, self.is_batchnorm)
        self.down4 = unetDown(64, 128, self.is_batchnorm)
        self.down5 = unetDown(128, 256, self.is_batchnorm)

        self.center = unetDown(256, 512, self.is_batchnorm)

        self.up5 = unetUp(512, 256, output_lim=[9, 9], is_deconv=self.is_deconv)
        self.up4 = unetUp(256, 128, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.up3 = netUp(128, 64, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.up2 = netUp(64, 32, output_lim=[70, 70], is_deconv=self.is_deconv)

        self.dc1_final = ConvBlock_Tanh(32, self.n_classes)
        self.dc2_final = ConvBlock_Tanh(32, 2)

    def forward(self, inputs, _=None):
        '''

        :param inputs:      Input Image
        :param _:           Variables for filling, for alignment with DD-Net
        :return:
        '''

        compress_seis = self.pre_seis_conv(inputs)

        down3 = self.down3(compress_seis)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        center = self.center(down5)

        # 16*18*512
        decoder1_image = center
        decoder2_image = center

        #################
        ###  Decoder1 ###
        #################
        dc1_up5 = self.up5(down5, decoder1_image)
        dc1_up4 = self.up4(down4, dc1_up5)
        dc1_up3 = self.up3(dc1_up4)
        dc1_up2 = self.up2(dc1_up3)


        #################
        ###  Decoder2 ###
        #################
        dc2_up5 = self.up5(down5, decoder2_image)
        dc2_up4 = self.up4(down4, dc2_up5)
        dc2_up3 = self.up3(dc2_up4)
        dc2_up2 = self.up2(dc2_up3)


        return self.dc1_final(dc1_up2), self.dc2_final(dc2_up2)

class DDNetDeepOnet(nn.Module):
    def __init__(self, n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, layer_sizes_t=[5, 256,256, 256, 512],  **kwargs):
        super(DDNetDeepOnet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes

        self.pre_seis_conv = SeismicRecordDownSampling(self.in_channels)

        # Intrinsic UNet section
        self.down3 = unetDown(32, 64, self.is_batchnorm)
        self.down4 = unetDown(64, 128, self.is_batchnorm)
        self.down5 = unetDown(128, 256, self.is_batchnorm)

        self.center = unetDown(256, 512, self.is_batchnorm)

        self.up5 = unetUp(512, 256, output_lim=[9, 9], is_deconv=self.is_deconv)
        self.up4 = unetUp(256, 128, output_lim=[18, 18], is_deconv=self.is_deconv)
        self.up3 = netUp(128, 64, output_lim=[35, 35], is_deconv=self.is_deconv)
        self.up2 = netUp(64, 32, output_lim=[70, 70], is_deconv=self.is_deconv)

        self.dc1_final = ConvBlock_Tanh(32, self.n_classes)

        # 全连接神经网络
        self.layer_sizes_t = layer_sizes_t
        self.trunk = Smooth_Net(self.layer_sizes_t)




    def forward(self, x, source):

        compress_seis = self.pre_seis_conv(x)

        down3 = self.down3(compress_seis)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        center = self.center(down5)

        decoder1_image = center     # (None, 512, 5, 5)

        s = self.trunk(source).reshape(-1, 512, 1, 1)   # (None, 512, 1, 1)

        X = decoder1_image * s  # (None, 512, 5, 5)

        #################
        ###  Decoder1 ###
        #################
        dc1_up5 = self.up5(down5, X)
        dc1_up4 = self.up4(down4, dc1_up5)
        dc1_up3 = self.up3(dc1_up4)
        dc1_up2 = self.up2(dc1_up3)

        return self.dc1_final(dc1_up2)



# Fourier-deeponet
class FourierDeeponet(nn.Module):
    def __init__(self, **kwargs):
        super(FourierDeeponet, self).__init__()
        # 全连接神经网络
        # two linear transformations to increase the number of channels to C = 64
        self.layer_sizes_b = [5, 32, 64]
        self.branch = Smooth_Net(self.layer_sizes_b)

        self.layer_sizes_t = [5, 32, 64]
        self.trunk = Smooth_Net(self.layer_sizes_t)
        self.FNO_Decoder = fno_model.UFNO2d(modes1=16, modes2=16, width=64)


    def forward(self, data, source):

        data = F.pad(data, [1, 1])  # (None, 5 ,1000, 72)
        # data = F.interpolate(data, size=(1000, 72), mode='bilinear', align_corners=False)  # (None, 5 ,1000, 72)
        data = data.permute(0, 2, 3, 1)  # (None, 1000, 72, 5)

        b = self.branch(data)  # (None, 1000, 72, 64)
        t = self.trunk(source).reshape(-1, 1, 1, 64)  # (None, 64)
        t = t.repeat(1, 1000, 72, 1)

        out = b * t  # (None, 1000, 72, 64)


        out = self.FNO_Decoder(out)

        return out








model_dict = {
    'InversionNet': InversionNet,
    'Discriminator': Discriminator,
    'UPFWI': FCN4_Deep_Resize_2,
    'FWIDeeponet': FWIDeeponet,
    'FWIDeeponet2': FWIDeeponet2,
    'FWIDeeponet3': FWIDeeponet3,
    'FWIEnDeepOnet': FWIEnDeepOnet,
    'DDeeponet': DDeeponet,
    'DDeeponet2': DDeeponet2,
    'DDNet70Model': DDNet70Model,
    'InversionDeepOnet': InversionDeepOnet,
    'InversionDeepOnet2': InversionDeepOnet2,
    'DDNetDeepOnet': DDNetDeepOnet,
    'FourierDeeponet': FourierDeeponet,
    'DeepONet': DeepONet
}

