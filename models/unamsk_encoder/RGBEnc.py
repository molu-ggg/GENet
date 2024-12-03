import math
import torch
import torch.nn as nn
from models.unamsk_encoder.van import  Block

from models.STG_NF.modules_pose import (
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    SqueezeLayer,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)
from models.STG_NF.utils import split_feature
from models.STG_NF.graph import Graph
from models.STG_NF.stgcn import st_gcn


class stu_stgcn(nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels,
              temporal_kernel_size=9, spatial_kernel_size=2, first=False):
        super(stu_stgcn, self).__init__()
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.block = st_gcn(in_channels, out_channels, kernel_size, 1),

    def forward(self,x):
        return self.block(x)

# class RGB_Encoder(nn.Module):
#
#     def __init__(self,
#                  pose_shape,
#                  hidden_channels,
#                  K,
#                  L,
#                  actnorm_scale,
#                  flow_permutation,
#                  flow_coupling,
#                  LU_decomposed,
#                  learn_top,
#                  device,
#                  R=0,
#                  edge_importance=False,
#                  temporal_kernel_size=None,
#                  strategy='uniform',
#                  max_hops=8,
#
#                  layer_nums=4):
#         super().__init__()
#         self.device = device
#         self.flow_coupling = flow_coupling
#         print("layer_nums",layer_nums)
#         g = Graph(strategy=strategy, max_hop=max_hops)
#         self.A = torch.from_numpy(g.A).float().to(device)
#         self.res = list()
#         self.layer_nums = layer_nums
#
#         self.block = list()
#
#         C,T,V = pose_shape
#         self.in_channels =C
#         if temporal_kernel_size is None: ### temporal_kernel_size 为什么这样设置呢？有什么更好的办法吗？设置其他的值试一试，比如Kernel  = 3
#             temporal_kernel_size = T // 2 +  1
#         for i in range(self.layer_nums):
#             # spatial_kernel_size = 2
#             spatial_kernel_size = self.A.size(0)
#             kernel_size = (temporal_kernel_size, spatial_kernel_size)
#             self.block.append(st_gcn(self.in_channels, self.in_channels, kernel_size, 1))
#             # self.block.append(stu_stgcn(self.in_channels,  hidden_channels,self.in_channels,
#             #                        temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=self.A.size(0),
#             #                       first=False))
#         self.res = nn.ModuleList(self.block)
#         self.act = nn.LeakyReLU()
#     def forward(self,x,label, score,logdet=0):
#         h = x
#         for i in range(self.layer_nums):
#             h,_ = self.res[i](h,self.A)
#         logdet = torch.sum(torch.log(h+0.00001), dim=[1, 2, 3]) + logdet
#         #print(logdet)
#
#         return h,logdet


class res_block3D(nn.Module):
    def __init__(self, channels):
        super(res_block3D, self).__init__()
        self.l1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.l2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm3d(channels)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        inp = x
        x = self.l1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.l2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = x + inp
        return x


# class res_block3D(nn.Module):
#     def __init__(self, channels):
#         super(res_block3D, self).__init__()
#         self.l1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
#         self.act = nn.LeakyReLU()
#         self.bn1 = nn.BatchNorm3d(channels)
#         self.van = Block(channels)
#
#     def forward(self, x):
#         inp = x
#         x = self.van(x)
#         x = self.l1(x)
#         x = self.bn1(x)
#         x = self.act(x)
#         x = x + inp
#         return x



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, op="none", scale_factor=1, kernel_size=3, stride=1, padding=1,
                 output_padding=0):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.op = op
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, 1), )
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                                         padding=padding,
                                         output_padding=output_padding)
        assert op in ["concat", "none"]

        if op == "concat":
            self.conv = double_conv(out_ch * 2, out_ch)
        else:
            self.conv = double_conv(out_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if self.op == "concat":
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2),
            # double_conv(in_ch, out_ch)

            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            double_conv(out_ch, out_ch),

        )

    def forward(self, x):
        x = self.mpconv(x)
        return x



class RGB_Encoder(nn.Module):
    def __init__(self, inp_feat,n_blocks=4):
        super(RGB_Encoder, self).__init__()
        self.conv1 = nn.Conv3d(inp_feat, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(256, 128, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm3d(128)
        self.res = list()
        for _ in range(n_blocks):
            self.res.append(res_block3D(128)) ### 128,3,3,3 (384,3,3)

        self.conv4 = nn.Conv3d(128, 64, kernel_size=1, padding=0)
        self.conv5 = nn.Conv3d(64, 32, kernel_size=1, padding=0)
        self.res = nn.ModuleList(self.res)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.act(x)
        for i in range(len(self.res)):
            x = self.res[i](x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x





