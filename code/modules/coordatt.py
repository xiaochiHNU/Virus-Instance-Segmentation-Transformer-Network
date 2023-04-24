import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return scale

class CoordAttplus(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttplus, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.SpatialGate = SpatialGate()

    def forward(self, x):
        identity = x   #  1, 256, 112, 112

        # 在CoordAtt基础上，加上空间attention
        n,c,h,w = x.size()
        x_h = self.pool_h(x)                       # 1, 256, 112, 1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # 1, 256, 112, 1 转置之后

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)        # 1, 8 ,224, 1
        
        x_h, x_w = torch.split(y, [h, w], dim=2)  # x_h  1, 8, 112 ,1  x_w 1, 8 ,112 ,1
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()     # x_h  1, 256, 112, 1
        a_w = self.conv_w(x_w).sigmoid()     # x_w  1, 256, 1, 112
                                             # x_s  1, 1, 112, 112
                                             # x_c  1, 256 , 1,  1

        Spatial_att = self.SpatialGate(identity)  # 加入空间attention  x_s  1, 1, 112, 112
        out = identity * a_w * a_h * Spatial_att          # 1, 256, 112 ,112
        return out

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x  # 1, 256, 112, 112

        # 在CoordAtt基础上，加上通道attention
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # 1, 256, 112, 1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 1, 256, 112, 1 转置之后

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)  # 1, 8 ,224, 1

        x_h, x_w = torch.split(y, [h, w], dim=2)  # x_h  1, 8, 112 ,1  x_w 1, 8 ,112 ,1
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()  # x_h  1, 256, 112, 1
        a_w = self.conv_w(x_w).sigmoid()  # x_w  1, 256, 1, 112

        out = identity * a_w * a_h  # 1, 256, 112 ,112

        return out