import os, sys
import math

import torch
import torch.nn as nn

from torchsummary import summary

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1, stride=1, padding=0, groups=1):
        super(LinearBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, groups=groups, stride=stride,
                padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x

class ConvBNPReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1, stride=1, padding=0, groups=1):
        super(ConvBNPReLU, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, groups=groups, stride=stride,
                padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)

        return x

class DepthWiseBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2, padding=1, groups=1):
        super(DepthWiseBlock, self).__init__()
        self.p_conv = ConvBNPReLU(in_c, groups, 1, 1, 0)
        self.d_conv = ConvBNPReLU(groups, groups, kernel_size, stride=stride,
                                    padding=padding)
        self.project = LinearBlock(groups, out_c, 1, 1, 0)
        if in_c == out_c  and stride == 1:
            self.residual = True
        else:
            self.residual = False

    def forward(self, x):
        x_ = x
        x = self.p_conv(x)
        x = self.d_conv(x)
        x = self.project(x)
        if self.residual:
            output = x_ + x
        else:
            output = x
        return output

class ResidualBlock(nn.Module):
    def __init__(self, c, num_blocks, groups, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        modules = []
        for _ in range(num_blocks):
            modules.append(DepthWiseBlock(c, c, kernel_size, stride, padding, groups))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

def l2_norm(x, axis=1):
    norm = torch.norm(x, p=2, dim=axis, keepdim=True)
    x = torch.div(x, norm)
    return x


class MobileFaceNet(nn.Module):
    def __init__(self, feature_dim=512):
        super(MobileFaceNet, self).__init__()
        self.conv_1 = ConvBNPReLU(3, 64, 3, 2, 1)
        self.conv_2_dw = ConvBNPReLU(64, 64, 3, 1, 1, groups=64)
        self.conv_23 = DepthWiseBlock(64, 64, 3, 2, 1, groups=128)
        self.conv_3 = ResidualBlock(64, 4, 128, 3, 1, 1)
        self.conv_34 = DepthWiseBlock(64, 128, 3, 2, 1, groups=256)
        self.conv_4 = ResidualBlock(128, 6, 256, 3, 1, 1)
        self.conv_45 = DepthWiseBlock(128, 128, 3, 2, 1, groups=512)
        self.conv_5 = ResidualBlock(128, 2, 256, 3, 1, 1)
        self.conv_6_sep = ConvBNPReLU(128, 512, 1, 1, 0)
        self.conv_6_dw = LinearBlock(512, 512, 7, 1, 0, groups=512)
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, feature_dim, bias=False)
        self.bn = nn.BatchNorm1d(feature_dim)
    
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return l2_norm(x)*20
        # return x

if __name__ == "__main__":
    net = MobileFaceNet()
    torch.save(net.state_dict(), 'tmp.pth')
    summary(net.cuda(), (3, 112, 112))

