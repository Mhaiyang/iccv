"""
  @Time    : 2019-1-9 04:41
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : dsc.py
  @Function: 
  
"""
import torch
import torch.nn.functional as F
from torch import nn

from backbone.resnext.resnext101_regular import ResNeXt101


# Module Function
def upsample(in_planes, out_planes, kernel_size, stride, group, bias):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride=stride, groups=group, bias=bias)


# Module Class
class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x





# Network Class
class DSC(nn.Module):
    def __init__(self):
        super(DSC, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.later4_conv1 = LayerConv(2048, 1024, 7, 1, 3, True)
        self.later4_conv2 = LayerConv(1024, 1024, 7, 1, 3, True)
        self.later4_conv3 = LayerConv(1024, 32, 1, 1, 0, False)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        if self.training:
            return a
        return b