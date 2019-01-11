"""
  @Time    : 2019-1-6 18:35
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : edge_cbam_x.py
  @Function: 
  
"""

import torch
import torch.nn.functional as F
from torch import nn

from backbone.resnext.resnext101_regular import ResNeXt101


###################################################################
###################################################################
###################################################################
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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

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
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


####################################################################
####################################################################
####################################################################
####################################################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Up(nn.Module):

    def __init__(self, in_planes, out_planes, scin_planes, scout_planes):
        super(Up, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.scin_planes = scin_planes
        self.scout_planes = scout_planes
        self.residual = nn.ConvTranspose2d(self.in_planes, self.out_planes, 4, stride=2, padding=1)
        self.skip_connection = conv1x1(self.scin_planes, self.scout_planes)
        self.bn = nn.BatchNorm2d(self.out_planes)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        residual = self.residual(x)
        skip_connection = self.skip_connection(y)
        output = residual + skip_connection
        output = self.bn(output)
        output = self.relu(output)

        return output


class Predict(nn.Module):

    def __init__(self, in_planes):
        super(Predict, self).__init__()
        self.in_planes = in_planes
        self.out_planes = int(in_planes / 8)
        self.predict = nn.Sequential(
            nn.Conv2d(in_planes, self.out_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_planes),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.out_planes, 1, 1)
        )

    def forward(self, x):
        output = self.predict(x)

        return output


class Fuse(nn.Module):

    def __init__(self):
        super(Fuse, self).__init__()
        self.conv = nn.Conv2d(128, 1, 3, 1, padding=1)
        self.cbam = CBAM(128)

    def forward(self, f, b, e):
        f_feature = torch.cat([f, e], 1)
        f_feature = self.cbam(f_feature)
        b_feature = torch.cat([b, e], 1)
        b_feature = self.cbam(b_feature)

        f = self.conv(f_feature)
        b = self.conv(b_feature)
        fuse = f - b

        return fuse


class EDGE_CBAM_X(nn.Module):
    def __init__(self, backbone_path=None):
        super(EDGE_CBAM_X, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.cbam512 = CBAM(512)
        self.cbam256 = CBAM(256)
        self.cbam128 = CBAM(128)
        self.cbam64 = CBAM(64)

        self.f3 = Up(2048, 512, 1024, 512)
        self.f2 = Up(512, 256, 512, 256)
        self.f1 = Up(256, 128, 256, 128)
        self.f0 = Up(128, 64, 64, 64)

        self.b3 = Up(2048, 512, 1024, 512)
        self.b2 = Up(512, 256, 512, 256)
        self.b1 = Up(256, 128, 256, 128)
        self.b0 = Up(128, 64, 64, 64)

        self.e3 = Up(2048, 512, 1024, 512)
        self.e2 = Up(512, 256, 512, 256)
        self.e1 = Up(256, 128, 256, 128)
        self.e0 = Up(128, 64, 64, 64)

        self.predict_f3 = Predict(512)
        self.predict_f2 = Predict(256)
        self.predict_f1 = Predict(128)
        self.predict_f0 = Predict(64)

        self.predict_b3 = Predict(512)
        self.predict_b2 = Predict(256)
        self.predict_b1 = Predict(128)
        self.predict_b0 = Predict(64)

        self.predict_edge = nn.Conv2d(64, 1, 3, 1, padding=1)

        self.predict_fb = Fuse()

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        f3 = self.f3(layer4, layer3)
        f3 = self.cbam512(f3)
        f2 = self.f2(f3, layer2)
        f2 = self.cbam256(f2)
        f1 = self.f1(f2, layer1)
        f1 = self.cbam128(f1)
        f0 = self.f0(f1, layer0)
        f0 = self.cbam64(f0)

        b3 = self.b3(layer4, layer3)
        b3 = self.cbam512(b3)
        b2 = self.b2(b3, layer2)
        b2 = self.cbam256(b2)
        b1 = self.b1(b2, layer1)
        b1 = self.cbam128(b1)
        b0 = self.b0(b1, layer0)
        b0 = self.cbam64(b0)

        e3 = self.e3(layer4, layer3)
        e2 = self.e2(e3, layer2)
        e1 = self.e1(e2, layer1)
        e0 = self.e0(e1, layer0)

        predict_e = self.predict_edge(e0)

        predict_f3 = self.predict_f3(f3)
        predict_f2 = self.predict_f2(f2)
        predict_f1 = self.predict_f1(f1)
        predict_f0 = self.predict_f0(f0)

        predict_b3 = self.predict_b3(b3)
        predict_b2 = self.predict_b2(b2)
        predict_b1 = self.predict_b1(b1)
        predict_b0 = self.predict_b0(b0)

        predict_fb = self.predict_fb(f0, b0, e0)

        predict_f3 = F.upsample(predict_f3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f2 = F.upsample(predict_f2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f1 = F.upsample(predict_f1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f0 = F.upsample(predict_f0, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict_b3 = F.upsample(predict_b3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b2 = F.upsample(predict_b2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b1 = F.upsample(predict_b1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b0 = F.upsample(predict_b0, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict_e = F.upsample(predict_e, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict_fb = F.upsample(predict_fb, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict_f3, predict_f2, predict_f1, predict_f0, \
                   predict_b3, predict_b2, predict_b1, predict_b0, \
                   predict_e, predict_fb
        return torch.sigmoid(predict_e), torch.sigmoid(predict_fb)