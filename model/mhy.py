"""
  @Time    : 2019-1-1 22:52
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : mhy.py
  @Function: 
  
"""
import torch
import torch.nn.functional as F
from torch import nn

from backbone.resnet import ResNet, Bottleneck


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
        self.conv = conv3x3(64, 1)

    def forward(self, f, b):
        f = self.conv(f)
        b = self.conv(b)
        fuse = f - b

        return fuse


class MHY(nn.Module):
    def __init__(self, backbone_path=None):
        super(MHY, self).__init__()
        resnet101 = ResNet(Bottleneck, [3, 4, 23, 3])
        if backbone_path is not None:
            resnet101.load_state_dict(torch.load(backbone_path))
            print("{}\nLoad Pre-trained Weights Succeed!".format(backbone_path))

        self.layer0 = nn.Sequential(resnet101.conv1, resnet101.bn1, resnet101.relu)
        self.layer1 = nn.Sequential(resnet101.maxpool, resnet101.layer1)
        self.layer2 = resnet101.layer2
        self.layer3 = resnet101.layer3
        self.layer4 = resnet101.layer4

        self.f3 = Up(2048, 512, 1024, 512)
        self.f2 = Up(512, 256, 512, 256)
        self.f1 = Up(256, 128, 256, 128)
        self.f0 = Up(128, 64, 64, 64)

        self.b3 = Up(2048, 512, 1024, 512)
        self.b2 = Up(512, 256, 512, 256)
        self.b1 = Up(256, 128, 256, 128)
        self.b0 = Up(128, 64, 64, 64)

        self.predict_f3 = Predict(512)
        self.predict_f2 = Predict(256)
        self.predict_f1 = Predict(128)
        self.predict_f0 = Predict(64)

        self.predict_b3 = Predict(512)
        self.predict_b2 = Predict(256)
        self.predict_b1 = Predict(128)
        self.predict_b0 = Predict(64)

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
        f2 = self.f2(f3, layer2)
        f1 = self.f1(f2, layer1)
        f0 = self.f0(f1, layer0)

        b3 = self.b3(layer4, layer3)
        b2 = self.b2(b3, layer2)
        b1 = self.b1(b2, layer1)
        b0 = self.b0(b1, layer0)

        predict_f3 = self.predict_f3(f3)
        predict_f2 = self.predict_f2(f2)
        predict_f1 = self.predict_f1(f1)
        predict_f0 = self.predict_f0(f0)

        predict_b3 = self.predict_b3(b3)
        predict_b2 = self.predict_b2(b2)
        predict_b1 = self.predict_b1(b1)
        predict_b0 = self.predict_b0(b0)

        predict_fb = self.predict_fb(f0, b0)

        predict_f3 = F.interpolate(predict_f3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f2 = F.interpolate(predict_f2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f1 = F.interpolate(predict_f1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f0 = F.interpolate(predict_f0, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict_b3 = F.interpolate(predict_b3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b2 = F.interpolate(predict_b2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b1 = F.interpolate(predict_b1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b0 = F.interpolate(predict_b0, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict_fb = F.interpolate(predict_fb, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict_f3, predict_f2, predict_f1, predict_f0, \
                   predict_b3, predict_b2, predict_b1, predict_b0, predict_fb
        return torch.sigmoid(predict_fb)
