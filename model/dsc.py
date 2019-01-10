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


# Module Class
class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class GlobalConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(GlobalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class Predict(nn.Module):
    def __init__(self, in_planes=32, out_planes=1, kernel_size=1):
        super(Predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size)

    def forward(self, x):
        y = self.conv(x)

        return y


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

        self.layer4_conv1 = LayerConv(2048, 1024, 7, 1, 3, True)
        self.layer4_conv2 = LayerConv(1024, 1024, 7, 1, 3, True)
        self.layer4_conv3 = LayerConv(1024, 32, 1, 1, 0, False)

        self.layer3_conv1 = LayerConv(1024, 512, 5, 1, 2, True)
        self.layer3_conv2 = LayerConv(512, 512, 5, 1, 2, True)
        self.layer3_conv3 = LayerConv(512, 32, 1, 1, 0, False)

        self.layer2_conv1 = LayerConv(512, 256, 5, 1, 2, True)
        self.layer2_conv2 = LayerConv(256, 256, 5, 1, 2, True)
        self.layer2_conv3 = LayerConv(256, 32, 1, 1, 0, False)

        self.layer1_conv1 = LayerConv(256, 128, 3, 1, 1, True)
        self.layer1_conv2 = LayerConv(128, 128, 3, 1, 1, True)
        self.layer1_conv3 = LayerConv(128, 32, 1, 1, 0, False)

        self.layer0_conv1 = LayerConv(64, 128, 3, 1, 1, True)
        self.layer0_conv2 = LayerConv(128, 128, 3, 1, 1, True)
        self.layer0_conv3 = LayerConv(128, 32, 1, 1, 0, False)

        self.relu = nn.ReLU()

        self.global_conv = GlobalConv(160, 32, 1, 1, 0, True)

        self.layer4_predict = Predict(32, 1, 1)
        self.layer3_predict_ori = Predict(32, 1, 1)
        self.layer3_predict = Predict(2, 1, 1)
        self.layer2_predict_ori = Predict(32, 1, 1)
        self.layer2_predict = Predict(3, 1, 1)
        self.layer1_predict_ori = Predict(32, 1, 1)
        self.layer1_predict = Predict(4, 1, 1)
        self.layer0_predict_ori = Predict(32, 1, 1)
        self.layer0_predict = Predict(5, 1, 1)
        self.global_predict = Predict(32, 1, 1)
        self.fusion_predict = Predict(6, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4_conv1 = self.layer4_conv1(layer4)
        layer4_conv2 = self.layer4_conv2(layer4_conv1)
        layer4_conv3 = self.layer4_conv3(layer4_conv2)
        layer4_up = F.upsample(layer4_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer4_up = self.relu(layer4_up)

        layer3_conv1 = self.layer3_conv1(layer3)
        layer3_conv2 = self.layer3_conv2(layer3_conv1)
        layer3_conv3 = self.layer3_conv3(layer3_conv2)
        layer3_up = F.upsample(layer3_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_up = self.relu(layer3_up)

        layer2_conv1 = self.layer2_conv1(layer2)
        layer2_conv2 = self.layer2_conv2(layer2_conv1)
        layer2_conv3 = self.layer2_conv3(layer2_conv2)
        layer2_up = F.upsample(layer2_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_up = self.relu(layer2_up)

        layer1_conv1 = self.layer1_conv1(layer1)
        layer1_conv2 = self.layer1_conv2(layer1_conv1)
        layer1_conv3 = self.layer1_conv3(layer1_conv2)
        layer1_up = F.upsample(layer1_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_up = self.relu(layer1_up)

        layer0_conv1 = self.layer0_conv1(layer0)
        layer0_conv2 = self.layer0_conv2(layer0_conv1)
        layer0_conv3 = self.layer0_conv3(layer0_conv2)
        layer0_up = F.upsample(layer0_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer0_up = self.relu(layer0_up)

        global_concat = torch.cat((layer0_up, layer1_up, layer2_up, layer3_up, layer4_up), 1)
        global_conv = self.global_conv(global_concat)

        layer4_predict = self.layer4_predict(layer4_up)

        layer3_predict_ori = self.layer3_predict_ori(layer3_up)
        layer3_concat = torch.cat((layer3_predict_ori, layer4_predict), 1)
        layer3_predict = self.layer3_predict(layer3_concat)

        layer2_predict_ori = self.layer2_predict_ori(layer2_up)
        layer2_concat = torch.cat((layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)
        layer2_predict = self.layer2_predict(layer2_concat)

        layer1_predict_ori = self.layer1_predict_ori(layer1_up)
        layer1_concat = torch.cat((layer1_predict_ori, layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)
        layer1_predict = self.layer1_predict(layer1_concat)

        layer0_predict_ori = self.layer0_predict_ori(layer0_up)
        layer0_concat = torch.cat((layer0_predict_ori, layer1_predict_ori, layer2_predict_ori,
                                   layer3_predict_ori, layer4_predict), 1)
        layer0_predict = self.layer0_predict(layer0_concat)

        global_predict = self.global_predict(global_conv)

        # fusion
        fusion_concat = torch.cat((layer0_predict, layer1_predict, layer2_predict, layer3_predict,
                                   layer4_predict, global_predict), 1)
        fusion_predict = self.fusion_predict(fusion_concat)

        if self.training:
            return layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_predict, \
                   global_predict, fusion_predict
        return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
               F.sigmoid(layer1_predict), F.sigmoid(layer0_predict), F.sigmoid(global_predict), \
               F.sigmoid(fusion_predict)
