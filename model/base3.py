"""
  @Time    : 2019-1-13 17:42
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : base3.py
  @Function: 
  
"""
import torch
import torch.nn.functional as F
from torch import nn

from backbone.resnext.resnext101_regular import ResNeXt101


###################################################################
# ########################## CBAM #################################
###################################################################
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
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
        channel_att_sum = True
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


###################################################################
# ########################## CCL ##################################
###################################################################
class CCL(nn.Module):
    def __init__(self, in_planes, out_planes, rate):
        super(CCL, self).__init__()
        self.input_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.output_conv = nn.Conv2d(out_planes * 4, out_planes, kernel_size=1, stride=1, padding=0)
        self.local = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=rate, dilation=rate)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_conv(x)
        local_1 = self.local(x)
        context_1 = self.context(x)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn(ccl_1)
        ccl_1 = self.relu(ccl_1)

        local_2 = self.local(ccl_1)
        context_2 = self.context(ccl_1)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn(ccl_2)
        ccl_2 = self.relu(ccl_2)

        local_3 = self.local(ccl_2)
        context_3 = self.context(ccl_2)
        ccl_3 = local_3 - context_3
        ccl_3 = self.bn(ccl_3)
        ccl_3 = self.relu(ccl_3)

        local_4 = self.local(ccl_3)
        context_4 = self.context(ccl_3)
        ccl_4 = local_4 - context_4
        ccl_4 = self.bn(ccl_4)
        ccl_4 = self.relu(ccl_4)

        ccl_fusion = torch.cat((ccl_1, ccl_2, ccl_3, ccl_4), 1)
        output = self.output_conv(ccl_fusion)

        return output


###################################################################
# ########################## MHY ##################################
###################################################################
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


###################################################################
# ########################## NETWORK ##############################
###################################################################
class BASE3(nn.Module):
    def __init__(self):
        super(BASE3, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.ccl_4 = CCL(2048, 512, 2)
        self.ccl_3 = CCL(1024, 256, 3)
        self.ccl_2 = CCL(512, 128, 4)
        self.ccl_1 = CCL(256, 64, 5)

        self.sc_4 = nn.Conv2d(2048, 512, 1)
        self.sc_3 = nn.Conv2d(1024, 256, 1)
        self.sc_2 = nn.Conv2d(512, 128, 1)
        self.sc_1 = nn.Conv2d(256, 64, 1)

        self.layer4_fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512), nn.ReLU()
        )
        self.layer3_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.layer2_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.layer1_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.up_4 = nn.Sequential(nn.ConvTranspose2d(512, 64, 16, 8, 4), nn.BatchNorm2d(64), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(256, 64, 8, 4, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Conv2d(64, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU())

        self.cbam_4 = CBAM(64)
        self.cbam_3 = CBAM(64)
        self.cbam_2 = CBAM(64)
        self.cbam_1 = CBAM(64)
        self.cbam_fusion = CBAM(256)

        self.layer4_predict = Predict(64)
        self.layer3_predict = Predict(64)
        self.layer2_predict = Predict(64)
        self.layer1_predict = Predict(64)

        self.fusion_predict = Predict(256)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4_ccl = self.ccl_4(layer4)
        layer4_sc = self.sc_4(layer4)
        layer4_concat = torch.cat((layer4_sc, layer4_ccl), 1)
        layer4_fusion = self.layer4_fusion(layer4_concat)

        layer3_ccl = self.ccl_3(layer3)
        layer3_sc = self.sc_3(layer3)
        layer3_concat = torch.cat((layer3_sc, layer3_ccl), 1)
        layer3_fusion = self.layer3_fusion(layer3_concat)

        layer2_ccl = self.ccl_2(layer2)
        layer2_sc = self.sc_2(layer2)
        layer2_concat = torch.cat((layer2_sc, layer2_ccl), 1)
        layer2_fusion = self.layer2_fusion(layer2_concat)

        layer1_ccl = self.ccl_1(layer1)
        layer1_sc = self.sc_1(layer1)
        layer1_concat = torch.cat((layer1_sc, layer1_ccl), 1)
        layer1_fusion = self.layer1_fusion(layer1_concat)

        layer4_feature = self.up_4(layer4_fusion)
        layer4_feature = self.cbam_4(layer4_feature)

        layer3_feature = self.up_3(layer3_fusion)
        layer3_feature = self.cbam_3(layer3_feature)

        layer2_feature = self.up_2(layer2_fusion)
        layer2_feature = self.cbam_2(layer2_feature)

        layer1_feature = self.up_1(layer1_fusion)
        layer1_feature = self.cbam_1(layer1_feature)

        fusion_feature = torch.cat((layer1_feature, layer2_feature, layer3_feature, layer4_feature), 1)
        fusion_feature = self.cbam_fusion(fusion_feature)

        layer4_predict = self.layer4_predict(layer4_feature)
        layer3_predict = self.layer3_predict(layer3_feature)
        layer2_predict = self.layer2_predict(layer2_feature)
        layer1_predict = self.layer1_predict(layer1_feature)

        fusion_predict = self.fusion_predict(fusion_feature)

        layer4_predict = F.upsample(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        fusion_predict = F.upsample(fusion_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
                   F.sigmoid(layer1_predict), F.sigmoid(fusion_predict)

        return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
               F.sigmoid(layer1_predict), F.sigmoid(fusion_predict)
