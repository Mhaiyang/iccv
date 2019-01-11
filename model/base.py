"""
  @Time    : 2019-1-11 04:30
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : base.py
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
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

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
    def __init__(self, in_planes, out_planes):
        super(CCL, self).__init__()
        self.input_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.output_conv = nn.Conv2d(out_planes * 4, out_planes, kernel_size=1, stride=1, padding=0)
        self.local = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=5, dilation=5)
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
        output = self.bn(output)
        output = self.relu(output)

        return output


###################################################################
# ########################## MHY ##################################
###################################################################
class Activation(nn.Module):
    def __init__(self, planes):
        super(Activation, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)

        return x


###################################################################
# ########################## NETWORK ##############################
###################################################################
class BASE(nn.Module):
    def __init__(self):
        super(BASE, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.ccl = CCL(2048, 512)

        self.cbam_4 = CBAM(512)
        self.cbam_3 = CBAM(512)
        self.cbam_2 = CBAM(256)
        self.cbam_1 = CBAM(256)
        self.cbam_0 = CBAM(128)

        self.up_4 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
        self.up_3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.up_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)

        self.sc_3 = nn.Conv2d(1024, 512, 1)
        self.sc_2 = nn.Conv2d(512, 256, 1)
        self.sc_1 = nn.Conv2d(256, 256, 1)
        self.sc_0 = nn.Conv2d(64, 128, 1)

        self.activation_3 = Activation(512)
        self.activation_2 = Activation(256)
        self.activation_1 = Activation(256)
        self.activation_0 = Activation(128)

        self.layer4_feature = nn.Sequential(
            nn.ConvTranspose2d(512, 32, 64, 32, 16),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3_feature = nn.Sequential(
            nn.ConvTranspose2d(512, 32, 32, 16, 8),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2_feature = nn.Sequential(
            nn.ConvTranspose2d(256, 32, 16, 8, 4),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer1_feature = nn.Sequential(
            nn.ConvTranspose2d(256, 32, 8, 4, 2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer0_feature = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer4_predict = nn.Conv2d(32, 1, 1)
        self.layer3_predict = nn.Conv2d(32, 1, 1)
        self.layer2_predict = nn.Conv2d(32, 1, 1)
        self.layer1_predict = nn.Conv2d(32, 1, 1)
        self.layer0_predict = nn.Conv2d(32, 1, 1)

        self.global_predict = nn.Conv2d(160, 1, 1)

        self.fusion_predict = nn.Conv2d(5, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4_decoder = self.ccl(layer4)
        layer4_decoder = self.cbam_4(layer4_decoder)

        layer3_decoder = self.up_4(layer4_decoder)
        layer3_sc = self.sc_3(layer3)
        layer3_decoder = self.activation_3(layer3_decoder + layer3_sc)
        layer3_decoder = self.cbam_3(layer3_decoder)

        layer2_decoder = self.up_3(layer3_decoder)
        layer2_sc = self.sc_2(layer2)
        layer2_decoder = self.activation_2(layer2_decoder + layer2_sc)
        layer2_decoder = self.cbam_2(layer2_decoder)

        layer1_decoder = self.up_2(layer2_decoder)
        layer1_sc = self.sc_1(layer1)
        layer1_decoder = self.activation_1(layer1_decoder + layer1_sc)
        layer1_decoder = self.cbam_1(layer1_decoder)

        layer0_decoder = self.up_1(layer1_decoder)
        layer0_sc = self.sc_0(layer0)
        layer0_decoder = self.activation_0(layer0_decoder + layer0_sc)
        layer0_decoder = self.cbam_0(layer0_decoder)

        layer4_feature = self.layer4_feature(layer4_decoder)
        layer3_feature = self.layer3_feature(layer3_decoder)
        layer2_feature = self.layer2_feature(layer2_decoder)
        layer1_feature = self.layer1_feature(layer1_decoder)
        layer0_feature = self.layer0_feature(layer0_decoder)

        global_feature = torch.cat((layer0_feature, layer1_feature, layer2_feature, layer3_feature, layer4_feature), 1)

        layer4_predict = self.layer4_predict(layer4_feature)
        layer4_predict = F.sigmoid(layer4_predict)
        layer3_predict = self.layer3_predict(layer3_feature)
        layer3_predict = F.sigmoid(layer3_predict)
        layer2_predict = self.layer2_predict(layer2_feature)
        layer2_predict = F.sigmoid(layer2_predict)
        layer1_predict = self.layer1_predict(layer1_feature)
        layer1_predict = F.sigmoid(layer1_predict)
        layer0_predict = self.layer0_predict(layer0_feature)
        layer0_predict = F.sigmoid(layer0_predict)

        global_predict = self.global_predict(global_feature)
        global_predict = F.sigmoid(global_predict)

        fusion_feature = torch.cat((layer0_predict, layer1_predict, layer2_predict, layer3_predict, layer4_predict), 1)

        fusion_predict = self.fusion_predict(fusion_feature)
        fusion_predict = F.sigmoid(fusion_predict)

        return layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_predict, \
               global_predict, fusion_predict

