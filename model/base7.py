"""
  @Time    : 2019-1-19 03:08
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : base7.py
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


###################################################################
# ########################## NETWORK ##############################
###################################################################
class BASE7(nn.Module):
    def __init__(self, backbone_path=None):
        super(BASE7, self).__init__()
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        # Mirror Region Stream
        self.f_ccl_4 = CCL(2048, 512, 2)
        self.f_ccl_3 = CCL(1024, 256, 3)
        self.f_ccl_2 = CCL(512, 128, 4)
        self.f_ccl_1 = CCL(256, 64, 5)
        self.f_ccl_0 = CCL(64, 32, 6)

        self.f_fixed_4 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(), nn.ConvTranspose2d(512, 64, 16, 8, 4),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.f_fixed_3 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(), nn.ConvTranspose2d(256, 64, 8, 4, 2),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.f_fixed_2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.f_fixed_1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.f_fixed_0 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.f_attention = CBAM(320)

        self.f_predict = nn.Conv2d(320, 1, 3, 1, 1)

        # Non-Mirror Region Stream
        self.b_ccl_4 = CCL(2048, 512, 2)
        self.b_ccl_3 = CCL(1024, 256, 3)
        self.b_ccl_2 = CCL(512, 128, 4)
        self.b_ccl_1 = CCL(256, 64, 5)
        self.b_ccl_0 = CCL(64, 32, 6)

        self.b_fixed_4 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(), nn.ConvTranspose2d(512, 64, 16, 8, 4),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.b_fixed_3 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(), nn.ConvTranspose2d(256, 64, 8, 4, 2),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.b_fixed_2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(), nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.b_fixed_1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, 1, 1),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.b_fixed_0 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1),
                                       nn.BatchNorm2d(64), nn.ReLU())
        self.b_attention = CBAM(320)

        self.b_predict = nn.Conv2d(320, 1, 3, 1, 1)

        # Feature Mosaic
        self.output_predict = nn.Conv2d(320, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Mirror Region Stream
        f_ccl_4 = self.f_ccl_4(layer4)
        f_ccl_3 = self.f_ccl_3(layer3)
        f_ccl_2 = self.f_ccl_2(layer2)
        f_ccl_1 = self.f_ccl_1(layer1)
        f_ccl_0 = self.f_ccl_0(layer0)

        f_fixed_4 = self.f_fixed_4(f_ccl_4)
        f_fixed_3 = self.f_fixed_3(f_ccl_3)
        f_fixed_2 = self.f_fixed_2(f_ccl_2)
        f_fixed_1 = self.f_fixed_1(f_ccl_1)
        f_fixed_0 = self.f_fixed_0(f_ccl_0)

        f_feature = torch.cat((f_fixed_4, f_fixed_3, f_fixed_2, f_fixed_1, f_fixed_0), 1)

        f_attention = self.f_attention(f_feature)

        f_predict = self.f_predict(f_attention)

        # Boundary Stream
        b_ccl_4 = self.b_ccl_4(layer4)
        b_ccl_3 = self.b_ccl_3(layer3)
        b_ccl_2 = self.b_ccl_2(layer2)
        b_ccl_1 = self.b_ccl_1(layer1)
        b_ccl_0 = self.b_ccl_0(layer0)

        b_fixed_4 = self.b_fixed_4(b_ccl_4)
        b_fixed_3 = self.b_fixed_3(b_ccl_3)
        b_fixed_2 = self.b_fixed_2(b_ccl_2)
        b_fixed_1 = self.b_fixed_1(b_ccl_1)
        b_fixed_0 = self.b_fixed_0(b_ccl_0)

        b_feature = torch.cat((b_fixed_4, b_fixed_3, b_fixed_2, b_fixed_1, b_fixed_0), 1)

        b_attention = self.b_attention(b_feature)

        b_predict = self.b_predict(b_attention)

        # Feature Mosaic
        output_feature = f_attention.mul(torch.sigmoid(f_predict)).mul(1 - torch.sigmoid(b_predict)) - \
                         b_attention.mul(torch.sigmoid(b_predict)).mul(1 - torch.sigmoid(f_predict))

        output_predict = self.output_predict(output_feature)

        # Upsample
        f_predict = F.interpolate(f_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        b_predict = F.interpolate(b_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        output_predict = F.interpolate(output_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return f_predict, b_predict, output_predict

        return torch.sigmoid(f_predict), torch.sigmoid(b_predict), torch.sigmoid(output_predict)
