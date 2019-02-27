"""
 @Time    : 201/23/19 16:26
 @Author  : TaylorMei
 @Email   : mhy845879017@gmail.com

 @Project : iccv
 @File    : mhy1.py
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
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg']):
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
        # original
        # return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        # max
        # torch.max(x, 1)[0].unsqueeze(1)
        # avg
        return torch.mean(x, 1).unsqueeze(1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(1, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels=128, reduction_ratio=16, pool_types=['avg'], no_spatial=False):
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
    def __init__(self, planes, rate1=3, rate2=5):
        super(CCL, self).__init__()
        self.local_1a = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, dilation=1)
        self.local_1b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1a = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=rate1, dilation=rate1)
        self.context_1b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=rate2, dilation=rate2)

        self.local_2a = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, dilation=1)
        self.local_2b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2a = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=rate1, dilation=rate1)
        self.context_2b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=rate2, dilation=rate2)

        self.local_3a = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, dilation=1)
        self.local_3b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_3a = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=rate1, dilation=rate1)
        self.context_3b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=rate2, dilation=rate2)

        self.local_4a = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, dilation=1)
        self.local_4b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_4a = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=rate1, dilation=rate1)
        self.context_4b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=rate2, dilation=rate2)

        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.cbam = CBAM(4 * planes)

    def forward(self, x):
        local_1 = self.local_1a(x) + self.local_1b(x)
        context_1 = self.context_1a(x) + self.context_1b(x)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn(ccl_1)
        ccl_1 = self.relu(ccl_1)

        local_2 = self.local_2a(ccl_1) + self.local_2b(ccl_1)
        context_2 = self.context_2a(ccl_1) + self.context_2b(ccl_1)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn(ccl_2)
        ccl_2 = self.relu(ccl_2)

        local_3 = self.local_3a(ccl_2) + self.local_3b(ccl_2)
        context_3 = self.context_3a(ccl_2) + self.context_3b(ccl_2)
        ccl_3 = local_3 - context_3
        ccl_3 = self.bn(ccl_3)
        ccl_3 = self.relu(ccl_3)

        local_4 = self.local_4a(ccl_3) + self.local_4b(ccl_3)
        context_4 = self.context_4a(ccl_3) + self.context_4b(ccl_3)
        ccl_4 = local_4 - context_4
        ccl_4 = self.bn(ccl_4)
        ccl_4 = self.relu(ccl_4)

        output = self.cbam(torch.cat((ccl_1, ccl_2, ccl_3, ccl_4), 1))

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
class MHY2(nn.Module):
    def __init__(self, backbone_path=None):
        super(MHY2, self).__init__()
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.conv_4 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.conv_1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.ccl_4 = CCL(512)
        self.ccl_3 = CCL(256)
        self.ccl_2 = CCL(128)
        self.ccl_1 = CCL(64)

        self.up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 64, 16, 8, 4), nn.BatchNorm2d(64), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 64, 8, 4, 2), nn.BatchNorm2d(64), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(512, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU())

        self.cbam_4 = CBAM(64)
        self.cbam_3 = CBAM(64)
        self.cbam_2 = CBAM(64)
        self.cbam_1 = CBAM(64)
        self.cbam_fusion = CBAM(256)

        self.layer4_predict = Predict(64)
        self.layer3_predict = Predict(64)
        self.layer2_predict = Predict(64)
        self.layer1_predict = Predict(64)

        self.fusion_predict = nn.Conv2d(256, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        conv_4 = self.conv_4(layer4)
        conv_3 = self.conv_3(layer3)
        conv_2 = self.conv_2(layer2)
        conv_1 = self.conv_1(layer1)

        ccl_4 = self.ccl_4(conv_4)
        ccl_3 = self.ccl_3(conv_3)
        ccl_2 = self.ccl_2(conv_2)
        ccl_1 = self.ccl_1(conv_1)

        up_4 = self.up_4(ccl_4)
        up_3 = self.up_3(ccl_3)
        up_2 = self.up_2(ccl_2)
        up_1 = self.up_1(ccl_1)

        cbam_4 = self.cbam_4(up_4)
        cbam_3 = self.cbam_3(up_3)
        cbam_2 = self.cbam_2(up_2)
        cbam_1 = self.cbam_1(up_1)

        fusion = torch.cat((cbam_1, cbam_2, cbam_3, cbam_4), 1)
        cbam_fusion = self.cbam_fusion(fusion)

        layer4_predict = self.layer4_predict(cbam_4)
        layer3_predict = self.layer3_predict(cbam_3)
        layer2_predict = self.layer2_predict(cbam_2)
        layer1_predict = self.layer1_predict(cbam_1)

        fusion_predict = self.fusion_predict(cbam_fusion)

        layer4_predict = F.upsample(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        fusion_predict = F.upsample(fusion_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return layer4_predict, layer3_predict, layer2_predict, layer1_predict, fusion_predict

        return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
               F.sigmoid(layer1_predict), F.sigmoid(fusion_predict)
