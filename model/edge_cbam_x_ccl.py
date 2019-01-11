"""
  @Time    : 2019-1-12 03:06
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : iccv
  @File    : edge_cbam_x_ccl.py
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
class EDGE_CBAM_X_CCL(nn.Module):
    def __init__(self):
        super(EDGE_CBAM_X_CCL, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.ccl_f = CCL(2048, 512, 5)
        self.ccl_e = CCL(2048, 512, 5)
        self.ccl_b = CCL(2048, 512, 5)

        self.f4_up = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.f3_up = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.f2_up = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.e4_up = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.e3_up = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.e2_up = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.b4_up = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.b3_up = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.b2_up = nn.ConvTranspose2d(128, 64, 4, 2, 1)

        self.sc_f4 = nn.Conv2d(2048, 512, 1)
        self.sc_f3 = nn.Conv2d(1024, 256, 1)
        self.sc_f2 = nn.Conv2d(512, 128, 1)
        self.sc_f1 = nn.Conv2d(256, 64, 1)

        self.sc_e4 = nn.Conv2d(2048, 512, 1)
        self.sc_e3 = nn.Conv2d(1024, 256, 1)
        self.sc_e2 = nn.Conv2d(512, 128, 1)
        self.sc_e1 = nn.Conv2d(256, 64, 1)

        self.sc_b4 = nn.Conv2d(2048, 512, 1)
        self.sc_b3 = nn.Conv2d(1024, 256, 1)
        self.sc_b2 = nn.Conv2d(512, 128, 1)
        self.sc_b1 = nn.Conv2d(256, 64, 1)

        self.f4_fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512), nn.ReLU()
        )
        self.f3_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.f2_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.f1_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.e4_activate = nn.Sequential(
            nn.BatchNorm2d(512), nn.ReLU()
        )
        self.e3_active = nn.Sequential(
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.e2_active = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.e1_active = nn.Sequential(
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.b4_fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512), nn.ReLU()
        )
        self.b3_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.b2_fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.b1_fusion = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.cbam_f4 = CBAM(512)
        self.cbam_f3 = CBAM(256)
        self.cbam_f2 = CBAM(128)
        self.cbam_f1 = CBAM(64)
        self.cbam_fe = CBAM(128)

        self.cbam_b4 = CBAM(512)
        self.cbam_b3 = CBAM(256)
        self.cbam_b2 = CBAM(128)
        self.cbam_b1 = CBAM(64)
        self.cbam_be = CBAM(128)

        self.f4_predict = Predict(512)
        self.f3_predict = Predict(256)
        self.f2_predict = Predict(128)
        self.f1_predict = Predict(64)
        self.fe_predict = Predict(64)

        self.b4_predict = Predict(512)
        self.b3_predict = Predict(256)
        self.b2_predict = Predict(128)
        self.b1_predict = Predict(64)
        self.be_predict = Predict(64)

        self.e_predict = Predict(64)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # f branch
        f4_ccl = self.ccl_f(layer4)
        f4_sc = self.sc_f4(layer4)
        f4_concat = torch.cat((f4_ccl, f4_sc), 1)
        f4_fusion = self.f4_fusion(f4_concat)
        f4_attention = self.cbam_f4(f4_fusion)

        f4_up = self.f4_up(f4_attention)
        f3_sc = self.sc_f3(layer3)
        f3_concat = torch.cat((f4_up, f3_sc), 1)
        f3_fusion = self.f3_fusion(f3_concat)
        f3_attention = self.cbam_f3(f3_fusion)

        f3_up = self.f3_up(f3_attention)
        f2_sc = self.sc_f2(layer2)
        f2_concat = torch.cat((f3_up, f2_sc), 1)
        f2_fusion = self.f2_fusion(f2_concat)
        f2_attention = self.cbam_f2(f2_fusion)

        f2_up = self.f2_up(f2_attention)
        f1_sc = self.sc_f1(layer1)
        f1_concat = torch.cat((f2_up, f1_sc), 1)
        f1_fusion = self.f1_fusion(f1_concat)
        f1_attention = self.cbam_f1(f1_fusion)

        # e branch
        e4_ccl = self.ccl_e(layer4)
        e4_sc = self.sc_e4(layer4)
        e4 = self.e4_activate(e4_ccl + e4_sc)

        e4_up = self.e4_up(e4)
        e3_sc = self.sc_e3(layer3)
        e3 = self.e3_active(e4_up + e3_sc)

        e3_up = self.e3_up(e3)
        e2_sc = self.sc_e2(layer2)
        e2 = self.e2_active(e3_up + e2_sc)

        e2_up = self.e2_up(e2)
        e1_sc = self.sc_e1(layer1)
        e1 = self.e1_active(e2_up + e1_sc)

        # b branch
        b4_ccl = self.ccl_b(layer4)
        b4_sc = self.sc_b4(layer4)
        b4_concat = torch.cat((b4_ccl, b4_sc), 1)
        b4_fusion = self.b4_fusion(b4_concat)
        b4_attention = self.cbam_b4(b4_fusion)

        b4_up = self.b4_up(b4_attention)
        b3_sc = self.sc_b3(layer3)
        b3_concat = torch.cat((b4_up, b3_sc), 1)
        b3_fusion = self.b3_fusion(b3_concat)
        b3_attention = self.cbam_b3(b3_fusion)

        b3_up = self.b3_up(b3_attention)
        b2_sc = self.sc_b2(layer2)
        b2_concat = torch.cat((b3_up, b2_sc), 1)
        b2_fusion = self.b2_fusion(b2_concat)
        b2_attention = self.cbam_b2(b2_fusion)

        b2_up = self.b2_up(b2_attention)
        b1_sc = self.sc_b1(layer1)
        b1_concat = torch.cat((b2_up, b1_sc), 1)
        b1_fusion = self.b1_fusion(b1_concat)
        b1_attention = self.cbam_b1(b1_fusion)

        # predict
        predict_f4 = self.f4_predict(f4_attention)
        predict_f3 = self.f3_predict(f3_attention)
        predict_f2 = self.f2_predict(f2_attention)
        predict_f1 = self.f1_predict(f1_attention)

        predict_b4 = self.b4_predict(b4_attention)
        predict_b3 = self.b3_predict(b3_attention)
        predict_b2 = self.b2_predict(b2_attention)
        predict_b1 = self.b1_predict(b1_attention)

        predict_e = self.e_predict(e1)

        fe_concat = torch.cat((f1_attention, e1), 1)
        fe_attention = self.cbam_fe(fe_concat)
        predict_fe = self.fe_predict(fe_attention)

        be_concat = torch.cat((b1_attention, e1), 1)
        be_attention = self.cbam_be(be_concat)
        predict_be = self.be_predict(be_attention)

        predict_fb = predict_fe - predict_be

        predict_f4 = F.upsample(predict_f4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f3 = F.upsample(predict_f3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f2 = F.upsample(predict_f2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_f1 = F.upsample(predict_f1, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict_b4 = F.upsample(predict_b4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b3 = F.upsample(predict_b3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b2 = F.upsample(predict_b2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict_b1 = F.upsample(predict_b1, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict_e = F.upsample(predict_e, size=x.size()[2:], mode='bilinear', align_corners=True)

        predict_fb = F.upsample(predict_fb, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict_f4, predict_f3, predict_f2, predict_f1, \
                   predict_b4, predict_b3, predict_b2, predict_b1, \
                   predict_e, predict_fb

        return F.sigmoid(predict_e), F.sigmoid(predict_fb)

