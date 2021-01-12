"""
 @Time    : 1/12/21 18:39
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : iccv
 @File    : mirrornet_plus.py
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
# ###################### Contrast Module ##########################
###################################################################
class Contrast_Module(nn.Module):
    def __init__(self, planes):
        super(Contrast_Module, self).__init__()
        self.inplanes = int(planes)
        self.inplanes_half = int(planes / 2)
        self.outplanes = int(planes / 4)

        self.conv1 = nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes_half, 3, 1, 1),
                                   nn.BatchNorm2d(self.inplanes_half), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(self.inplanes_half, self.outplanes, 3, 1, 1),
                                   nn.BatchNorm2d(self.outplanes), nn.ReLU())

        self.contrast_block_1 = Contrast_Block(self.outplanes)
        self.contrast_block_2 = Contrast_Block(self.outplanes)
        self.contrast_block_3 = Contrast_Block(self.outplanes)
        self.contrast_block_4 = Contrast_Block(self.outplanes)

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        contrast_block_1 = self.contrast_block_1(conv2)
        contrast_block_2 = self.contrast_block_2(contrast_block_1)
        contrast_block_3 = self.contrast_block_3(contrast_block_2)
        contrast_block_4 = self.contrast_block_4(contrast_block_3)

        output = self.cbam(torch.cat((contrast_block_1, contrast_block_2, contrast_block_3, contrast_block_4), 1))

        return output


###################################################################
# ###################### Contrast Block ###########################
###################################################################
class Contrast_Block(nn.Module):
    def __init__(self, planes):
        super(Contrast_Block, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 4)

        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=2, dilation=2)

        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=4, dilation=4)

        self.local_3 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_3 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=8, dilation=8)

        self.local_4 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_4 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=16, dilation=16)

        self.bn_1 = nn.BatchNorm2d(self.outplanes)
        self.bn_2 = nn.BatchNorm2d(self.outplanes)
        self.bn_3 = nn.BatchNorm2d(self.outplanes)
        self.bn_4 = nn.BatchNorm2d(self.outplanes)
        self.relu = nn.ReLU()

        self.cbam = CBAM(self.inplanes)

    def forward(self, x):
        local_1 = self.local_1(x)
        context_1 = self.context_1(x)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn_1(ccl_1)
        ccl_1 = self.relu(ccl_1)

        local_2 = self.local_2(x)
        context_2 = self.context_2(x)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn_2(ccl_2)
        ccl_2 = self.relu(ccl_2)

        local_3 = self.local_3(x)
        context_3 = self.context_3(x)
        ccl_3 = local_3 - context_3
        ccl_3 = self.bn_3(ccl_3)
        ccl_3 = self.relu(ccl_3)

        local_4 = self.local_4(x)
        context_4 = self.context_4(x)
        ccl_4 = local_4 - context_4
        ccl_4 = self.bn_4(ccl_4)
        ccl_4 = self.relu(ccl_4)

        output = self.cbam(torch.cat((ccl_1, ccl_2, ccl_3, ccl_4), 1))

        return output


###################################################################
####################### Non-Local Block ###########################
###################################################################
class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
############################## MPM ################################
###################################################################
class MPM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(MPM, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)
        down = self.conv2(down)
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        w,b = down[:,:256,:,:], down[:,256:,:,:]
        return F.relu(w*left+b, inplace=True)


###################################################################
####################Dynamic Filter MPM#############################
###################################################################
class Dynamic_MPM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(Dynamic_MPM, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0   = nn.BatchNorm2d(256)
        # self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        # use dynamic filter to replace self.conv1
        self.dynamic_kernel = nn.Sequential(nn.AdaptiveAvgPool2d(3),nn.Conv2d(in_channel_down, 256, kernel_size=1))
        self.conv1_1 = nn.Conv2d(in_channel_down, 256, kernel_size=1)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)
        # down = F.relu(self.bn1(self.conv1(down)), inplace=True)
        # use dynamic filter to replace conv1
        dynamic_filter = self.dynamic_kernel(down)
        down = self.conv1_1(down)
        down = F.conv2d(down, dynamic_filter, bias=None, stride=1, padding=1, dilation=1, groups=256)
        down = self.conv1_2(down)
        # same as MPM
        down = self.conv2(down)
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        w,b = down[:,:256,:,:], down[:,256:,:,:]
        return F.relu(w*left+b, inplace=True)

##########################  RCCFE #################################
class GCM(nn.Module):
    def __init__(self, in_channels, theta):
        super(GCM, self).__init__()
        self.glo = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.theta = theta
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_glo = self.glo(x)  # (batch, C, 1, 1)
        x_D = torch.norm(x - x_glo, dim=1)  # (batch, H, W)
        x_D_min = x_D.view(x_D.shape[0], -1).min(dim=1)[0]  # (batch,)
        x_D_min = x_D_min.view(-1, 1, 1)  # (batch, 1, 1)
        W_glo = torch.exp(-(x_D - x_D_min) / self.theta)  # (batch, H, W)
        W_glo = W_glo.unsqueeze(1)  # (batch, 1, H, W)
        x_c = self.alpha * x_glo * W_glo + x
        return x_c  # return global coefficient


##########################  RCCFE #################################
class RCCFE(nn.Module):
    def __init__(self, in_channel, in_shape):
        super(RCCFE, self).__init__()
        # self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.glo = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.ccfe = Contrast_Module(in_channel)
        self.in_shape = in_shape
        self.adp = nn.AdaptiveAvgPool2d(in_shape)
        self.pam = PAM_Module(in_channel)
        # self.gcm = GCM(in_channel, 5)

    def forward(self, x):
        # x = self.conv2d(x)
        x_ccfe = self.ccfe(x)
        # x_gcm = self.gcm(x)
        x_gcm = self.glo(x)
        x_adp = self.adp(x)
        x_pam = self.pam(x_adp)
        x_up = F.interpolate(x_pam, size=x.size()[2:], mode='bilinear', align_corners=True)
        x_up = x_up + x_gcm
        x_out = x_ccfe * x_up

        return x_out

###################################################################
# ########################## NETWORK ##############################
###################################################################
class MirrorNet_Plus(nn.Module):
    def __init__(self, backbone_path=None):
        super(MirrorNet_Plus, self).__init__()
        resnext = ResNeXt101(backbone_path)
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4



        self.contrast_4 = RCCFE(2048, 12)
        self.contrast_3 = RCCFE(1024, 12)
        self.contrast_2 = RCCFE(512, 12)
        self.contrast_1 = RCCFE(256, 12)

        self.up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 4, 2, 1), nn.BatchNorm2d(1024), nn.ReLU())
        self.up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.up_2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.up_1 = nn.Sequential(nn.Conv2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())

        self.cbam_4 = CBAM(1024)
        self.cbam_3 = CBAM(512)
        self.cbam_2 = CBAM(256)
        self.cbam_1 = CBAM(128)

        self.layer4_predict = nn.Conv2d(1024, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(128, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        #print(layer4.shape)
        ###### Add PAM Module paralleled with CCFE Module ######


        contrast_4 = self.contrast_4(layer4)
        up_4 = self.up_4(contrast_4)
        cbam_4 = self.cbam_4(up_4)
        layer4_predict = self.layer4_predict(cbam_4)

        ###### left is layer3 down is cbam_4; as these filters are at the same W,H, we don't need upsampling here? #####

        # layer3_mpm = self.mpm_3(layer3, cbam_4)
        layer3_mpm = layer3 + cbam_4
        contrast_3 = self.contrast_3(layer3_mpm)
        up_3 = self.up_3(contrast_3)
        cbam_3 = self.cbam_3(up_3)
        layer3_predict = self.layer3_predict(cbam_3)

        # layer2_mpm = self.mpm_2(layer2, cbam_3)
        layer2_mpm = layer2 + cbam_3
        contrast_2 = self.contrast_2(layer2_mpm)
        up_2 = self.up_2(contrast_2)
        cbam_2 = self.cbam_2(up_2)
        layer2_predict = self.layer2_predict(cbam_2)

        # layer1_mpm = self.mpm_1(layer1, cbam_2)
        layer1_mpm = layer1 + cbam_2
        contrast_1 = self.contrast_1(layer1_mpm)
        up_1 = self.up_1(contrast_1)
        cbam_1 = self.cbam_1(up_1)
        layer1_predict = self.layer1_predict(cbam_1)

        layer4_predict = F.upsample(layer4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_predict = F.upsample(layer3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_predict = F.upsample(layer2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_predict = F.upsample(layer1_predict, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return layer4_predict, layer3_predict, layer2_predict, layer1_predict

        return F.sigmoid(layer4_predict), F.sigmoid(layer3_predict), F.sigmoid(layer2_predict), \
               F.sigmoid(layer1_predict)
