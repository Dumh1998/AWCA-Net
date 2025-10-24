import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
import numpy as np
from .backbones.pvtv2 import *
from .modules.NECM import *
import warnings

warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])
        self.n_scales = len(self.kernel_sizes)
        self.attention_fc = nn.Sequential(
            nn.Linear(self.n_scales, self.n_scales, bias=True),
            act_layer(self.activation, inplace=True),
            nn.Linear(self.n_scales, self.n_scales, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        pooled = []
        for out in outputs:
            scalar = out.mean(dim=[1, 2, 3], keepdim=True)
            pooled.append(scalar)
        pooled = torch.cat(pooled, dim=1)
        pooled = pooled.squeeze(-1).squeeze(-1)
        att_weights = self.attention_fc(pooled)
        att_weights = F.softmax(att_weights, dim=1)
        weighted_outs = []
        for i, out in enumerate(outputs):
            w = att_weights[:, i].view(-1, 1, 1, 1)
            weighted_out = w * out
            weighted_outs.append(weighted_out)
        return weighted_outs


class MSCB(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
                 add=True, activation='relu6'):
        super(MSCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        assert self.stride in [1, 2]
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation,
                         dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels * 1
        else:
            self.combined_channels = self.ex_channels * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def AMSDC(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1, 3, 5], expansion_factor=2, dw_parallel=True,
              add=True, activation='relu6'):
    convs = []
    mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                dw_parallel=dw_parallel, add=add, activation=activation)
    convs.append(mscb)
    if n > 1:
        for i in range(1, n):
            mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor,
                        dw_parallel=dw_parallel, add=add, activation=activation)
            convs.append(mscb)
    conv = nn.Sequential(*convs)
    return conv


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x


class LGDM(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGDM, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(2*F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        fusion = torch.cat([g1+x1, g1-x1], dim=1)
        psi = self.activation(fusion)
        psi = self.psi(psi)

        return x * psi


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)



class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class MSAWM(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6'):
        super(MSAWM, self).__init__()
        eucb_ks = 3  # kernel size for eucb
        self.amsdc4 = AMSDC(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgdm3 = LGDM(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks,
                          groups=channels[1] // 2)
        self.mamsdc3 = AMSDC(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgdm2 = LGDM(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks,
                          groups=channels[2] // 2)
        self.amsdc2 = AMSDC(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks // 2)
        self.lgdm1 = LGDM(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), kernel_size=lgag_ks,
                          groups=int(channels[3] / 2))
        self.amsdc1 = AMSDC(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes,
                               expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add,
                               activation=activation)

        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])

        self.sab = SAB()

    def forward(self, x, skips):
        d4 = self.cab4(x) * x
        d4 = self.sab(d4) * d4
        d4 = self.amsdc4(d4)
        d3 = self.eucb3(d4)
        x3 = self.lgdm3(g=d3, x=skips[0])
        d3 = d3 + x3
        d3 = self.cab3(d3) * d3
        d3 = self.sab(d3) * d3
        d3 = self.amsdc3(d3)
        d2 = self.eucb2(d3)
        x2 = self.lgdm2(g=d2, x=skips[1])
        d2 = d2 + x2
        d2 = self.cab2(d2) * d2
        d2 = self.sab(d2) * d2
        d2 = self.amsdc2(d2)
        d1 = self.eucb1(d2)
        x1 = self.lgdm1(g=d1, x=skips[2])
        d1 = d1 + x1
        d1 = self.cab1(d1) * d1
        d1 = self.sab(d1) * d1
        d1 = self.amsdc1(d1)

        return [d4, d3, d2, d1]


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pvt = pvt_v2_b2()

        pvt_path = '../pvt_v2_b2.pth'
        pvt_save_model = torch.load(pvt_path)
        pvt_model_dict = self.pvt.state_dict()
        pvt_state_dict = {k: v for k, v in pvt_save_model.items() if k in pvt_model_dict.keys()}
        pvt_model_dict.update(pvt_state_dict)
        self.pvt.load_state_dict(pvt_model_dict)

    def forward(self, A, B):
        pvta = self.pvt(A)
        pvtb = self.pvt(B)
        return pvta, pvtb


class Decoder(nn.Module):
    def __init__(self, channels, num_classes):
        super(Decoder, self).__init__()
        self.enhance = neighborenhance_conv1()
        self.conv1 = BasicConv2d(2 * 64, 64, 1)
        self.conv2 = BasicConv2d(2 * 128, 128, 1)
        self.conv3 = BasicConv2d(2 * 320, 320, 1)
        self.conv4 = BasicConv2d(2 * 512, 512, 1)

        self.conv_4 = BasicConv2d(channels[0], channels[0], 3, 1, 1)
        self.conv_3 = BasicConv2d(channels[1], channels[1], 3, 1, 1)
        self.conv_2 = BasicConv2d(channels[2], channels[2], 3, 1, 1)
        self.conv_1 = BasicConv2d(channels[3], channels[3], 3, 1, 1)

        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        self.decoder = MSAWM(channels=channels, kernel_sizes=[1,3,5], expansion_factor=6,
                             dw_parallel=True, add=True, lgag_ks=3, activation='relu6')

        self.sig = nn.Sigmoid()

    def forward(self,pvta, pvtb):
        pvt_a1, pvt_a2, pvt_a3, pvt_a4 = self.enhance(pvta)
        pvt_b1, pvt_b2, pvt_b3, pvt_b4 = self.enhance(pvtb)

        layer_1 = self.conv_1(self.conv1(torch.cat((pvt_a1, pvt_b1), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((pvt_a2, pvt_b2), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((pvt_a3, pvt_b3), dim=1)))
        layer_4 = self.conv_4(self.conv4((torch.cat((pvt_a4, pvt_b4), dim=1))))
        dec_outs = self.decoder(layer_4, [layer_3, layer_2, layer_1])

        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear')

        return [self.sig(p4), self.sig(p3), self.sig(p2), self.sig(p1)]




class AWCANet(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(AWCANet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(channels=[512, 320, 128, 64],num_classes=1)

    def forward(self, A, B):
        pvta, pvtb = self.encoder(A, B)
        pred = self.decoder(pvta, pvtb)
        return pred



if __name__ == '__main__':
    A = torch.rand(4, 1, 256, 256).cuda()
    B = torch.rand(4, 1, 256, 256).cuda()

    model = AWCANet(latent_dim=8, num_classes=1).cuda()

    outs = model(A, B)


