import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.pvtv2 import *
from .modules.NECM import *
from .modules.MSAWM import *


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
        self.enhance = NECM()
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
        
        self.LM = MSAWM(channels=channels, kernel_sizes=[1,3,5], expansion_factor=6,
                             dw_parallel=True, add=True, lgag_ks=3, activation='relu6')

        self.sig = nn.Sigmoid()

    def forward(self,pvta, pvtb):
        pvt_a1, pvt_a2, pvt_a3, pvt_a4 = self.enhance(pvta)
        pvt_b1, pvt_b2, pvt_b3, pvt_b4 = self.enhance(pvtb)

        layer_1 = self.conv_1(self.conv1(torch.cat((pvt_a1, pvt_b1), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((pvt_a2, pvt_b2), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((pvt_a3, pvt_b3), dim=1)))
        layer_4 = self.conv_4(self.conv4((torch.cat((pvt_a4, pvt_b4), dim=1))))
        outs = self.LM(layer_4, [layer_3, layer_2, layer_1])

        u4 = self.out_head4(outs[0])
        u3 = self.out_head3(outs[1])
        u2 = self.out_head2(outs[2])
        u1 = self.out_head1(outs[3])

        u4 = F.interpolate(u4, scale_factor=32, mode='bilinear')
        u3 = F.interpolate(u3, scale_factor=16, mode='bilinear')
        u2 = F.interpolate(u2, scale_factor=8, mode='bilinear')
        u1 = F.interpolate(u1, scale_factor=4, mode='bilinear')

        return [self.sig(u4), self.sig(u3), self.sig(u2), self.sig(u1)]


class AWCANet(nn.Module):
    def __init__(self):
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

    model = AWCANet().cuda()

    outs = model(A, B)


