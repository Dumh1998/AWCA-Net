import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEnhancementModule_conv1(nn.Module):
    def __init__(self, in_channels_list, out_channels, target_size):
        super(FeatureEnhancementModule_conv1, self).__init__()
        self.target_size = target_size
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_channels in in_channels_list
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        processed = []
        for i, conv in enumerate(self.conv_layers):
            feat = conv(features[i])
            if feat.size(2) != self.target_size:
                feat = nn.functional.interpolate(feat, size=self.target_size, mode='bilinear', align_corners=True)
            processed.append(feat)
        fused = torch.cat(processed, dim=1) 
        out = self.fuse_conv(fused)
        return out


class NECM(nn.Module):
    def __init__(self):
        super(NECM, self).__init__()

        self.feature_enhance = FeatureEnhancementModule_conv1(
            in_channels_list=[64, 128, 320, 512],
            out_channels=128,
            target_size=64
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=1),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(320, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )



    def forward(self, pvt):
        pvt_1, pvt_2, pvt_3, pvt_4 = pvt
        enhanced = self.feature_enhance([pvt_1, pvt_2, pvt_3, pvt_4])

        dec1 = self.decoder1(enhanced)  
        dec1_fused = dec1 + pvt_1

        dec2 = self.decoder2(dec1_fused)  
        dec2 = F.interpolate(dec2, size=pvt_2.shape[2:], mode='bilinear', align_corners=True)
        dec2_fused = dec2 + pvt_2

        dec3 = self.decoder3(dec2_fused)  
        dec3 = F.interpolate(dec3, size=pvt_3.shape[2:], mode='bilinear', align_corners=True)
        dec3_fused = dec3 + pvt_3

        dec4 = self.decoder4(dec3_fused) 
        dec4 = F.interpolate(dec4, size=pvt_4.shape[2:], mode='bilinear', align_corners=True)
        dec4_fused = dec4 + pvt_4

        return dec1_fused,dec2_fused,dec3_fused,dec4_fused
    
    
    




