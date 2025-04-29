import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.CSPdarknet53_tiny import darknet53_tiny
from nets.Common import CARAFE, cbam_block, eca_block, se_block, CA_Block
attention_block = [se_block, cbam_block, eca_block, CA_Block]

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class MSFNet_Head(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False):
        super(MSFNet_Head, self).__init__()
        self.phi            = phi
        self.backbone       = darknet53_tiny(pretrained)
        self.conv_for_P5    = BasicConv(512,256,1)
        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)
        self.upsample_1       = Upsample(256,128)
        self.conv1 = BasicConv(256,128,1)
        self.upsample_2 = CARAFE(128)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)
        if 1 <= self.phi and self.phi <= 4:
            self.feat1_att      = attention_block[self.phi - 1](256)
            self.feat2_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)
            
    def forward(self, x):
        feat1, feat2 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 4:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)
        P5 = self.conv_for_P5(feat2)
        out0 = self.yolo_headP5(P5)
        P6 = self.conv_for_P5(feat2)
        P6_Upsample = self.upsample_1(P6)
        P5 = self.conv1(P5)
        P5_Upsample = self.upsample_2(P5)
        sum = P5_Upsample + P6_Upsample
        P4 = torch.cat([sum, feat1],axis=1)
        out1 = self.yolo_headP4(P4)
        
        return out0, out1
