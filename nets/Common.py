import torch
import torch.nn as nn
import cv2
from torch.nn import functional as F
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x

class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        _, _, h, w = x.size()
        
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)
 
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
    
class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):

        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)
    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        W = self.comp(X) 
        W = self.enc(W) 
        W = self.pix_shf(W)  
        W = F.softmax(W, dim=1)  
        X = self.upsmp(X)  
        X = self.unfold(X)  
        X = X.view(b, c, -1, h_, w_) 
        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  
        return X

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):

        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)
    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale
        W = self.comp(X) 
        W = self.enc(W) 
        W = self.pix_shf(W)  
        W = F.softmax(W, dim=1)  
        X = self.upsmp(X)  
        X = self.unfold(X)  
        X = X.view(b, c, -1, h_, w_) 
        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  
        return X
    
class DilatedConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, padding, kernel_size):
        super(DilatedConvNet, self).__init__()
        self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        x = self.dilated_conv(x)
        x = self.relu(x)

        return x

class LAM(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.eca = eca_block(ch)
        self.conv1 = nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, x):
        x = self.eca(x)
        x = self.conv1(x)
        return x

class RFEM(nn.Module):
    def __init__(
            self,
            ch_blocks=64,
            ch_mask=16,
    ):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, ch_blocks, 3, padding=1),
                                     nn.LeakyReLU(True))

        self.dconv1 = DilatedConvNet(ch_blocks,
                                  ch_blocks // 4,
                                  kernel_size=3,
                                  padding=1, dilation=1)
        self.dconv2 = DilatedConvNet(ch_blocks,
                                  ch_blocks // 4,
                                  kernel_size=3,
                                  padding=2, dilation=2)
        self.dconv3 = DilatedConvNet(ch_blocks,
                                  ch_blocks // 4,
                                  kernel_size=3,
                                  padding=3, dilation=3)
        self.dconv4 = nn.Conv2d(ch_blocks,
                                  ch_blocks // 4,
                                  kernel_size=7,
                                  padding=3)

        self.decoder = nn.Sequential(nn.Conv2d(ch_blocks, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, 3, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     )

        self.lam = LAM(ch_mask)

    def forward(self, x):
        x1 = self.encoder(x)
        x1_1 = self.dconv1(x1)
        x1_2 = self.dconv2(x1)
        x1_3 = self.dconv3(x1)
        x1_4 = self.dconv4(x1)
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1 = self.decoder(x1)
        out = x + x1
        out = torch.relu(out)
        mask = self.lam(torch.cat([x, out], dim=1))
        return out, mask

class ATEM(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(True),
        )
        self.shift_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))


        self.decoder = nn.Sequential(
            nn.Conv2d(inter_ch, out_ch, kernel_size, padding=kernel_size // 2))

    def forward(self, x, tag):
        x = self.encoder(x)
        scale = self.scale_conv(tag)
        shift = self.shift_conv(tag)
        x = x +(x * scale + shift)
        x = self.decoder(x)
        return x

class Trans_high(nn.Module):
    def __init__(self, in_ch=3, inter_ch=16, out_ch=3, kernel_size=3):
        super().__init__()
        self.atem = ATEM(in_ch, inter_ch, out_ch, kernel_size)
    def forward(self, x, tag):
        x = x + self.atem(x, tag)
        return x


class Up_tag(nn.Module):
    def __init__(self, kernel_size=1, ch=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch,
                      ch,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                mode='reflect') 
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x
    def downsample(self, x):
        return x[:, :, ::2, ::2]
    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))
    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image

class FAENet(nn.Module):
    def __init__(self,
                 num_high=1,
                 ch_blocks=32,
                 up_ksize=1,
                 high_ch=32,
                 high_ksize=3,
                 ch_mask=32,
                 gauss_kernel=7):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.rfem = RFEM(ch_blocks, ch_mask)

        for i in range(0, self.num_high):
            self.__setattr__('up_tag_layer_{}'.format(i),
                             Up_tag(up_ksize, ch=3))
            self.__setattr__('trans_high_layer_{}'.format(i),
                             Trans_high(3, high_ch, 3, high_ksize))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []
        trans_pyr, tag = self.rfem(pyrs[-1])
        trans_pyrs.append(trans_pyr)

        commom_tag = []
        for i in range(self.num_high):
            tag = self.__getattr__('up_tag_layer_{}'.format(i))(tag)
            commom_tag.append(tag)

        for i in range(self.num_high):
            trans_pyr = self.__getattr__('trans_high_layer_{}'.format(i))(
                pyrs[-2 - i], commom_tag[i])
            trans_pyrs.append(trans_pyr)

        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        return out
    
    