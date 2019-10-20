import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = nn.ReLU()


def getmodule(module):
    from .ASPP import ASPP
    return ASPP(512)


class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=9, padding=9)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=17, padding=17)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out


# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, n_filters):
#         super(DecoderBlock,self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nonlinearity
#
#         self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
#         self.norm2 = nn.BatchNorm2d(in_channels // 4)
#         self.relu2 = nonlinearity
#
#         self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nonlinearity
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)
#         x = self.deconv2(x)
#         x = self.norm2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x


# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, n_filters):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         # self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         # self.relu1 = nonlinearity
#         self.upsample = nn.Upsample(scale_factor=2**i, mode='bilinear')
#
#     def forward(self, x):
#         x = self.conv1(x)
#         # x = self.norm1(x)
#         # x = self.relu1(x)
#         return x


class aspp_dnet_add(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = nonlinearity 
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.ASPP     = getmodule('ASPP')

        self.dblock = Dblock(512)

        self.dilate1 = nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(128, 256, kernel_size=3, dilation=2, padding=2)
        self.dilate4 = nn.Conv2d(256, 512, kernel_size=3, dilation=4, padding=4)

        # self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        # self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.conv1 = nn.Conv2d(filters[3], filters[2], 1)
        self.norm1 = nn.BatchNorm2d(filters[3])
        self.conv2 = nn.Conv2d(filters[2], filters[1], 1)
        self.norm2 = nn.BatchNorm2d(filters[2])
        self.conv3 = nn.Conv2d(filters[1], filters[0], 1)
        self.norm3 = nn.BatchNorm2d(filters[1])
        self.conv4 = nn.Conv2d(filters[0], num_classes, 1)

        self.norm1_ = nn.BatchNorm2d(256)
        self.norm2_ = nn.BatchNorm2d(128)
        self.norm3_ = nn.BatchNorm2d(64)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # ASPP
        e4 = self.ASPP(e4)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = nonlinearity(self.norm1_(self.conv1(self.upsample1(e4) + nonlinearity(self.norm1(self.dilate4(e3))))))
        d3 = nonlinearity(self.norm2_(self.conv2(self.upsample2(d4) + nonlinearity(self.norm2(self.dilate2(e2))))))
        d2 = nonlinearity(self.norm3_(self.conv3(self.upsample3(d3) + nonlinearity(self.norm3(self.dilate1(e1))))))
        d1 = self.conv4(d2)
        out = self.upsample4(d1)

        # out = self.finaldeconv1(d1)
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        # out = self.finalconv3(out)

        return nn.Sigmoid()(out)
