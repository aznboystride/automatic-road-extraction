import torch
import torch.nn as nn
from .drn import drn_c_26 as fk
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.elu,inplace=True)

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

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class drnlinknet(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = fk(pretrained=True)
        self.pool = nn.MaxPool2d(2)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = nonlinearity
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.encoder5 = resnet.layer5
        self.encoder6 = resnet.layer6

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        # self.decoder3 = DecoderBlock(filters[2], filters[1])
        # self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[2], filters[0])

        self.conv1 = nn.Conv2d(64, 256, 1)
        self.norm1 = nn.BatchNorm2d(256)

        # self.conv2 = nn.Conv2d(64, 128, 1)
        # self.norm2 = nn.BatchNorm2d(128)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.pool(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)

        # Center
        e6 = self.dblock(e6)
        # Decoder
        d4 = self.decoder4(e6) + nonlinearity(self.norm1(self.conv1(e3)))
        # d3 = self.decoder3(d4) + nonlinearity(self.norm2(self.conv2(e3)))
        # d2 = self.decoder2Z(d3)
        d1 = self.decoder1(d4)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finaldeconv2(out)
        out = nonlinearity(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return nn.Sigmoid()(out)
