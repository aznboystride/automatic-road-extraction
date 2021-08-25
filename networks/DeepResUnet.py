import torch.nn as nn
import torch
import math


# DeepResUnet bridge
class Bridge(nn.Module):
    def __init__(self, init_in_channels=64, num_blocks=5):
        super().__init__()

        in_channels = 2**(math.log2(init_in_channels) + num_blocks)
        out_channels = in_channels * 2
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bridge = nn.Sequential(self.bn1, self.relu, self.conv1, self.self.bn2, self.relu, self.conv2)

    def forward(self, x):
        out = self.bridge(x)
        return out


class Encoder(nn.Module):
    def __init__(self, init_in_channels=64, num_blocks=5):
        super().__init__()
        residual_blocks_list = []
        in_channels = init_in_channels
        for i in range(num_blocks):
            out_channels = in_channels*2
            residual_block = ResidualBlock(in_channels=in_channels, out_channels=out_channels, is_first_encoder_block=i == 0)
            residual_blocks_list.append(residual_block)
            in_channels = out_channels
        self.residual_blocks = nn.Sequential(*residual_blocks_list)

    def forward(self, x):
        out = self.residual_blocks(x)
        return out


class Decoder(nn.Module):
    def __init__(self, init_in_channels=64, num_blocks=5):
        super().__init__()

        in_channels = int(2**(math.log2(init_in_channels) + num_blocks))
        residual_blocks_list = []
        for i in range(num_blocks):
            out_channels = in_channels * 2
            residual_block = ResidualBlock(in_channels=in_channels, out_channels=out_channels, is_first_encoder_block=i == 0)
            residual_blocks_list.append(residual_block)
            in_channels = out_channels
        self.residual_blocks = nn.Sequential(*residual_blocks_list)

    def forward(self, x):
        out = self.bridge(x)
        return out


class DeepResUnet(nn.Module):
    def __init__(self, init_in_channels=64, num_blocks=5):
        super().__init__()

        # initial convolution to downsample
        self.firstconv = conv1x1(3, init_in_channels)

        # Encoder block
        encoder_residual_blocks_list = []
        in_channels = init_in_channels
        for i in range(num_blocks):
            out_channels = in_channels * 2
            residual_block = ResidualBlock(in_channels=in_channels, out_channels=out_channels, is_encoder=True, is_first_encoder_block=i == 0)
            encoder_residual_blocks_list.append(residual_block)
            in_channels = out_channels
        self.encoder_residual_blocks = nn.Sequential(*encoder_residual_blocks_list)
        self.encoder_residual_blocks_list = encoder_residual_blocks_list

        # Bridge block
        in_channels = int(2 ** (math.log2(init_in_channels) + num_blocks))
        out_channels = in_channels * 2
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bridge = nn.Sequential(self.bn1, self.relu, self.conv1, self.bn2, self.relu, self.conv2)

        # Decoder block
        in_channels = out_channels
        decoder_residual_block_list = []
        for i in range(num_blocks):
            out_channels = in_channels // 2
            residual_block = ResidualBlock(in_channels=in_channels, out_channels=out_channels, is_encoder=False, is_first_encoder_block=i == 0)
            decoder_residual_block_list.append(residual_block)
            in_channels = out_channels
        self.decoder_residual_blocks = nn.Sequential(*decoder_residual_block_list)
        self.decoder_residual_block_list = decoder_residual_block_list
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.sigmoid = nn.Sigmoid()

        # final conv
        self.finalconv1x1 = conv1x1(in_channels, 1, stride=1)

    def forward(self, x):
        out = self.firstconv(x)
        out = self.encoder_residual_blocks(out)
        out = self.bridge(out)

        for i in range(len(self.decoder_residual_block_list)):
            out = self.upsample(out)
            out = self.decoder_residual_blocks(out)
            out = torch.cat((self.encoder_residual_blocks_list[-1-i], out), 1)
        out = self.finalconv1x1(out)
        out = self.sigmoid(out)
        return out


##############################
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_encoder=True, is_first_encoder_block=False):
        super().__init__()

        self.bsmall_block = is_first_encoder_block
        self.bencoder = is_encoder
        if is_encoder:
            if is_first_encoder_block:
                self.conv1 = conv3x3(in_channels, out_channels, stride=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(out_channels, out_channels)
                self.residual_block = nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2)
                self.downsample = nn.Sequential(conv1x1(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            else:
                self.bn1 = nn.BatchNorm2d(in_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv1 = conv3x3(in_channels, out_channels, stride=2)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(out_channels, out_channels)
                self.residual_block = nn.Sequential(self.bn1, self.relu, self.conv1, self.bn2, self.relu, self.conv2)
                self.downsample = nn.Sequential(conv1x1(in_channels, out_channels, 2), nn.BatchNorm2d(out_channels))
        else:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = conv3x3(in_channels, out_channels, stride=2)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels, out_channels)
            self.residual_block = nn.Sequential(self.bn1, self.relu, self.conv1, self.bn2, self.relu, self.conv2)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.residual_block(x)
        # print("residual", residual.shape)
        # print("input", x.shape)
        # print("output", out.shape)
        out += residual
        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)