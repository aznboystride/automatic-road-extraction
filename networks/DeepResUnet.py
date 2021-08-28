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
    def __init__(self, init_in_channels=32, num_blocks=5):
        super().__init__()

        self._is_verbose = False

        # initial convolution to downsample
        self.firstconv = conv1x1(3, init_in_channels)

        # Encoder block
        encoder_residual_blocks_list = nn.ModuleList()
        in_channels = init_in_channels
        for i in range(num_blocks):
            out_channels = in_channels * 2
            residual_block = ResidualBlock(in_channels=in_channels, out_channels=out_channels, is_encoder=True, is_first_encoder_block=i == 0)
            encoder_residual_blocks_list.append(residual_block)
            in_channels = out_channels
        # self.encoder_residual_blocks = nn.Sequential(*encoder_residual_blocks_list)
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
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        decoder_conv1x1_list = nn.ModuleList()
        decoder_residual_block_list = nn.ModuleList()
        for i in range(num_blocks):
            out_channels = in_channels // 2
            decoder_conv1x1 = nn.Sequential(conv1x1(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            residual_block = ResidualBlock(in_channels=in_channels, out_channels=out_channels, is_encoder=False, is_first_encoder_block=i == 0)
            decoder_conv1x1_list.append(decoder_conv1x1)
            decoder_residual_block_list.append(residual_block)
            in_channels = out_channels
        # self.decoder_residual_blocks = nn.Sequential(*decoder_residual_block_list)
        self.decoder_conv1x1_list = decoder_conv1x1_list
        self.decoder_residual_block_list = decoder_residual_block_list

        # final conv
        self.finalconv1x1 = conv1x1(in_channels, 1, stride=1)

        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoder_residual_block_tensors = []
        out = self.firstconv(x)
        if self._is_verbose:
            print(f"[firstconv], {out.shape}")
        for i, encoder_residual_block in enumerate(self.encoder_residual_blocks_list):
            out = encoder_residual_block(out)
            if self._is_verbose:
                print(f"[encoder block {i+1}], {out.shape}")
            encoder_residual_block_tensors.append(out)
        out = self.bridge(out)

        if self._is_verbose:
            print(f"[bridge], {out.shape}")
        for i in range(len(self.decoder_residual_block_list)):
            out = self.upsample(out)
            if self._is_verbose:
                print(f"[upsample {i+1}] {out.shape}")
            out = self.decoder_conv1x1_list[i](out)
            if self._is_verbose:
                print(f"[decoder conv1x1 {i + 1}] {out.shape}")
            out = torch.cat((encoder_residual_block_tensors.pop(), out), 1)
            if self._is_verbose:
                print(f"[concat block {i+1}] {out.shape}")
            out = self.decoder_residual_block_list[i](out)
            if self._is_verbose:
                print(f"[decoder block {i+1}] {out.shape}")
        out = self.finalconv1x1(out)
        if self._is_verbose:
            print(f"[finalconv1x1] {out.shape}")
        out = self.sigmoid(out)
        if self._is_verbose:
            print(f"[sigmoid (output layer)] {out.shape}\n")
        return out


##############################
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_encoder=True, is_first_encoder_block=False):
        super().__init__()

        self.is_encoder = is_first_encoder_block
        self.is_encoder = is_encoder
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
            self.conv1 = conv3x3(in_channels, out_channels, stride=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(out_channels, out_channels)
            self.residual_block = nn.Sequential(self.bn1, self.relu, self.conv1, self.bn2, self.relu, self.conv2)
            self.downsample = nn.Sequential(conv1x1(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        # print("input", x.shape)
        out = self.residual_block(x)

        # print("output", out.shape)
        residual = self.downsample(x)

        # print("residual", residual.shape)

        out += residual
        return out


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)