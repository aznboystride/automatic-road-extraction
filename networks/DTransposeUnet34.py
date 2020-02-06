import torch
import torch.nn as nn
from torchvision import models

class DTransposeUnet34(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.max_pool = nn.MaxPool2d(2)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.down0 = nn.Sequential(self.firstconv, self.firstbn, self.firstrelu)
        self.down1 = resnet.layer1
        self.down2 = resnet.layer2
        self.down3 = resnet.layer3
        self.down4 = resnet.layer4

        self.center = self.conv_stage(512, 1024)

        self.up4 = self.conv_stage(1024, 512)
        self.up3 = self.conv_stage(512, 256)
        self.up2 = self.conv_stage(256, 128)
        self.up1 = self.conv_stage(128, 64)
        self.up0 = self.conv_stage(96, 32)


        self.d4 = nn.Sequential(
            nn.Conv2d(filters[-1], filters[-1], 3, stride=1, padding=16, dilation=16),
            nn.BatchNorm2d(filters[-1]),
            nn.ReLU()
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(filters[-2], filters[-2], 3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(filters[-2]),
            nn.ReLU()
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(filters[-3], filters[-3], 3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(filters[-3]),
            nn.ReLU()
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(filters[-4], filters[-4], 3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(filters[-4]),
            nn.ReLU()
        )

        self.d0 = nn.Sequential(
            nn.Conv2d(filters[-4], filters[-4], 3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(filters[-4]),
            nn.ReLU()
        )

        self.trans4 = self.upsample(1024, 512)
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)
        self.trans0 = self.upsample(64, 32)

        self.finalup = self.upsample(32, 32)

        self.conv_last = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=True):
        if useBN:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(dim_out),
              nn.ReLU(),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.BatchNorm2d(dim_out),
              nn.ReLU(),
            )
        else:
            return nn.Sequential(
              nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.ReLU(),
              nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
              nn.ReLU()
            )

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(
            nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        conv0_out = self.down0(x)
        conv1_out = self.down1(self.max_pool(conv0_out))
        conv2_out = self.down2(conv1_out)
        conv3_out = self.down3(conv2_out)
        conv4_out = self.down4(conv3_out)

        out = self.center(self.max_pool(conv4_out))

        out = self.up4(torch.cat((self.trans4(out), self.d4(conv4_out)), 1))
        out = self.up3(torch.cat((self.trans3(out), self.d3(conv3_out)), 1))
        out = self.up2(torch.cat((self.trans2(out), self.d2(conv2_out)), 1))
        out = self.up1(torch.cat((self.trans1(out), self.d1(conv1_out)), 1))
        out = self.up0(torch.cat((self.trans0(out), self.d0(conv0_out)), 1))
        out = self.finalup(out)

        out = self.conv_last(out)
        return out
