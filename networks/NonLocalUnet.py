import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torchvision import models

from enum import Enum


class NonLocalBlock(nn.Module):
    class PairwiseFn(Enum):
        GAUSSIAN = 'gaussian'
        EMBEDDED = 'embedded'
        DOT = 'dot'
        CONCATENATE = 'concatenate'

    def __init__(self,
                 in_channels,
                 pairwise_fn: PairwiseFn = PairwiseFn.EMBEDDED,
                 batch_norm=True):

        super(NonLocalBlock, self).__init__()

        self.pairwise_fn = pairwise_fn

        self.in_channels = in_channels

        # 512 in paper. Assume half of input channels (1024 in paper)
        self.inside_channels = max(1, in_channels // 2)

        # g(x) = W_g * x
        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inside_channels,
                           kernel_size=1)

        # final conv. In paper, input 512 output 1024
        self.W_z = nn.Conv2d(in_channels=self.inside_channels,
                             out_channels=self.in_channels,
                             kernel_size=1)

        # Wz is initialized as zero (Paper 3.3)
        nn.init.constant_(self.W_z.weight, 0)
        nn.init.constant_(self.W_z.bias, 0)

        # compute theta, phi (except for in the case of gaussian)
        if self.pairwise_fn != NonLocalBlock.PairwiseFn.GAUSSIAN:
            self.theta = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.inside_channels,
                                   kernel_size=1)
            self.phi = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.inside_channels,
                                 kernel_size=1)

        if self.pairwise_fn == NonLocalBlock.PairwiseFn.CONCATENATE:
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inside_channels * 2,
                          out_channels=1,
                          kernel_size=1), nn.ReLU())

    def forward(self, x):
        """
        args
            x: (N, C, H, W) 
            paper x: (TxHxWx1024) - needs to be reshaped
        """

        batch_size = x.size(0)

        # (N, C, THW)
        g_x = self.g(x).view(batch_size, self.inside_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.pairwise_fn == NonLocalBlock.PairwiseFn.GAUSSIAN:
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.pairwise_fn == NonLocalBlock.PairwiseFn.EMBEDDED or self.pairwise_fn == NonLocalBlock.PairwiseFn.DOT:
            theta_x = self.theta(x).view(batch_size, self.inside_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inside_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.pairwise_fn == NonLocalBlock.PairwiseFn.CONCATENATE:
            theta_x = self.theta(x).view(batch_size, self.inside_channels, -1,
                                         1)
            phi_x = self.phi(x).view(batch_size, self.inside_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)

            # print('h,w:', h,w)

            # gpu memory does NOT like this...
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            f = self.W_f(torch.cat([theta_x, phi_x], dim=1))
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.pairwise_fn == NonLocalBlock.PairwiseFn.GAUSSIAN or self.pairwise_fn == NonLocalBlock.PairwiseFn.EMBEDDED:
            # Gaussian and Embedded Gaussian get a softmax
            f_normalized = softmax(f, dim=-1)
        elif self.pairwise_fn == NonLocalBlock.PairwiseFn.DOT or self.pairwise_fn == NonLocalBlock.PairwiseFn.CONCATENATE:
            # Normalization factor C(x)=N, where N is the number of positions in x (In notes after Equation 4)
            f_normalized = f / f.size(-1)

        y = torch.matmul(f_normalized, g_x)

        y = y.permute(0, 2, 1)
        y = y.view(batch_size, self.inside_channels, *x.size()[2:])

        # Equation 6
        z = self.W_z(y) + x

        return z


class NonLocalUnet(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.max_pool = nn.MaxPool2d(2)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.down0 = nn.Sequential(self.firstconv, self.firstbn,
                                   self.firstrelu)
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

        pairwise_fn = NonLocalBlock.PairwiseFn.CONCATENATE
        # pairwise_fn = NonLocalBlock.PairwiseFn.GAUSSIAN
        # pairwise_fn = NonLocalBlock.PairwiseFn.DOT

        self.d4 = nn.Sequential(
            nn.Conv2d(filters[-1],
                      filters[-1],
                      3,
                      stride=1,
                      padding=16,
                      dilation=16), nn.BatchNorm2d(filters[-1]), nn.ReLU())
        # TODO: Get rid of nn.Sequential if I'm only gonna be using one.
        # TODO: Use filters[-1], etc
        self.d4 = nn.Sequential(
            NonLocalBlock(
                in_channels=512,
                pairwise_fn=pairwise_fn,
            ), )

        self.d3 = nn.Sequential(
            nn.Conv2d(filters[-2],
                      filters[-2],
                      3,
                      stride=1,
                      padding=8,
                      dilation=8), nn.BatchNorm2d(filters[-2]), nn.ReLU())
        # self.d3 = nn.Sequential(
        #     NonLocalBlock(
        #         in_channels=256,
        #         pairwise_fn=pairwise_fn,
        #     ),
        # )

        self.d2 = nn.Sequential(
            nn.Conv2d(filters[-3],
                      filters[-3],
                      3,
                      stride=1,
                      padding=4,
                      dilation=4), nn.BatchNorm2d(filters[-3]), nn.ReLU())
        # self.d2 = nn.Sequential(
        #     NonLocalBlock(
        #         in_channels=128,
        #         pairwise_fn=pairwise_fn,
        #     ),
        # )

        self.d1 = nn.Sequential(
            nn.Conv2d(filters[-4],
                      filters[-4],
                      3,
                      stride=1,
                      padding=2,
                      dilation=2), nn.BatchNorm2d(filters[-4]), nn.ReLU())
        # self.d1 = nn.Sequential(
        #     NonLocalBlock(
        #         in_channels=64,
        #         pairwise_fn=pairwise_fn,
        #     ),
        # )

        self.d0 = nn.Sequential(
            nn.Conv2d(filters[-4],
                      filters[-4],
                      3,
                      stride=1,
                      padding=1,
                      dilation=1), nn.BatchNorm2d(filters[-4]), nn.ReLU())
        # self.d0 = nn.Sequential(
        #     NonLocalBlock(
        #         # in_channels=32,
        #         in_channels=filters[-4],
        #         pairwise_fn=pairwise_fn,
        #     ),
        # )

        self.trans4 = self.upsample(1024, 512)
        self.trans3 = self.upsample(512, 256)
        self.trans2 = self.upsample(256, 128)
        self.trans1 = self.upsample(128, 64)
        self.trans0 = self.upsample(64, 32)

        self.finalup = self.upsample(32, 32)

        self.conv_last = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def conv_stage(self,
                   dim_in,
                   dim_out,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=True,
                   useBN=True):
        if useBN:
            return nn.Sequential(
                nn.Conv2d(dim_in,
                          dim_out,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(),
                nn.Conv2d(dim_out,
                          dim_out,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(dim_in,
                          dim_out,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias), nn.ReLU(),
                nn.Conv2d(dim_out,
                          dim_out,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          bias=bias), nn.ReLU())

    def upsample(self, ch_coarse, ch_fine):
        return nn.Sequential(nn.Conv2d(ch_coarse, ch_fine, 1),
                             nn.Upsample(scale_factor=2, mode='bilinear'))

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
