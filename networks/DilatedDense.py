import torch
import torch.nn as nn

from .layers import *


class DilatedDense(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(3,3,4,5),
                 up_blocks=(5,4,3,3), bottleneck_layers=4,
                 growth_rate=8, out_chans_first_conv=16, n_classes=1,
                 pre_conv=False, stride_pre_conv=2, dropout_rate=0.2
                ):
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        
        ## Pre- Convolution ##
        if pre_conv:
            self.add_module('preconv', nn.Conv2d(in_channels=in_channels,
                      out_channels=out_chans_first_conv, kernel_size=7,
                      stride=stride_pre_conv, padding=1, bias=True))
        
        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                  out_channels=out_chans_first_conv, kernel_size=3,
                  stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv
        
        self.add_module('premaxpool', nn.MaxPool2d(2))

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        self.dropoutBlocksDown = nn.ModuleList([])

        for i in range(len(down_blocks)):           
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i],dropout_rate=None))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0,cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count)) 

            self.dropoutBlocksDown.append(nn.Dropout2d(dropout_rate))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck',Bottleneck(cur_channels_count,
                                     growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate*bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        #   Skip Dilation Pipeline    #
        '''
        torch.Size([4, 136, 64, 64])
        torch.Size([4, 96, 128, 128])
        torch.Size([4, 64, 256, 256])
        torch.Size([4, 40, 512, 512])
        '''
        self.skipDilationPipeline = nn.ModuleList([
                nn.Sequential(*[nn.Conv2d(136, 136, kernel_size=3, dilation=8, padding=8),nn.BatchNorm2d(136), nn.ReLU()]),
                nn.Sequential(*[nn.Conv2d(96, 96, kernel_size=3, dilation=4, padding=4),nn.BatchNorm2d(96), nn.ReLU()]),
                nn.Sequential(*[nn.Conv2d(64, 64, kernel_size=3, dilation=2, padding=2),nn.BatchNorm2d(64), nn.ReLU()]),
                nn.Sequential(*[nn.Conv2d(40, 40, kernel_size=3, dilation=1, padding=1),nn.BatchNorm2d(40), nn.ReLU()])
        ])

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.transUpBlocks.append(TransitionUpB(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                    upsample=True))
            prev_block_channels = growth_rate*up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUpB(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
                upsample=False))
        cur_channels_count += growth_rate*up_blocks[-1]

        ## Softmax ##
        
        self.finalupsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
               out_channels=n_classes, kernel_size=1, stride=1,
                   padding=0, bias=True)
#         self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.firstconv(x)
        
        out = self.premaxpool(out)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)
            
            out = self.dropoutBlocksDown[i](out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            skip = self.skipDilationPipeline[i](skip)
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalupsample(out)
            
        out = self.finalConv(out)
        out = self.sigmoid(out)
        return out

'''
def FCDenseNetSmall(n_classes,dropout_rate=None):
    return FCDenseNet(
        in_channels=3, down_blocks=(3,3,4,5),
        up_blocks=(5,4,3,3), bottleneck_layers=4,
        growth_rate=8, out_chans_first_conv=16, pre_conv=False, stride_pre_conv=2, n_classes=n_classes, dropout_rate=dropout_rate)
'''

