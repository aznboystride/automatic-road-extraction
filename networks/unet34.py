import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import models
import torch.nn.functional as F
import cv2
import os
from functools import partial
from time import time
import numpy as np
from torch.autograd import Variable as V

class UNet34(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet34(pretrained=True)

        decoder_inputs = [512, 512+256, 256+256, 256+128, 64+64,128]
        decoder_outputs= [256, 256,256, 64, 128, 32]

        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(
                        decoder_inputs[0],
                        decoder_outputs[0]
                    )

        self.decoder3 = DecoderBlock(
                        decoder_inputs[1],
                        decoder_outputs[1]
                    )

        self.decoder2 = DecoderBlock(
                        decoder_inputs[2],
                        decoder_outputs[2]
                    )

        self.decoder1 = DecoderBlock(
                        decoder_inputs[3],
                        decoder_outputs[3]
                    )

        self.lastdecoder1 = DecoderBlock(
                        decoder_inputs[4],
                        decoder_outputs[4]
                    )

        self.lastdecoder2 = DecoderBlock(
                        decoder_inputs[5],
                        decoder_outputs[5]
                    )

        self.finalconv1 = nn.Conv2d(decoder_outputs[5], decoder_outputs[5], kernel_size=3, stride=1, padding=1)
        self.norm2 = norm(decoder_outputs[5])
        self.relu2 = nonlinear()

        self.finalconv = nn.Conv2d(decoder_outputs[5], 1, kernel_size=1, stride=1)
#         self.norm3 = norm(1)
#         self.relu3 = nonlinear()

#         self.dropout = nn.Dropout2d(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, t):
        conv1 = self.conv1(t)
        bn1 = self.bn1(conv1)
        relu = self.relu(bn1)
        maxpool = self.maxpool(relu)
        encoder1 = self.encoder1(maxpool)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        encoder4 = self.encoder4(encoder3)
        pool = nn.MaxPool2d(2,2)(encoder4)
        block4 = torch.cat((self.decoder4(pool), encoder4), 1)
        block3 = torch.cat((self.decoder3(block4), encoder3), 1)
        block2 = torch.cat((self.decoder2(block3), encoder2), 1)
        block1 = torch.cat((self.decoder1(block2), encoder1), 1)

        lastdecoder1 = self.lastdecoder1(block1)
        lastdecoder2 = self.lastdecoder2(lastdecoder1)

        out = self.finalconv1(lastdecoder2)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.finalconv(out)
#         out = self.norm3(out)
#         out = self.relu3(out)

#         out = self.dropout(out)
        output = self.sigmoid(out)
        return output
