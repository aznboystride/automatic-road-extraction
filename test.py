import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import models
import torch.nn.functional as F
import cv2
import os
import sys
from functools import partial
from time import time
import numpy as np
from torch.autograd import Variable as V
from networks.unet34 import UNet34
from networks.dinknet34 import DinkNet34
from networks.dinkducnet34 import DinkDUCNet34
from testframe import TTAFrame

import argparse

parser = argparse.ArgumentParser(description='Tester')

parser.add_argument('-b', '--batchsize', type=int, metavar='', required=True, help='Batch size')

parser.add_argument('-s', '--save', type=str, metavar='', required=True, help='Save folder name')

parser.add_argument('-w', '--weight', type=str, metavar='', required=False, help='Weights file path')

parser.add_argument('-d', '--data', type=str, metavar='', required=True, help='Path to data test folder')

parser.add_argument('-i', '--idevices', type=str, metavar='', required=True, help='Device ids')

parser.add_argument("network", required=True, type=str, help='unet34, dinknet34, dinkducnet34')

arguments = parser.parse_args()

dirs = ('submits', 'weights', 'logs')

for dir in dirs:
    if not os.path.isdir(dir):
        os.mkdir(dir)

available_nets = ("unet34", "dinknet34", "dinkducnet34")

if arguments.network not in available_nets:
    raise Exception("You must specify network name")

#source = 'dataset/test/'
source = arguments.data
val = os.listdir(source)
solver = TTAFrame(, arguments.idevices, arguments.batchsize)
solver.load(arguments.weight)
tic = time()
target = 'submits/{}'.format(arguments.save)

if not os.path.isdir(target) : os.mkdir(target)
for i,name in enumerate(val):
    if i%10 == 0:
        print(i/10, '    ','%.2f'%(time()-tic))

    mask = solver.test_one_img_from_path(source+name)
    mask[mask>4.0] = 255
    mask[mask<=4.0] = 0
    mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
    cv2.imwrite(target+name[:-7]+'mask.png',mask.astype(np.uint8))
