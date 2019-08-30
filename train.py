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
from loss import dice_bce_loss
from augmentations import *
from torch.autograd import Variable as V
from networks.unet import UNet34
from networks.dinknet import DinkNet34
from networks.hdcducnet import ResNetDUCHDC34, ResNetDUC34, DinkNetHDC18
from optimizer import Optimizer
from pytz import timezone
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='Trainer')

parser.add_argument('-l', '--lr', type=float, metavar='learning_rate', required=True, help='Learning Rate')

parser.add_argument('-b', '--batchsize', type=int, metavar='batchsize', required=True, help='Batch size')

parser.add_argument('-e', '--epochs', type=int, metavar='epochs', required=True, help='Epochs')

parser.add_argument('-s', '--save', type=str, metavar='savename', required=True, help='Save file name')

parser.add_argument('-w', '--weight', type=str, metavar='weightspath', required=False, help='Weights file path')

parser.add_argument('-o', '--optimizer', type=str, metavar='optimizer path', required=False, help='Optimizer file path')

parser.add_argument('-t', '--train', type=str, metavar='trainpath', required=True, help='Path to data train folder')

parser.add_argument('-i', '--idevices', type=str, metavar='deviceids', required=True, help='Device ids')

parser.add_argument("network", type=str, nargs=1, help='unet34, dinknet34, dinkducnet34')

arguments = parser.parse_args()

dirs = ('submits', 'weights', 'logs')

for dir in dirs:
    if not os.path.isdir(dir):
        os.mkdir(dir)

available_nets = ("unet", "dinknet", "resnetduchdc", "resnetduc", "dinknethdc")

assert sys.argv[-1].lower() in available_nets

SHAPE = (1024,1024)
ROOT = arguments.train if arguments.train.endswith('/') else arguments.train + '/'
imagelist = list(filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT)))
trainlist = list(map(lambda x: x[:-8], imagelist))
NAME = arguments.save
BATCHSIZE_PER_CARD = arguments.batchsize

ids = [int(x) for x in arguments.idevices.split(',')]

torch.cuda.set_device(ids[0])

solver = Optimizer(UNet34 if sys.argv[-1].lower() == 'unet' \
            else DinkNet34 if sys.argv[-1].lower() == 'dinknet' \
            else ResNetDUCHDC34 if sys.argv[-1].lower() == 'resnetduchdc' \
            else DinkNetHDC18 if sys.argv[-1].lower() == 'dinknethdc' \
            else ResNetDUC34 if sys.argv[-1].lower() == 'resnetduc' \
            else None, \
            dice_bce_loss, \
            ids, \
            arguments.lr, \
            arguments.optimizer)

if arguments.weight != None:
    solver.load(arguments.weight)
batchsize = len(ids) * BATCHSIZE_PER_CARD
dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize)

mylog = open('logs/'+NAME+'.log','w')
tic = time()
no_optim = 0
total_epoch = arguments.epochs
train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    count = 1
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
        if count % 30 == 0:
            print("\tbatch ({}/{}) loss = {}".format(count, len(data_loader_iter), train_epoch_loss / count))
        count += 1
    train_epoch_loss /= len(data_loader_iter)
    print('********\t{}'.format(datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")), file=mylog)
    print('epoch:',epoch,'    time:',int(time()-tic), file=mylog)
    print('train_loss:',train_epoch_loss, file=mylog)
    print('SHAPE:',SHAPE, file=mylog)
    print('********\t{}'.format(datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True, mylog = mylog)
    mylog.flush()

print('Finish!')
