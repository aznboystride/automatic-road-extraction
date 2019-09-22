
import os
import sys
import cv2
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from datetime import datetime
from pytz import timezone

parser = argparse.ArgumentParser(description='trainer')

parser.add_argument('-lr',  '--learning_rate',  type=float, required=True,  dest='lr',          help='learning rate')
parser.add_argument('-b',   '--batch',          type=int,   required=True,  dest='batch',       help='batch size')
parser.add_argument('-it',  '--iterations',     type=int,   required=True,  dest='iterations',  help='# of iterations')
parser.add_argument('-dv',  '--devices',        type=str,   required=True,  dest='devices',     help='gpu indices sep. by comma')
parser.add_argument('-wt',  '--weights',        type=str,   required=False, dest='weights',     help='path to weights file')
parser.add_argument('-au'   '--augment',        type=str,   required=False, dest='augment',     help='name of augmentation')
parser.add_argument('-s'    '--stats',          type=int,   required=False, dest='stats',       help='print statistics')
parser.add_argument('-ls',  '--loss',           type=str,   required=False, dest='loss',        help='name of loss')
parser.add_argument('model', type=str, help='name of model')

args = parser.parse_args()

class Dataset(data.Dataset):

    def __init__(self, augment=None):
        from loader import Loader
        self.loader = Loader('train', augment)

    def __getitem__(self, index):
        return self.loader(index)

    def __len__(self):
        return len(self.loader)

args = parser.parse_args()

args.stats = 30 if not args.stats else args.stats

# Get Attributes From Modules
model = importlib.import_module('networks.{}'.format(args.model))

model = getattr(model, args.model)()

augment = None

if args.augment:
    augment = importlib.import_module('augments.{}'.format(args.augment))
    augment = getattr(augment, 'augment')

criterion = None

if args.loss:
    criterion = importlib.import_module('loss.{}'.format(args.loss))
    criterion = getattr(criterion, args.loss)()

# Get Attributes From Modules End

ids = [int(x) for x in args.devices.split(',')] if args.devices else None

if ids:
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=ids)
    torch.cuda.set_device(ids[0])

dataset = Dataset(augment)

trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch,
    shuffle=True)

criterion = nn.BCELoss() if not criterion else criterion

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

print('Training start')
print('Arguments -> {}'.format(' '.join(sys.argv)))
for epoch in range(1, args.iterations + 1):
    running_loss = 0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs.cuda())
        loss = criterion(outputs.cuda(), labels.cuda())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % (args.stats-1) == 0:
            print('[%d, %5d] loss: %.3f time: %s' %
                    (epoch, i + 1, running_loss / (i+1), datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
    print('Finished training')
