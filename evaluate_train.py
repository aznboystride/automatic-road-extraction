
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
from sklearn.metrics import jaccard_score

parser = argparse.ArgumentParser(description='trainer')

parser.add_argument('-b',   '--batch',          type=int,   required=True,  dest='batch',       help='batch size')
parser.add_argument('-it',  '--iterations',     type=int,   required=True,  dest='iterations',  help='# of iterations')
parser.add_argument('-dv',  '--devices',        type=str,   required=True,  dest='devices',     help='gpu indices sep. by comma')
parser.add_argument('-lw',  '--lweights',       type=str,   required=False, dest='lweights',    help='name of weights file to load')
parser.add_argument('-au'   '--augment',        type=str,   required=False, dest='augment',     help='name of augmentation')
parser.add_argument('-s'    '--stats',          type=int,   required=False, dest='stats',       help='print statistics')
parser.add_argument('model', type=str, help='name of model')

args = parser.parse_args()

class Dataset(data.Dataset):

    def __init__(self, test, augment=None):
        from loader import Loader
        self.loader = Loader('train', test, augment)

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

ids = [int(x) for x in args.devices.split(',')] if args.devices else None

torch.cuda.set_device(ids[0])
model = torch.nn.DataParallel(model, device_ids=ids)
model.cuda()

if args.lweights:
    model.load_state_dict(torch.load("weights/{}.pth".format(args.lweights)))

dataset = Dataset(test=False, augment=augment)

trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=len(ids)*4,
    drop_last=True,
    shuffle=True)

criterion = nn.BCELoss()

print('Train Evaluation Started')
print('Arguments -> {}'.format(' '.join(sys.argv)))

batch_multiplier = args.batch / (len(ids)*4)

no_optim = 0
model.eval()
with torch.no_grad():
    total_loss = 0
    for epoch in range(1, args.iterations + 1):
        running_loss = 0
        counter = batch_multiplier
        batchloss = 0
        batchcount = 0
        for i, (inputs, labels) in enumerate(trainloader,1):
            if (len(trainloader) - i + 1) < args.batch:
                break
            inputs = inputs.cuda()
            labels = labels.cuda()
            if counter == 0:
                counter = batch_multiplier
                running_loss += batchloss
                batchcount += 1
                if batchcount % args.stats == 0:
                    print('[%d, %5d] loss: %.5f time: %s' %
                            (epoch, batchcount, running_loss/batchcount, datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
                batchloss = 0
                
            outputs = model(inputs)
            loss = criterion(outputs, labels) / batch_multiplier
            batchloss += loss.item()
            counter -= 1
        
        print('Epoch Total Loss: {:.5f}'.format(running_loss / batchcount))
        total_loss += (running_loss / batchcount)

print('Finished Train Evaluation')
print('Average Epoch Loss: {:.5f}'.format(total_loss / args.iterations))
print('Weights Used {}.pth'.format(args.lweights))
