
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
parser.add_argument('-wt',  '--weights',        type=str,   required=True,  dest='weights',     help='path to weights file')
parser.add_argument('-lw',  '--lweights',       type=str,   required=False, dest='lweights',    help='name of weights file to load')
parser.add_argument('-au'   '--augment',        type=str,   required=False, dest='augment',     help='name of augmentation')
parser.add_argument('-s'    '--stats',          type=int,   required=False, dest='stats',       help='print statistics')
parser.add_argument('-ls',  '--loss',           type=str,   required=False, dest='loss',        help='name of loss')
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

def update_lr(optimizer,net):
    with open('learning_rate', 'r') as f:
        lr = float(f.read())
    if lr == args.lr:
        return
    print("New learning rate {} -> {}".format(args.lr, lr))
    args.lr = lr
    net.load_state_dict(torch.load("weights/{}.pth".format(args.weights)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

torch.cuda.set_device(ids[0])
model = torch.nn.DataParallel(model, device_ids=ids)
model.cuda()

optimizer = None 
if args.lweights:
    model.load_state_dict(torch.load("weights/{}.pth".format(args.lweights)))
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(torch.load("optimizers/{}.pth".format(args.lweights)))

dataset = Dataset(test=False, augment=augment)

trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=len(ids)*4,
    drop_last=True,
    shuffle=True)

criterion = nn.BCELoss() if not criterion else criterion

optimizer = optim.Adam(model.parameters()) if not optimizer else optimizer

print('Training start')
print('Arguments -> {}'.format(' '.join(sys.argv)))
best_loss = len(trainloader) * 100
batch_multiplier = args.batch / (len(ids)*4)
with open('learning_rate', 'w+') as f:
    f.write(str(args.lr))


bce = nn.BCELoss()
no_optim = 0
for epoch in range(1, args.iterations + 1):
    update_lr(optimizer, model)
    running_loss = 0
    running_loss_bce = 0
    counter = batch_multiplier
    batchloss = 0
    batchloss_bce = 0
    batchcount = 0
    for i, (inputs, labels) in enumerate(trainloader,1):
        if (len(trainloader) - i + 1) < args.batch:
            break
        inputs = inputs.cuda()
        labels = labels.cuda()
        if counter == 0:
            optimizer.step()
            optimizer.zero_grad()
            counter = batch_multiplier
            running_loss += batchloss
            running_loss_bce += batchloss_bce
            batchcount += 1
            if batchcount % args.stats == 0:
                print('[%d, %5d] %s loss: %.5f time: %s\t %s loss: %.5f' %
                        (epoch, batchcount, criterion.__class__.__name__, running_loss/batchcount, datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p"), bce.__class__.__name__, running_loss_bce/batchcount))
            batchloss = 0
            batchloss_bce = 0
            
        outputs = model(inputs)
        loss = criterion(outputs, labels) / batch_multiplier
        loss.backward()

        with torch.no_grad():
            loss_bce = bce(outputs, labels) / batch_multiplier
            batchloss_bce += loss_bce.item()

        batchloss += loss.item()
        counter -= 1
    if best_loss / batchcount > running_loss / batchcount:
        print("new better loss %.5f" % (running_loss / batchcount))
        best_loss = running_loss
        torch.save(model.state_dict(), "weights/" + args.weights + ".pth")
        torch.save(optimizer.state_dict(), "optimizers/" + args.weights + ".pth")
        no_optim = 0
    elif no_optim >= 3:
        with open('learning_rate', 'w+') as f:
            f.write(str(args.lr/5))
        update_lr(optimizer, model)
        no_optim += 1
    elif no_optim == 7:
        print('Early stop')
        break
    else:
        no_optim += 1
print('Finished training')
