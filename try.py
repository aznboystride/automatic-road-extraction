import numpy as np
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

# SMOOTH = 1e-6

# def iou_pytorch(outputs, labels):
#     outputs[outputs>=0.5] = 1
#     outputs[outputs<0.5] = 0
#     outputs = outputs.int()
#     labels = labels.int()
#     # You can comment out this line if you are passing tensors of equal shape
#     # But if you are passing output from UNet or something it will most probably
#     # be with the BATCH x 1 x H x W shape
#     outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
#
#     intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
#     union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
#
#     iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
#
#     #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
#
#     return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch


def validate(model, trainloader):
    model.eval()
    global criterion
    running_loss = 0
    counter = batch_multiplier
    batchloss = 0
    batchcount = 0
    for i, (inputs, labels) in enumerate(trainloader, 1):
        if (len(trainloader) - i + 1) < args.batch:
            break
        inputs = inputs.cuda()
        labels = labels.cuda()
        if counter == 0:
            counter = batch_multiplier
            running_loss += batchloss
            batchcount += 1
            batchloss = 0

        outputs = model(inputs)
        loss = criterion(outputs, labels) / batch_multiplier
        batchloss += loss.item()
        counter -= 1
    print("[*] validation -- loss %.5f" % (running_loss / batchcount))

    model.train()


class ValidDataset(data.Dataset):
    def __init__(self):
        self.iml = list(filter(lambda x: x.find('sat') != -1, os.listdir('valid')))
        self.trl = list(map(lambda x: x[:-8], self.iml))

    def __getitem__(self, index):
        id = self.trl[index]
        img = cv2.imread(os.path.join('valid','{}_sat.jpg').format(id))
        mask = cv2.imread(os.path.join('valid', '{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2,0,1)/255.0
        mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return img, mask

    def __len__(self):
        return len(self.iml)


class Dataset(data.Dataset):

    def __init__(self, test, augment=None):
        from loader import Loader
        self.loader = Loader('train', test, augment)

    def __getitem__(self, index):
        return self.loader(index)

    def __len__(self):
        return len(self.loader)


args = parser.parse_args()

# def update_lr(optimizer,net):
#     with open('learning_rate', 'r') as f:
#         lr = float(f.read())
#     if lr == args.lr:
#         return
#     print("New learning rate {} -> {}".format(args.lr, lr))
#     args.lr = lr
#     net.load_state_dict(torch.load("weights/{}.pth".format(args.weights)))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

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

validloader = torch.utils.data.DataLoader(ValidDataset(), batch_size=len(ids)*4, shuffle=True)

criterion = nn.BCELoss() if not criterion else criterion

optimizer = optim.Adam(model.parameters()) if not optimizer else optimizer

print('Training start')
print('Arguments -> {}'.format(' '.join(sys.argv)))
best_loss = len(trainloader) * 100
batch_multiplier = args.batch / (len(ids)*4)
# with open('learning_rate', 'w+') as f:
#     f.write(str(args.lr))

no_optim = 0
best_train_loss = len(trainloader) * 100
for epoch in range(1, args.iterations + 1):
    print("[+] Epoch ({}/{}) - {}".format(epoch, args.iterations,  datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
    running_loss = 0
    counter = batch_multiplier
    batchloss = 0
    batchcount = 0
    for i, (inputs, labels) in enumerate(trainloader, 1):
        if (len(trainloader) - i + 1) < args.batch:
            break
        inputs = inputs.cuda()
        labels = labels.cuda()
        if counter == 0:
            optimizer.step()
            optimizer.zero_grad()
            counter = batch_multiplier
            running_loss += batchloss
            batchcount += 1
            # if batchcount % args.stats == 0:
            #     print('training -- [%d, %5d] %s loss: %.5f time: %s' %
            #             (epoch, batchcount, criterion.__class__.__name__, running_loss/batchcount, datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
            batchloss = 0
        counter -= 1

        outputs = model(inputs)
        loss = criterion(outputs, labels) / batch_multiplier
        loss.backward()
        batchloss += loss.item()
    
    if running_loss/batchcount < best_train_loss/batchcount:
        print('[+] training -- new better loss  %.5f -> %.5f ' % (best_train_loss/batchcount, running_loss/batchcount))
        best_train_loss = running_loss
        torch.save(model.state_dict(), "weights/" + args.weights + ".pth")
        torch.save(optimizer.state_dict(), "optimizers/" + args.weights + ".pth")
        no_optim = 0
    elif no_optim >= 3:
        print("[-] training -- loss %.5f" % (running_loss / batchcount))
        model.load_state_dict(torch.load("weights/{}.pth".format(args.weights)))
        lr = args.lr/5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print("[!] New learning rate %.5f -> %.5f" % (args.lr, lr))
        args.lr = lr
        no_optim += 1
    elif no_optim == 7:
        print("[-] training -- loss %.5f" % (running_loss / batchcount))
        print("[!] Early stop")
    else:
        print('[-] training -- loss %.5f ' % (running_loss/batchcount))
        no_optim += 1

    with torch.no_grad():
        validate(model, validloader)

print('Finished training')

