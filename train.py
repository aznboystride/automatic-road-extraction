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
import math
#from time import time
#from sklearn.metrics import jaccard_score as jsc

class ValidDataset(data.Dataset):
    def __init__(self):
        self.iml = list(filter(lambda x: x.find('sat') != -1, os.listdir('valid')))
        self.trl = list(map(lambda x: x[:-8], self.iml))

    def __getitem__(self, index):
        id = self.trl[index]
        img = cv2.imread(os.path.join('valid', '{}_sat.jpg').format(id))
        mask = cv2.imread(os.path.join('valid', '{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=2)
        img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0
        mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
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


parser = argparse.ArgumentParser(description='trainer')

parser.add_argument('-lr',  '--learning_rate',  type=float, required=True,  dest='lr',          help='learning rate') 
parser.add_argument('-b',   '--batch',          type=int,   required=True,  dest='batch',       help='batch size')
parser.add_argument('-it',  '--iterations',     type=int,   required=True,  dest='iterations',  help='# of iterations')
parser.add_argument('-dv',  '--devices',        type=str,   required=True,  dest='devices',     help='gpu indices sep. by comma')
parser.add_argument('-lw',  '--lweights',       type=str,   required=False, dest='lweights',    help='name of weights file to load')
parser.add_argument('-au',   '--augment',       type=str,   required=False, dest='augment',     help='name of augmentation')
parser.add_argument('-ls',  '--loss',           type=str,   required=False, dest='loss',        help='name of loss')
parser.add_argument('-e',   '--epoch',          type=int,   required=False, dest='epoch',       help='epoch to start')
parser.add_argument('model', type=str, help='name of model')

MAX_BATCH_PER_CARD = 4
SMOOTH = 1e-6
'''
What should I save for validation? Best loss and Accuracy.
Should I save the optimizer? Just save best training loss optimizer.
What should I name the validation weights? val_loss_<modelname>_<criterion>_<loss>.pth
Name validation and accuracy base on criterion used / model and quantity.
Loss should be bce + ssim.
'''

minValLoss = float('inf')
maxValAcc = 0.0
def iou(outputs, labels):
    outputs = outputs >= 0.5
    labels = labels >= 0.5
    acc = 0
    for out, lab in zip(outputs, labels):
        intersection = (lab & out).int().sum().float().item() + SMOOTH
        union = (lab | out).int().sum().float().item() + SMOOTH
        acc += (intersection/union)
    ret = acc / len(outputs)
    return ret

def validate():
    global minValLoss
    global maxValAcc
    model.eval() 
    print("[+] Validating.. - {}".format(datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
    running_loss = 0
    running_acc = 0
    counter = batch_multiplier
    batchloss = 0
    batchacc  = 0
    batchcount = 0
    for i, (inputs, labels) in enumerate(validloader, 1):
        if len(validloader) + 1 - i < args.batch:
            break
        inputs = inputs.cuda()
        labels = labels.cuda()
        if counter == 0:
            counter = batch_multiplier
            running_loss += batchloss
            running_acc += batchacc
            batchcount += 1
            batchloss = 0
            batchacc = 0
        
        outputs = model(inputs)
        loss = criterion(outputs, labels).item() / batch_multiplier
        acc = iou(outputs, labels) / batch_multiplier
        batchloss += loss
        batchacc += acc
        counter -= 1

    if running_loss / batchcount < minValLoss:
        print('[+] validation -- new better loss  {:.5f} -> {:.5f}\n'.format(minValLoss, running_loss / batchcount))
        old_path = 'weights/val_loss_{}_{}_{:.5f}.pth'.format(args.model, criterion.__class__.__name__, minValLoss)
        if os.path.exists(old_path):
            os.system('rm ' + old_path)
        minValLoss = running_loss / batchcount
        savepath = 'weights/val_loss_{}_{}_{:.5f}.pth'.format(args.model, criterion.__class__.__name__, minValLoss)
        torch.save(model.state_dict(), savepath)
    else:
        print("[-] validation -- loss {:.5f}\n".format(running_loss / batchcount))


    if running_acc / batchcount > maxValAcc:
        print('[+] validation -- new better acc  {:.5f} -> {:.5f}\n'.format(maxValAcc, running_acc / batchcount))
        old_path = 'weights/val_acc_{}_{}_{:.5f}.pth'.format(args.model, "iouscore", maxValAcc)
        if os.path.exists(old_path):
            os.system('rm ' + old_path)
        maxValAcc = running_acc / batchcount
        savepath = 'weights/val_acc_{}_{}_{:.5f}.pth'.format(args.model, "iouscore", maxValAcc)
        torch.save(model.state_dict(), savepath)
    else:
        print("[-] validation -- acc {:.5f}\n".format(running_acc / batchcount))

    model.train()

args = parser.parse_args()

args.epoch = 1 if not args.epoch else args.epoch

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

optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
if args.lweights:
    model.load_state_dict(torch.load("weights/{}".format(args.lweights)))
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer.load_state_dict(torch.load("optimizers/{}".format(args.lweights)))

dataset = Dataset(test=False, augment=augment)

trainloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=len(ids) * MAX_BATCH_PER_CARD,
    shuffle=True)

validloader = torch.utils.data.DataLoader(ValidDataset(), batch_size=len(ids) * 4, shuffle=True)

criterion = nn.BCELoss() if not criterion else criterion

print('Training start')
print('Arguments -> {}'.format(' '.join(sys.argv)))
batch_multiplier = args.batch / (len(ids) * MAX_BATCH_PER_CARD)

minTrainLoss = float('inf')
maxTrainAcc = 0
for epoch in range(args.epoch, args.iterations + args.epoch):
    print("[+] Epoch ({}/{}) - {}".format(epoch, args.iterations + args.epoch,
                                          datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
    running_loss = 0
    running_acc = 0
    counter = batch_multiplier
    batchloss = 0
    batchacc = 0
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
            running_acc += batchacc
            batchcount += 1
            batchloss = 0
            batchacc = 0
        counter -= 1
        outputs = model(inputs) 
        loss = criterion(outputs, labels) / batch_multiplier
        loss.backward()
        with torch.no_grad():
            acc = iou(outputs, labels) / batch_multiplier
        batchloss += loss.item()
        batchacc += acc

    if running_loss / batchcount < minTrainLoss:
        print('[+] train -- new better loss  {:.5f} -> {:.5f}\n'.format(minTrainLoss, running_loss / batchcount))
        old_path = 'weights/train_loss_{}_{}_{:.5f}.pth'.format(args.model, criterion.__class__.__name__, minTrainLoss)
        if os.path.exists(old_path):
            os.system('rm ' + old_path)
            os.system('rm ' + old_path.replace('weights', 'optimizers'))
        minTrainLoss = running_loss / batchcount
        savepath = 'weights/train_loss_{}_{}_{:.5f}.pth'.format(args.model, criterion.__class__.__name__, minTrainLoss)
        torch.save(model.state_dict(), savepath)
        torch.save(optimizer.state_dict(), savepath.replace("weights", "optimizers"))
    else:
        print("[-] train -- loss {:.5f}\n".format(running_loss / batchcount))

    assert not math.isnan(maxTrainAcc)
    assert not math.isnan(running_acc)
    assert not math.isnan(batchcount)
    assert not math.isnan(running_acc/batchcount)
    if running_acc / batchcount > maxTrainAcc:
        print('[+] train -- new better acc  {:.5f} -> {:.5f}\n'.format(maxTrainAcc, running_acc / batchcount))
        old_path = 'weights/train_acc_{}_{}_{:.5f}.pth'.format(args.model, "iouscore", maxTrainAcc)
        if os.path.exists(old_path):
            os.system('rm ' + old_path)
        maxTrainAcc = running_acc / batchcount
        savepath = 'weights/train_acc_{}_{}_{:.5f}.pth'.format(args.model, "iouscore", maxTrainAcc)
        torch.save(model.state_dict(), savepath)
    else:
        print("[-] train -- acc {:.5f}\n".format(running_acc / batchcount))

    with torch.no_grad():
        validate()

print('Finished Training')

