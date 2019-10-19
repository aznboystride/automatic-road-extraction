from skorch.net import NeuralNet
import argparse
from torch.optim import SGD
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring, ProgressBar
from skorch.callbacks.lr_scheduler import CyclicLR
import torch.optim as optim


parser = argparse.ArgumentParser(description='trainer')

parser.add_argument('-lr',  '--learning_rate',  type=float, required=True,  dest='lr',          help='learning rate')
parser.add_argument('-b',   '--batch',          type=int,   required=True,  dest='batch',       help='batch size')
parser.add_argument('-it',  '--iterations',     type=int,   required=True,  dest='iterations',  help='# of iterations')
parser.add_argument('-dv',  '--devices',        type=str,   required=False,  dest='devices',     help='gpu indices sep. by comma')
parser.add_argument('-wt',  '--weights',        type=str,   required=True,  dest='weights',     help='path to weights file')
parser.add_argument('-lw',  '--lweights',       type=str,   required=False, dest='lweights',    help='name of weights file to load')
parser.add_argument('-au'   '--augment',        type=str,   required=False, dest='augment',     help='name of augmentation')
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

if args.loss:
    criterion = importlib.import_module('loss.{}'.format(args.loss))
    criterion = getattr(criterion, args.loss)()

# Get Attributes From Modules End

ids = [int(x) for x in args.devices.split(',')] if args.devices else None

criterion = nn.BCELoss() if not criterion else criterion

if ids:
    torch.cuda.set_device(ids[0])
    model = torch.nn.DataParallel(model, device_ids=ids)
    model.cuda()

if args.lweights:
    model.load_state_dict(torch.load("weights/{}.pth".format(args.lweights)))

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

cyclicLR = LRScheduler(policy=CyclicLR, 
                       base_lr=args.lr,
                       max_lr=0.1,
                       step_size_up=1530,
                       step_size_down=1530)

dataset = Dataset(test=False, augment=augment)

def score(net, ds, y):
    pred = net(ds)
    return criterion(pred, y)

score = EpochScoring(score, name='bce_loss', lower_is_better=True)

net = NeuralNet(
    model,
    criterion=criterion.__class__,
    batch_size=args.batch,
    max_epochs=args.iterations,
    optimizer=optimizer.__class__,
    optimizer__momentum=0.9,
    train_split=None,
    callbacks=[ cyclicLR, 
               score,
               Checkpoint(f_params='best_params.pt'), ProgressBar()],
    device='cuda'
)

net.fit(dataset)

