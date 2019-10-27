import os
import sys
import cv2
import argparse
import importlib
import torch
import torch.utils.data as data
from datetime import datetime
from pytz import timezone


parser = argparse.ArgumentParser(description='tester')

parser.add_argument('-b',   '--batch',          type=int,   required=True,  dest='batch',       help='batch size')
parser.add_argument('-wt',  '--weights',        type=str,   required=True,  dest='weights',     help='path to weights file')
parser.add_argument('-ts'   '--tester',         type=str,   required=True,  dest='tester',      help='name of tester')
parser.add_argument('-dv',  '--devices',        type=str,   required=True,  dest='devices',     help='gpu indices sep. by comma')
parser.add_argument('-s'    '--stats',          type=int,   required=False, dest='stats',       help='print statistics')
parser.add_argument('model', type=str, help='name of model')

def iou(om, filename):
    import numpy as np
    vpath = os.path.join("valid", filename.replace("_sat.jpg", "_mask.png")) # Gives us valid/id_mask.png ( True Mask )

    vm = cv2.imread(vpath, cv2.IMREAD_GRAYSCALE) # Reads valid/true_mask.png as grey scale
    om = cv2.cvtColor(om, cv2.COLOR_RGB2GRAY) # Convert RGB Numpy Output To GreyScale
    intersection = np.sum(om*vm) # Intersection of all 1
    union = np.sum(om + vm) # Union of all 1

    return intersection / union

class Dataset(data.Dataset):

    def __init__(self, test, augment=None):
        from loader import Loader
        if not test:
            self.loader = Loader('train', test, augment)
        else:
            self.loader = Loader('test', test, augment)

    def __getitem__(self, index):
        return self.loader(index)

    def __len__(self):
        return len(self.loader)

args = parser.parse_args()

args.stats = 30 if not args.stats else args.stats

# Get Attributes From Modules
model = importlib.import_module('networks.{}'.format(args.model))

model = getattr(model, args.model)()

tester = importlib.import_module('testers.{}'.format(args.tester))
tester = getattr(tester, args.tester)
# Get Attributes From Modules End

ids = [int(x) for x in args.devices.split(',')] if args.devices else None

torch.cuda.set_device(ids[0])
model = model.cuda()
model = torch.nn.DataParallel(model, device_ids=ids)

model.load_state_dict(torch.load(os.path.join('weights', args.weights+".pth")))

dataset = Dataset(test=False, augment=None)

testloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True)

print('Testing start')
print('Arguments -> {}'.format(' '.join(sys.argv)))

model.eval()
tester = tester(model, batchsize=args.batch)

with torch.no_grad():
    miou = 0
    for i, (file, inputs) in enumerate(testloader):
        image = tester(os.path.join('valid', file[0].replace('_mask.png', '_sat.jpg'))) # RGB Numpy Output
        m = iou(image, file[0])
        miou += m
        if i % (args.stats-1) == 0:
            print('{}/{}\t{}'.format(i+1,len(testloader),datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
    print("MIOU: %.5f" % (miou / len(testloader)))