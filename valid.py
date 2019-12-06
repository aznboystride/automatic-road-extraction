import os
import sys
import cv2
import numpy as np
import argparse
import importlib
import torch
import torch.utils.data as data
from datetime import datetime
from pytz import timezone


parser = argparse.ArgumentParser(description='tester')

parser.add_argument('-wt',  '--weights',        type=str,   required=True,  dest='weights',     help='path to weights file')
parser.add_argument('-ts',  '--tester',         type=str,   required=True,  dest='tester',      help='name of tester')
parser.add_argument('-dv',  '--devices',        type=str,   required=True,  dest='devices',     help='gpu indices sep. by comma')
parser.add_argument('-s',   '--stats',          type=int,   required=False, dest='stats',       help='print statistics')
parser.add_argument('-f1',   '--f1score',       action='store_true',        dest='f1score',     help='Use f1 scoring?')
parser.add_argument('model', type=str, help='name of model')

labelDir = 'test'
def iou(outputImage, filename):
    labelFilePath = os.path.join(labelDir, filename.replace("_sat.jpg", "_mask.png")) # Gives us labelDir/id_mask.png ( True Mask )
    labelImage = cv2.imread(labelFilePath, cv2.IMREAD_GRAYSCALE)/255.0 # Reads labelDir/true_mask.png as grey scale
    outputImage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY)/255.0 # Convert RGB Numpy Output To GreyScale
    # Convert to Tensor
    labelImage = torch.from_numpy(labelImage)
    outputImage = torch.from_numpy(outputImage)
    # Thresh hold
    labelImage = labelImage >= 0.5
    outputImage = outputImage >= 0.5
    intersection = (labelImage & outputImage).int().sum().float().item()
    union = (labelImage | outputImage).int().sum().float().item()
    return intersection / union

def f1(outputImage, filename):
    labelFilePath = os.path.join(labelDir, filename.replace("_sat.jpg", "_mask.png")) # Gives us labelDir/id_mask.png ( True Mask )
    labelImage = cv2.imread(labelFilePath, cv2.IMREAD_GRAYSCALE)/255.0 # Reads labelDir/true_mask.png as grey scale
    outputImage = cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY)/255.0 # Convert RGB Numpy Output To GreyScale
    # Convert to Tensor
    labelImage = torch.from_numpy(labelImage)
    outputImage = torch.from_numpy(outputImage)
    # Thresh hold
    labelRoad = labelImage >= 0.5
    outputRoad = outputImage >= 0.5

    labelBackground = labelImage < 0.5
    outputBackground = outputImage < 0.5

    TP = (labelRoad & outputRoad).int().sum().float().item()
    FP = (labelBackground & outputRoad).int().sum().float().item()
    FN = (labelRoad & outputBackground).int().sum().float().item()
    return 2 * TP / (2 * TP + FN + FP)


class Dataset(data.Dataset):

    def __init__(self, test, augment=None):
        from loader import Loader
        if not test:
            self.loader = Loader('train', test, augment)
        else:
            self.loader = Loader(labelDir, test, augment)

    def __getitem__(self, index):
        return self.loader(index)

    def __len__(self):
        return len(self.loader)

args = parser.parse_args()

args.stats = 30 if not args.stats else args.stats

metric = f1 if args.f1score else iou

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

model.load_state_dict(torch.load(os.path.join('weights', args.weights), map_location={'cuda:2': 'cuda:0', 'cuda:3':'cuda:0'}))

dataset = Dataset(test=True, augment=None)

testloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True)

print('Testing start')
print('Arguments -> {}'.format(' '.join(sys.argv)))

model.eval()
tester = tester(model, batchsize=8)

with torch.no_grad():
    miou = 0
    for i, (file, inputs) in enumerate(testloader):
        image = tester(os.path.join(labelDir, file[0].replace('_mask.png', '_sat.jpg'))) # RGB Numpy Output
        m = metric(image, file[0])
        miou += m
        if i % (args.stats-1) == 0:
            print('{}/{}\t{}'.format(i+1,len(testloader),datetime.now(timezone("US/Pacific")).strftime("%m-%d-%Y - %I:%M %p")))
    print("%s MIOU: %.5f" % (args.tester, miou / len(testloader)))
