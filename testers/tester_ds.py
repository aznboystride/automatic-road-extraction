import cv2
import numpy as np
import torch

class tester_ds:
    def __init__(self, net, batchsize=1):
        self.net = net
        self.batch = batchsize
    
    def __call__(self, path):
        image = cv2.imread(path)
        image = np.transpose(image, (2,0,1))
        image = torch.as_tensor(image).unsqueeze(0).float().cuda()
        image /= 255.0
        # outputs = self.net(image).squeeze().squeeze().cpu().data.numpy()
        image = self.net(image.cuda())
        image = [i.squeeze().squeeze().cpu().data.numpy() for i in image]
        image = (image[0] + image[1] + image[2] + image[3] + image[4]) / 5
        image[image<0.5] = 0
        image[image>=0.5] = 255.0
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
'''
    def __call__(self, inputs):
        image = self.net(inputs.cuda()).squeeze().squeeze().cpu().data.numpy()
        image[image>0.5] = 255
        image[image<=0.5] = 0
        return cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
'''
