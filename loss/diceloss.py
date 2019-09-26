from .BCESSIM import BCESSIM
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.utils.data as data
from torchvision import models
import torch.nn.functional as F
import cv2
import os
from functools import partial
from time import time
import numpy as np

class diceloss(nn.Module):
    def __init__(self, batch=True):
        super().__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)

        return score.mean()

    def forward(self, y_true, y_pred):
        return 1 - self.soft_dice_coeff(y_true, y_pred)
