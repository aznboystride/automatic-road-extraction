import torch
import torch.nn as nn
from .diceloss import diceloss
import cv2
import numpy as np
class dice_bce_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.diceloss = diceloss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, y_true, y_pred):
        return self.diceloss(y_true, y_pred) + self.bce_loss(y_true, y_pred)  
