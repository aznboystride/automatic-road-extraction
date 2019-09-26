import torch.nn as nn
from .BCESSIM import BCESSIM
from .diceloss import diceloss

class dice_bce_ssim(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_ssim = BCESSIM()
        self.diceloss = diceloss()

    def forward(self, outputs, labels):
        bce_ssim = self.bce_ssim(outputs, labels)
        diceloss = self.diceloss(outputs, labels)
        return bce_ssim + diceloss
