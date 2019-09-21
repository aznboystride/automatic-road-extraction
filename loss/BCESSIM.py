import torch.nn as nn
from loss.SSIM import SSIM

class BCESSIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.ssim = SSIM()

    def forward(self, outputs, labels):
        bce = self.bce_loss(outputs, labels)
        ssim= self.ssim(outputs, labels)
        return bce + 1 - ssim
