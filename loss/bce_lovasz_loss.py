import torch.nn as nn
from .lovasz_loss import lovasz_loss

class bce_lovasz_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.lovaszloss = lovasz_loss()

    def forward(self, pred, label):
        return self.bce_loss(pred, label) + self.lovaszloss(pred, label, per_image=False)

