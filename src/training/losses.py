import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)

        return 0.5 * bce_loss + 0.5 * dice_loss