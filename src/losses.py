"""
Losses. The problem statement asks for ONE loss shared by both models.

We use the compound BCE + Dice loss (chosen on the validation split, see
tuning.py — BCE alone and Dice alone are the other candidates):

    L_BCE  = -(1/N) * sum_i [ g_i * log(p_i) + (1 - g_i) * log(1 - p_i) ]

    L_Dice = 1 - (2 * sum_i p_i * g_i + eps) / (sum_i p_i + sum_i g_i + eps)

    L      = lambda_bce * L_BCE + lambda_dice * L_Dice        (lambda = 1, 1)

    p_i = sigmoid(logit_i) is the predicted horse probability of pixel i,
    g_i in {0,1} is the ground truth, eps = 1 avoids division by zero.

Why: BCE gives a smooth, well-behaved gradient for every pixel (good for
early training), while Dice scores the overlap of the WHOLE mask and so is
not dominated by the background pixels; it is also directly related to the
F1 / IoU we report. Dice is computed per image and then averaged so that
small horses count as much as large ones.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float())
    t = target.float()
    dims = (1, 2, 3)                                   # per image
    inter = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, name="bce_dice", w_bce=1.0, w_dice=1.0):
        super().__init__()
        assert name in ("bce", "dice", "bce_dice")
        self.name, self.w_bce, self.w_dice = name, w_bce, w_dice

    def forward(self, logits, target):
        target = target.float()
        if self.name == "bce":
            return F.binary_cross_entropy_with_logits(logits.float(), target)
        if self.name == "dice":
            return soft_dice_loss(logits, target)
        return (self.w_bce * F.binary_cross_entropy_with_logits(logits.float(), target)
                + self.w_dice * soft_dice_loss(logits, target))
