# losses.py
import torch
import torch.nn.functional as F


def dice_loss(y_pred, y_true, smooth=1e-5):
    ndims = len(y_pred.shape) - 2
    vol_axes = list(range(2, ndims + 2))
    intersection = 2 * (y_true * y_pred).sum(dim=vol_axes)
    union = y_true.sum(dim=vol_axes) + y_pred.sum(dim=vol_axes)
    dice = (intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def cross_entropy_loss(pred, target):
    return F.cross_entropy(pred, target.argmax(dim=1))


def composite_loss(pred, target):
    dice_loss_val = dice_loss(pred, target)
    ce_loss_val = cross_entropy_loss(pred, target)
    return 0.7 * dice_loss_val + 0.3 * ce_loss_val