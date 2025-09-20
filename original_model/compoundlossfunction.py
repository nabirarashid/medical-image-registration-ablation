import torch
from losses import chamfer_distance_loss, hausdorff_distance_loss, surface_loss, bending_energy_loss

# This is to experiment with different variants of loss functions

from losses import (
    chamfer_distance_loss,
    hausdorff_distance_loss,
    surface_loss,
    bending_energy_loss,
    jacobian_det_loss,
    label_overlap_loss,
    deformation_direction_variation,
    dice_loss
)

def compound_loss(pred, target, deformation_field, loss_weights):
    """
    Combines multiple loss functions for surface registration.

    Args:
        pred (Tensor): Warped template (B, C, D, H, W)
        target (Tensor): Fixed map (B, C, D, H, W)
        deformation_field (Tensor): Deformation field (B, 3, D, H, W)
        loss_weights (dict): Weights for each individual loss term

    Returns:
        total_loss (Tensor): Combined weighted loss
        loss_dict (dict): Dictionary of individual loss components
    """
    loss_dict = {
          "chamfer": chamfer_distance_loss(pred, target),
          "hausdorff": hausdorff_distance_loss(pred, target),
          "surface": surface_loss(pred, target),
          "dice": dice_loss(pred, target),  
          "bending_energy": bending_energy_loss(deformation_field),
          "jacobian": jacobian_det_loss(deformation_field),
          "label_overlap": label_overlap_loss(pred),
          "direction": deformation_direction_variation(deformation_field)
      }
    total_loss = sum(loss_weights[k] * loss_dict[k] for k in loss_weights if k in loss_dict)

    return total_loss, loss_dict

