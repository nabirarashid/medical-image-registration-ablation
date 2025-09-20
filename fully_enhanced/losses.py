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


def smoothing_loss(deformation_field):
    """L2 gradient regularization - penalize non-smooth deformations"""
    dx = deformation_field[:, :, 1:, :, :] - deformation_field[:, :, :-1, :, :]
    dy = deformation_field[:, :, :, 1:, :] - deformation_field[:, :, :, :-1, :]
    dz = deformation_field[:, :, :, :, 1:] - deformation_field[:, :, :, :, :-1]

    return torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)


def displacement_magnitude_loss(deformation_field, max_displacement=10.0):
    """
    Penalize large displacements - increased threshold for medical images

    Args:
        deformation_field: (B, 3, D, H, W) displacement field
        max_displacement: maximum allowed displacement in voxels (not normalized)
    """
    # Calculate magnitude of displacement vectors
    displacement_magnitude = torch.sqrt(torch.sum(deformation_field**2, dim=1))
    # Only penalize displacements above threshold
    excess = F.relu(displacement_magnitude - max_displacement)
    return torch.mean(excess**2)


def local_rigidity_loss(deformation_field, window_size=3):
    """
    Encourage local rigidity by penalizing large variations in displacement within neighborhoods.
    """
    # Use 3D convolution for local rigidity (variance in window)
    B, C, D, H, W = deformation_field.shape
    kernel = torch.ones(
        (1, 1, window_size, window_size, window_size), device=deformation_field.device
    ) / (window_size**3)
    rigidity_loss = 0
    for c in range(C):
        field_c = deformation_field[:, c : c + 1, :, :, :]
        local_mean = F.conv3d(field_c, kernel, padding=window_size // 2)
        rigidity_loss += torch.mean((field_c - local_mean) ** 2)
    return rigidity_loss / C


def bending_energy_loss(flow):
    """Second-order derivatives for smoothness - penalize high curvature"""
    d2x = flow[:, :, 2:, :, :] - 2 * flow[:, :, 1:-1, :, :] + flow[:, :, :-2, :, :]
    d2y = flow[:, :, :, 2:, :] - 2 * flow[:, :, :, 1:-1, :] + flow[:, :, :, :-2, :]
    d2z = flow[:, :, :, :, 2:] - 2 * flow[:, :, :, :, 1:-1] + flow[:, :, :, :, :-2]

    return torch.mean(d2x**2) + torch.mean(d2y**2) + torch.mean(d2z**2)


def jacobian_det_loss(flow, eps=1e-6):
    """
    Accurate Jacobian determinant loss for 3D deformation fields.
    Penalizes negative determinants (folding) and extreme volume changes.
    Args:
        flow: displacement field (B, 3, D, H, W)
        eps: small value to avoid numerical issues
    """
    B, C, D, H, W = flow.shape
    # Compute gradients for each component
    dx = (flow[:, :, 2:, 1:-1, 1:-1] - flow[:, :, :-2, 1:-1, 1:-1]) / 2.0
    dy = (flow[:, :, 1:-1, 2:, 1:-1] - flow[:, :, 1:-1, :-2, 1:-1]) / 2.0
    dz = (flow[:, :, 1:-1, 1:-1, 2:] - flow[:, :, 1:-1, 1:-1, :-2]) / 2.0

    # The gradients are (B, 3, D-2, H-2, W-2)
    # Build Jacobian matrix J = I + grad(u)
    # For each voxel, J is 3x3:
    # J = [[du_x/dx+1, du_x/dy, du_x/dz],
    #      [du_y/dx, du_y/dy+1, du_y/dz],
    #      [du_z/dx, du_z/dy, du_z/dz+1]]

    # du_x/dx, du_x/dy, du_x/dz
    du_x_dx = dx[:, 0]
    du_x_dy = dy[:, 0]
    du_x_dz = dz[:, 0]
    # du_y/dx, du_y/dy, du_y/dz
    du_y_dx = dx[:, 1]
    du_y_dy = dy[:, 1]
    du_y_dz = dz[:, 1]
    # du_z/dx, du_z/dy, du_z/dz
    du_z_dx = dx[:, 2]
    du_z_dy = dy[:, 2]
    du_z_dz = dz[:, 2]

    # Construct Jacobian matrix
    # Shape: (B, D-2, H-2, W-2, 3, 3)
    J = torch.zeros((B, D - 2, H - 2, W - 2, 3, 3), device=flow.device)
    J[..., 0, 0] = du_x_dx + 1
    J[..., 0, 1] = du_x_dy
    J[..., 0, 2] = du_x_dz
    J[..., 1, 0] = du_y_dx
    J[..., 1, 1] = du_y_dy + 1
    J[..., 1, 2] = du_y_dz
    J[..., 2, 0] = du_z_dx
    J[..., 2, 1] = du_z_dy
    J[..., 2, 2] = du_z_dz + 1

    # Compute determinant for each voxel
    detJ = (
        J[..., 0, 0] * (J[..., 1, 1] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 1])
        - J[..., 0, 1] * (J[..., 1, 0] * J[..., 2, 2] - J[..., 1, 2] * J[..., 2, 0])
        + J[..., 0, 2] * (J[..., 1, 0] * J[..., 2, 1] - J[..., 1, 1] * J[..., 2, 0])
    )
    # Clamp detJ to avoid extreme values
    detJ = torch.clamp(detJ, min=eps, max=10.0)

    # Log-barrier for better numerical stability
    log_det_penalty = -torch.mean(torch.log(detJ + eps))
    # Robust penalty for negative determinants
    det_penalty = torch.mean(torch.max(torch.zeros_like(detJ), eps - detJ) ** 2)
    # Penalize extreme volume changes
    volume_penalty = torch.mean((detJ - 1.0) ** 2) * 0.1
    return det_penalty + log_det_penalty + volume_penalty


def get_adaptive_weights(epoch, max_epochs):
    """Adaptive regularization weight scheduling"""
    reg_weight = max(0.01, 0.1 * (1 - epoch / max_epochs))
    return reg_weight


def cross_entropy_loss(pred, target):
    return F.cross_entropy(pred, target.argmax(dim=1))


def composite_loss(pred, target, flow, max_displacement=10.0):
    """Enhanced composite loss with better weighting"""
    dice_loss_val = dice_loss(pred, target)
    ce_loss_val = cross_entropy_loss(pred, target)

    # Regularization losses
    smoothing_loss_val = smoothing_loss(flow)
    bending_loss_val = bending_energy_loss(flow)
    jacobian_loss_val = jacobian_det_loss(flow)
    displacement_loss_val = displacement_magnitude_loss(flow, max_displacement)
    rigidity_loss_val = local_rigidity_loss(flow)

    # Adaptive regularization weight
    epoch = getattr(composite_loss, "epoch", 0)
    max_epochs = getattr(composite_loss, "max_epochs", 100)
    reg_weight = get_adaptive_weights(epoch, max_epochs)

    return (
        0.7 * dice_loss_val
        + 0.3 * ce_loss_val
        + reg_weight
        * (
            0.01 * smoothing_loss_val
            + 0.005 * bending_loss_val
            + 0.1 * jacobian_loss_val
            + 0.01 * displacement_loss_val
            + 0.005 * rigidity_loss_val
        )
    )
