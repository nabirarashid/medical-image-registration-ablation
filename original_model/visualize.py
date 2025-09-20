# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import io
import os
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def visualize_registration_colab(moving, fixed, warped, 
                                        subject_ids=None, slice_idx=None, max_samples=2,
                                        epoch=None, show_metrics=True):
    """
    Enhanced Colab-friendly visualization for medical image registration with clearer labeling
    and optional metrics display. Adapted for baseline model without affine transformation.
    
    Args:
        moving: Moving/template images tensor [B, C, D, H, W]
        fixed: Fixed/target images tensor [B, C, D, H, W] 
        warped: Images after non-rigid transformation
        subject_ids: List of subject identifiers (e.g., ['Patient_410', 'Patient_411'])
        slice_idx: Which slice to visualize (default: middle slice)
        max_samples: Maximum number of samples to visualize
        epoch: Training epoch number for display
        show_metrics: Whether to compute and display basic metrics
    """
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    import io
    
    # Convert to numpy arrays (argmax for segmentation masks)
    moving_idx = moving.argmax(dim=1).cpu().numpy()
    fixed_idx = fixed.argmax(dim=1).cpu().numpy()
    warped_idx = warped.argmax(dim=1).cpu().numpy()

    B, C, D, H, W = moving.shape
    if slice_idx is None:
        slice_idx = D // 2

    for i in range(min(B, max_samples)):
        # Determine subject identifier
        if subject_ids and i < len(subject_ids):
            subject_id = subject_ids[i]
        else:
            # Use sample numbers that match your training (410, 411, etc.)
            subject_id = f"Sample_{410+i:03d}"
            
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Original images and result
        axs[0,0].imshow(moving_idx[i, slice_idx], cmap='tab20', interpolation='nearest')
        axs[0,0].set_title(f'Moving Template\n{subject_id}', fontsize=14, fontweight='bold')
        axs[0,0].add_patch(Rectangle((2, 2), 20, 5, linewidth=2, edgecolor='blue', facecolor='none'))
        axs[0,0].text(25, 7, 'Template', fontsize=10, color='blue', fontweight='bold')
        
        axs[0,1].imshow(fixed_idx[i, slice_idx], cmap='tab20', interpolation='nearest')
        axs[0,1].set_title(f'Fixed Target\n{subject_id}', fontsize=14, fontweight='bold')
        axs[0,1].add_patch(Rectangle((2, 2), 20, 5, linewidth=2, edgecolor='red', facecolor='none'))
        axs[0,1].text(25, 7, 'Target', fontsize=10, color='red', fontweight='bold')
        
        axs[0,2].imshow(warped_idx[i, slice_idx], cmap='tab20', interpolation='nearest')
        axs[0,2].set_title(f'After Non-Rigid Registration\n{subject_id}', fontsize=14, fontweight='bold')
        axs[0,2].add_patch(Rectangle((2, 2), 20, 5, linewidth=2, edgecolor='purple', facecolor='none'))
        axs[0,2].text(25, 7, 'Final', fontsize=10, color='purple', fontweight='bold')
        
        # Row 2: Difference maps and overlays
        # Compute difference maps
        moving_slice = moving_idx[i, slice_idx]
        fixed_slice = fixed_idx[i, slice_idx]
        warped_slice = warped_idx[i, slice_idx]
        
        # Difference maps (absolute difference)
        diff_initial = np.abs(moving_slice.astype(float) - fixed_slice.astype(float))
        diff_final = np.abs(warped_slice.astype(float) - fixed_slice.astype(float))
        
        axs[1,0].imshow(diff_initial, cmap='hot', interpolation='nearest')
        axs[1,0].set_title('Initial Difference\n(Template vs Target)', fontsize=12)
        
        axs[1,1].imshow(create_overlay(fixed_slice, moving_slice), interpolation='nearest')
        axs[1,1].set_title('Initial Overlay\n(Red=Target, Blue=Template)', fontsize=12)
        
        axs[1,2].imshow(diff_final, cmap='hot', interpolation='nearest')
        axs[1,2].set_title('Final Difference\n(Registered vs Target)', fontsize=12)
        
        # Remove axes for cleaner look
        for ax_row in axs:
            for ax in ax_row:
                ax.axis('off')
        
        # Add metrics if requested
        if show_metrics:
            # Simple Dice coefficient calculation
            dice_initial = calculate_dice(moving_slice, fixed_slice)
            dice_final = calculate_dice(warped_slice, fixed_slice)
            
            # Add text box with metrics
            metrics_text = f"""Registration Metrics - {subject_id}
Initial Dice: {dice_initial:.3f}
Final Dice: {dice_final:.3f}
Improvement: {dice_final - dice_initial:.3f}"""
            
            fig.text(0.02, 0.02, metrics_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Add epoch information if provided
        if epoch is not None:
            fig.suptitle(f'Registration Results - Epoch {epoch}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Registration Results', fontsize=16, fontweight='bold')
            
        # Add legend
        legend_elements = [
            mpatches.Patch(color='blue', label='Template (Moving)'),
            mpatches.Patch(color='red', label='Target (Fixed)'),
            mpatches.Patch(color='purple', label='Final Result')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)
        
        # Save figure to bytes buffer for Colab display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close(fig)
        
        # Display image from buffer in Colab
        display(Image(buf.read()))

def create_overlay(fixed, moving, alpha=0.7):
    """Create red-blue overlay for visual comparison"""
    overlay = np.zeros((fixed.shape[0], fixed.shape[1], 3))
    
    # Normalize to 0-1
    fixed_norm = (fixed - fixed.min()) / (fixed.max() - fixed.min() + 1e-8)
    moving_norm = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
    
    # Red channel for fixed, blue for moving
    overlay[:, :, 0] = fixed_norm * alpha  # Red
    overlay[:, :, 2] = moving_norm * alpha  # Blue
    
    return overlay

def calculate_dice(pred, target):
    """Calculate Dice coefficient between two segmentation masks"""
    pred_bool = pred > 0
    target_bool = target > 0
    
    intersection = np.logical_and(pred_bool, target_bool).sum()
    union = pred_bool.sum() + target_bool.sum()
    
    if union == 0:
        return 1.0
    return 2.0 * intersection / union

def visualize_training_progress(dice_scores, smooth_losses, epochs, save_path=None):
    """
    Visualize training progress over epochs for baseline model
    
    Args:
        dice_scores: List of dice scores per epoch
        smooth_losses: List of smoothing losses per epoch
        epochs: List of epoch numbers
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Dice scores
    ax1.plot(epochs, dice_scores, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Dice Score', fontsize=12)
    ax1.set_title('Registration Accuracy Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot smoothing loss
    ax2.plot(epochs, smooth_losses, 'g-o', label='Smoothing Loss', linewidth=2, markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.set_title('Smoothing Loss Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/training_progress.png", dpi=300, bbox_inches='tight')
        print(f"Saved training progress to {save_path}/training_progress.png")
        plt.close()
    else:
        plt.show()

def visualize_deformation_field(deformation_field, slice_idx=None, save_path=None, 
                               subject_id="Subject_001", step_size=8):
    """
    Visualize the deformation field as a vector field
    
    Args:
        deformation_field: Deformation field tensor [B, 3, D, H, W]
        slice_idx: Which slice to visualize
        save_path: Path to save the visualization
        subject_id: Subject identifier
        step_size: Step size for vector field visualization
    """
    # Convert to numpy
    if hasattr(deformation_field, 'cpu'):
        deformation_field = deformation_field.cpu().numpy()
    
    B, C, D, H, W = deformation_field.shape
    if slice_idx is None:
        slice_idx = D // 2
    
    # Take first sample and specified slice
    deform_slice = deformation_field[0, :, slice_idx, :, :]  # Shape: [3, H, W]
    
    # Create meshgrid for vector field
    y, x = np.mgrid[0:H:step_size, 0:W:step_size]
    
    # Sample deformation field at grid points
    dy = deform_slice[0, ::step_size, ::step_size] * 10  # Scale for visibility
    dx = deform_slice[1, ::step_size, ::step_size] * 10
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot vector field
    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=0.1, 
              color='red', alpha=0.7, width=0.003)
    
    ax.set_title(f'Deformation Field Visualization\n{subject_id} - Slice {slice_idx}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Match image coordinate system
    
    # Add colorbar for magnitude
    magnitude = np.sqrt(dx**2 + dy**2)
    im = ax.imshow(np.sqrt(deform_slice[0]**2 + deform_slice[1]**2), 
                   cmap='viridis', alpha=0.3, extent=[0, W, H, 0])
    plt.colorbar(im, ax=ax, label='Deformation Magnitude')
    
    plt.tight_layout()
    
    if save_path:
        filename = f"{save_path}/deformation_field_{subject_id}_slice_{slice_idx}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved deformation field to {filename}")
        plt.close()
    else:
        plt.show()

def visualize_simple_comparison(moving, fixed, warped, slice_idx=None, save_path=None):
    """
    Simple side-by-side comparison for quick visualization
    
    Args:
        moving: Moving/template images tensor [B, C, D, H, W]
        fixed: Fixed/target images tensor [B, C, D, H, W] 
        warped: Images after non-rigid transformation
        slice_idx: Which slice to visualize (default: middle slice)
        save_path: Path to save the visualization
    """
    # Convert to numpy arrays (argmax for segmentation masks)
    moving_idx = moving.argmax(dim=1).cpu().numpy()
    fixed_idx = fixed.argmax(dim=1).cpu().numpy()
    warped_idx = warped.argmax(dim=1).cpu().numpy()

    B, C, D, H, W = moving.shape
    if slice_idx is None:
        slice_idx = D // 2

    # Take first sample
    moving_slice = moving_idx[0, slice_idx]
    fixed_slice = fixed_idx[0, slice_idx]
    warped_slice = warped_idx[0, slice_idx]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(moving_slice, cmap='tab20', interpolation='nearest')
    axs[0].set_title('Moving Template', fontsize=14)
    axs[0].axis('off')
    
    axs[1].imshow(fixed_slice, cmap='tab20', interpolation='nearest')
    axs[1].set_title('Fixed Target', fontsize=14)
    axs[1].axis('off')
    
    axs[2].imshow(warped_slice, cmap='tab20', interpolation='nearest')
    axs[2].set_title('Registered Result', fontsize=14)
    axs[2].axis('off')
    
    # Calculate and display dice
    dice_initial = calculate_dice(moving_slice, fixed_slice)
    dice_final = calculate_dice(warped_slice, fixed_slice)
    
    fig.suptitle(f'Registration Comparison - Dice Improvement: {dice_initial:.3f} → {dice_final:.3f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/simple_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {save_path}/simple_comparison.png")
        plt.close()
    else:
        plt.show()

# Usage example for Colab:
"""
# Enhanced visualization with your sample IDs
subject_ids = ['Sample_410', 'Sample_411', 'Sample_412']
visualize_registration_colab(
    moving, fixed, warped,
    subject_ids=subject_ids,
    slice_idx=64,  # or whatever slice you prefer
    max_samples=2,
    epoch=3,
    show_metrics=True
)

# Simple comparison
visualize_simple_comparison(moving, fixed, warped, slice_idx=64)

# Training progress
visualize_training_progress(dice_scores, smooth_losses, epochs)

# Deformation field
visualize_deformation_field(deformation_field, slice_idx=64)
"""
