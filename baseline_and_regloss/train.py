# train.py - Baseline + Regularization
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import UNet, SpatialTransformer
from get_data import SegDataset
from losses import composite_loss, dice_loss, cross_entropy_loss
from tqdm import tqdm
import numpy as np
import argparse
import os
import wandb

torch.cuda.empty_cache()

# Logging with wandb
wandb.init(project='seg-deformation')

def train(model, stn, dataloader, optimizer, device, epoch, max_epochs=50):
    model.train()
    total_loss = 0
    dice_loss_total = 0
    ce_loss_total = 0
    
    for batch_idx, (moving, fixed) in enumerate(tqdm(dataloader, desc=f'Training for epoch: {epoch+1}/{max_epochs}', leave=False)):
        moving = moving.to(device)
        fixed = fixed.to(device)
        
        input_ = torch.cat([moving, fixed], dim=1)  # Shape: (B, 10, 128, 128, 128)
        deformation_field = model(input_)
        
        warped_template = stn(moving, deformation_field)

        # Calculate individual loss components for logging
        dice_loss_val = dice_loss(warped_template, fixed)
        ce_loss_val = cross_entropy_loss(warped_template, fixed)
        
        print(f"Dice Loss: {dice_loss_val:.4f}")
        print(f"Cross-entropy: {ce_loss_val:.6f}")
        
        # Debug info for first few batches
        if batch_idx < 3:
            max_def = torch.max(torch.abs(deformation_field)).item()
            mean_def = torch.mean(torch.abs(deformation_field)).item()
            print(f"  Debug - Max deformation: {max_def:.6f}, Mean deformation: {mean_def:.6f}")

        # Use composite_loss which includes all regularization terms
        loss = composite_loss(warped_template, fixed, deformation_field, max_displacement=10.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        
        # Clear cache to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_loss += loss.item()
        dice_loss_total += dice_loss_val.item()
        ce_loss_total += ce_loss_val.item()

    return (total_loss / len(dataloader), 
            dice_loss_total / len(dataloader), 
            ce_loss_total / len(dataloader))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_txt', type=str, default='train_npy.txt', help='Path to the training file listing subject paths')
    parser.add_argument('--template_path', type=str, default='neurite-oasis.v1.0/OASIS_OAS1_0016_MR1/seg4_onehot.npy', help='Path to the template segmentation map')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--save_model_path', type=str, default='checkpoints/original_unet_model.pth', help='Path to save the trained model')
    
    args = parser.parse_args()

    # GPU device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("Loading dataset")
    train_dataset = SegDataset(args.train_txt, args.template_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print("Dataset loaded.")

    # U-Net to predict deformations, STN to warp the deformations on top of the template
    model = UNet(in_channels=10, out_channels=3).to(device)
    stn = SpatialTransformer(size=(128,128,128), device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        avg_loss, avg_dice, avg_ce = train(
            model, stn, train_loader, optimizer, device, epoch, args.epochs
        )
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, CE: {avg_ce:.4f}")
        
        wandb.log({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'dice_loss': avg_dice,
            'ce_loss': avg_ce
        })

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model_path = f'checkpoints/original_unet_model_epoch_{epoch + 1}.pth'
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'stn_state_dict': stn.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
                'model_config': {
                    'in_channels': 10,
                    'out_channels': 3,
                    'size': (128,128,128)
                }
            }
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(checkpoint, model_path)
            print(f"Model checkpoint saved to {model_path}")

    # Final save
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'stn_state_dict': stn.state_dict(),
        'epoch': args.epochs,
        'loss': avg_loss,
        'model_config': {
            'in_channels': 10,
            'out_channels': 3,
            'size': (128,128,128)
        }
    }
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
    torch.save(final_checkpoint, args.save_model_path)
    print(f"Final model checkpoint saved to {args.save_model_path}")

if __name__ == "__main__":
    main()