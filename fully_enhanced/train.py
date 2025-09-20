# train.py - Fully Enhanced (Affine + All Regularization)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import RegistrationModel
from get_data import SegDataset
from losses import composite_loss, dice_loss, cross_entropy_loss
from tqdm import tqdm
import numpy as np
import argparse
import os
import wandb

torch.cuda.empty_cache()

# Logging with wandb
wandb.init(project="seg-deformation")


def train(model, dataloader, optimizer, device, epoch, max_epochs=50):
    model.train()
    total_loss = 0
    dice_loss_total = 0
    ce_loss_total = 0
    
    # Set epoch info for adaptive weights in composite_loss
    composite_loss.epoch = epoch
    composite_loss.max_epochs = max_epochs

    for batch_idx, (moving, fixed) in enumerate(
        tqdm(
            dataloader,
            desc=f"Training for epoch: {epoch + 1}/{max_epochs}",
            leave=False,
        )
    ):
        moving = moving.to(device)
        fixed = fixed.to(device)

        warped_template, affine_matrix, deformation_field, affine_warped = model(
            moving, fixed
        )

        # Use composite_loss - it handles ALL losses with proper weights
        loss = composite_loss(warped_template, fixed, deformation_field, max_displacement=10.0)
        
        # Calculate individual components for logging only
        dice_loss_affine = dice_loss(affine_warped, fixed)
        dice_loss_val = dice_loss(warped_template, fixed)
        ce_loss_val = cross_entropy_loss(warped_template, fixed)

        print(f"Affine Dice Loss: {dice_loss_affine:.4f} -> Final Dice: {dice_loss_val:.4f}")
        print(f"Total Loss: {loss:.4f}, CE: {ce_loss_val:.6f}")

        # Debug info for first few batches
        if batch_idx < 3:
            max_def = torch.max(torch.abs(deformation_field)).item()
            mean_def = torch.mean(torch.abs(deformation_field)).item()
            print(f"  Debug - Max deformation: {max_def:.6f}, Mean deformation: {mean_def:.6f}")

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

    return (
        total_loss / len(dataloader),
        dice_loss_total / len(dataloader),
        ce_loss_total / len(dataloader),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_txt",
        type=str,
        default="train_npy.txt",
        help="Path to the training file listing subject paths",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default="neurite-oasis.v1.0/OASIS_OAS1_0016_MR1/seg4_onehot.npy",
        help="Path to the template segmentation map",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default="checkpoints/fully_enhanced_model.pth",
        help="Path to save the trained model",
    )

    args = parser.parse_args()

    # GPU device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    print("Loading dataset")
    train_dataset = SegDataset(args.train_txt, args.template_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print("Dataset loaded.")

    volume_size = train_dataset[0][0].shape[1:]
    model = RegistrationModel(volume_size, num_channels=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        avg_loss, avg_dice, avg_ce = train(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
        )

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, CE: {avg_ce:.4f}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "dice_loss": avg_dice,
                "ce_loss": avg_ce,
            }
        )

        if (epoch + 1) % 5 == 0:
            model_path = f"checkpoints/fully_enhanced_model_epoch_{epoch + 1}.pth"
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "model_config": {"volume_size": volume_size, "num_channels": 5},
            }
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(checkpoint, model_path)
            print(f"Model checkpoint saved to {model_path}")

    # Final save
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "loss": avg_loss,
        "model_config": {"volume_size": volume_size, "num_channels": 5},
    }
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
    torch.save(final_checkpoint, args.save_model_path)
    print(f"Final model checkpoint saved to {args.save_model_path}")


if __name__ == "__main__":
    main()