# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# Enhanced affine parameter predictor with better architecture
class AffineParameterPredictor(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        # Memory-efficient encoder (lighter version)
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            # Second conv block
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            # Final pooling
            nn.AdaptiveAvgPool3d(1),
        )

        # Simpler FC layers
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 12),  # 12 affine parameters
        )

        # Initialize the last layer to predict identity transformation
        self._init_weights()

    def _init_weights(self):
        # Initialize the final layer to output identity transformation
        with torch.no_grad():
            # Identity matrix [1,0,0,0; 0,1,0,0; 0,0,1,0] flattened = [1,0,0,0,0,1,0,0,0,0,1,0]
            identity = torch.tensor(
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32
            )
            self.fc[-1].weight.data.zero_()
            self.fc[-1].bias.data.copy_(identity)

    def forward(self, x):
        x = self.encoder(x)  # (B, 64, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 64)
        x = self.fc(x)  # (B, 12)
        return x.view(-1, 3, 4)  # (B, 3, 4)


# Enhanced affine spatial transformer with better grid handling
class AffineSpatialTransformer(nn.Module):
    def __init__(self, size):
        super().__init__()
        D, H, W = size

        # Create normalized coordinate grid
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, D),
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij",
        )
        grid = torch.stack([x, y, z], dim=-1)
        self.register_buffer("grid", grid.view(1, D, H, W, 3))

    def forward(self, x, affine):
        B, C, D, H, W = x.shape

        # Expand grid for batch
        grid = self.grid.expand(B, -1, -1, -1, -1)
        ones = torch.ones(B, D, H, W, 1, device=x.device)
        grid_h = torch.cat([grid, ones], dim=-1)  # Homogeneous coordinates

        # Apply affine transformation
        grid_affine = torch.matmul(grid_h.view(B, -1, 4), affine.transpose(1, 2))
        grid_affine = grid_affine.view(B, D, H, W, 3)

        # Clamp to valid range to prevent extrapolation issues
        grid_affine = torch.clamp(grid_affine, -1.1, 1.1)

        warped = F.grid_sample(
            x, grid_affine, mode="bilinear", align_corners=False, padding_mode="border"
        )

        return warped


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )

        def upsample_block(in_channels, out_channels):
            return nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(256, 512)

        # Decoder
        self.up4 = upsample_block(512, 256)
        self.dec4 = conv_block(512, 256)
        self.up3 = upsample_block(256, 128)
        self.dec3 = conv_block(256, 128)
        self.up2 = upsample_block(128, 64)
        self.dec2 = conv_block(128, 64)
        self.up1 = upsample_block(64, 32)
        self.dec1 = conv_block(64, 32)

        # Output layer with small displacement initialization
        self.out_conv = nn.Conv3d(32, 3, kernel_size=1)

        # Initialize output to small values
        nn.init.normal_(self.out_conv.weight, 0, 0.01)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        # Decoder
        up4 = self.up4(b)
        d4 = self.dec4(torch.cat((up4, e4), dim=1))
        up3 = self.up3(d4)
        d3 = self.dec3(torch.cat((up3, e3), dim=1))
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat((up2, e2), dim=1))
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat((up1, e1), dim=1))

        deformation_field = self.out_conv(d1)

        # Less restrictive scaling - allow larger initial deformations
        deformation_field = (
            torch.tanh(deformation_field) * 0.3
        )  # Increased from 0.1 to 0.3

        return deformation_field


class SpatialTransformer(nn.Module):
    """
    A true dense STN for 3D one-hot maps, with an identity grid buffer
    and nearest‐neighbor sampling for crisp labels.
    """

    def __init__(self, size, device="cpu"):
        """
        Args:
            size: tuple of ints (D, H, W)
            device: tensor device
        """
        super().__init__()
        D, H, W = size

        lin_z = torch.linspace(-1, 1, D, device=device)
        lin_y = torch.linspace(-1, 1, H, device=device)
        lin_x = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(lin_z, lin_y, lin_x, indexing="ij")

        id_grid = torch.stack((xx, yy, zz), dim=-1)
        self.register_buffer("id_grid", id_grid.unsqueeze(0))

    def forward(self, moving, flow):
        """
        Args:
            moving: (B, C, D, H, W) one‐hot template
            flow:   (B, 3, D, H, W) displacement in normalized coords
        Returns:
            warped: (B, C, D, H, W) one‐hot warped template
        """
        B, C, D, H, W = moving.shape

        flow = flow.permute(0, 2, 3, 4, 1)

        grid = self.id_grid.expand(B, -1, -1, -1, -1)

        warped_grid = grid + flow

        # Clamp grid to prevent sampling issues
        warped_grid = torch.clamp(warped_grid, -1.1, 1.1)

        warped = F.grid_sample(
            moving,
            warped_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return warped


# chaining all components together
class RegistrationModel(nn.Module):
    def __init__(self, size, num_channels=5):
        super().__init__()
        self.affine_predictor = AffineParameterPredictor(in_channels=num_channels * 2)
        self.affine_stn = AffineSpatialTransformer(size)
        self.unet = UNet(in_channels=num_channels * 2, out_channels=3)
        self.nonrigid_stn = SpatialTransformer(size)

    def forward(self, source, target):
        x_affine_input = torch.cat([source, target], dim=1)
        affine_matrix = self.affine_predictor(x_affine_input)
        source_affine = self.affine_stn(source, affine_matrix)

        x_unet_input = torch.cat([source_affine, target], dim=1)
        deformation_field = self.unet(x_unet_input)
        warped_source = self.nonrigid_stn(source_affine, deformation_field)

        return warped_source, affine_matrix, deformation_field, source_affine

    def load_full_enhanced_model(
        checkpoint_path, device, size=(128, 128, 128), num_channels=5
    ):
        """Load full enhanced (affine + non-rigid) model from checkpoint (dict with state dicts and config)"""
        import torch

        try:
            print(f"Loading Full Enhanced Model: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Extract config if present
            config = checkpoint.get("model_config", {})
            size = config.get("size", size)
            num_channels = config.get("num_channels", num_channels)

            # Instantiate model
            model = RegistrationModel(size=size, num_channels=num_channels)

            # Load state dicts
            if "affine_predictor_state_dict" in checkpoint:
                model.affine_predictor.load_state_dict(
                    checkpoint["affine_predictor_state_dict"]
                )
            if "affine_stn_state_dict" in checkpoint:
                model.affine_stn.load_state_dict(checkpoint["affine_stn_state_dict"])
            if "unet_state_dict" in checkpoint:
                model.unet.load_state_dict(checkpoint["unet_state_dict"])
            if "nonrigid_stn_state_dict" in checkpoint:
                model.nonrigid_stn.load_state_dict(
                    checkpoint["nonrigid_stn_state_dict"]
                )
            # Fallback for single dict
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)

            model.to(device)
            model.eval()

            print(
                f"   ✅ Full enhanced model loaded successfully (size={size}, num_channels={num_channels})"
            )
            return model

        except Exception as e:
            print(f"   ❌ Failed to load full enhanced model: {str(e)}")
            return None
