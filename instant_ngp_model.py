"""
Instant-NGP NeRF Model Implementation
Combines multiresolution hash encoding with compact MLPs
Based on Section 5.4: "Neural Radiance and Density Fields (NeRF)"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SphericalHarmonicsEncoding(nn.Module):
    """
    Spherical harmonics encoding for view directions
    Section 5.4: "up to degree 4" = 16 coefficients
    """

    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.output_dim = degree ** 2

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Encode directions using spherical harmonics

        Args:
            directions: Unit direction vectors [batch_size, 3]

        Returns:
            Encoded directions [batch_size, 16] for degree=4
        """
        x, y, z = directions[..., 0:1], directions[..., 1:2], directions[..., 2:3]

        # Degree 0
        features = [torch.ones_like(x) * 0.28209479177387814]

        if self.degree > 1:
            # Degree 1
            features.extend([
                0.4886025119029199 * y,
                0.4886025119029199 * z,
                0.4886025119029199 * x
            ])

        if self.degree > 2:
            # Degree 2
            features.extend([
                1.0925484305920792 * x * y,
                1.0925484305920792 * y * z,
                0.9461746957575601 * z * z - 0.31539156525251999,
                1.0925484305920792 * x * z,
                0.5462742152960396 * (x * x - y * y)
            ])

        if self.degree > 3:
            # Degree 3
            features.extend([
                0.5900435899266435 * y * (3 * x * x - y * y),
                2.890611442640554 * x * y * z,
                0.4570457994644658 * y * (5 * z * z - 1),
                0.3731763325901154 * z * (5 * z * z - 3),
                0.4570457994644658 * x * (5 * z * z - 1),
                1.445305721320277 * z * (x * x - y * y),
                0.5900435899266435 * x * (x * x - 3 * y * y)
            ])

        return torch.cat(features, dim=-1)


class InstantNGPNetwork(nn.Module):
    """
    Complete Instant-NGP Network for NeRF

    Architecture (from Section 5.4):
    - Hash encoding → Density MLP (1 hidden layer, 64 neurons)
    - Density output + SH encoding → Color MLP (2 hidden layers, 64 neurons)
    """

    def __init__(
        self,
        # Hash encoding params
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 2048,
        # MLP params
        density_n_neurons: int = 64,
        density_n_hidden_layers: int = 1,
        color_n_neurons: int = 64,
        color_n_hidden_layers: int = 2,
        # Other params
        view_encoding_degree: int = 4,
        device: str = "cuda"
    ):
        super().__init__()

        # Normalize device to torch.device for consistent usage
        self.device = torch.device(device if not isinstance(device, torch.device) else device)

        # Import hash encoding (assuming it's in the same project)
        from hash_encoding import MultiresolutionHashEncoding

        # Position encoding
        self.position_encoding = MultiresolutionHashEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            device=self.device  # pass torch.device (supported by torch.empty)
        )

        # View direction encoding (spherical harmonics)
        self.view_encoding = SphericalHarmonicsEncoding(degree=view_encoding_degree)

        # Density MLP
        density_input_dim = self.position_encoding.output_dim

        density_layers = []
        in_dim = density_input_dim

        for i in range(density_n_hidden_layers):
            density_layers.append(nn.Linear(in_dim, density_n_neurons))
            density_layers.append(nn.ReLU(inplace=True))
            in_dim = density_n_neurons

        density_layers.append(nn.Linear(in_dim, 16))

        self.density_mlp = nn.Sequential(*density_layers)

        # Color MLP
        color_input_dim = 16 + self.view_encoding.output_dim

        color_layers = []
        in_dim = color_input_dim

        for i in range(color_n_hidden_layers):
            color_layers.append(nn.Linear(in_dim, color_n_neurons))
            color_layers.append(nn.ReLU(inplace=True))
            in_dim = color_n_neurons

        color_layers.append(nn.Linear(in_dim, 3))

        self.color_mlp = nn.Sequential(*color_layers)

        # Move whole module to device
        self.to(self.device)

    def query_density(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query density and intermediate features at positions

        Returns:
            density: [batch_size, 1]
            features: [batch_size, 16] -- full 16 outputs (first is log-density)
        """
        # Ensure input is on the model device
        if positions.device != self.device:
            positions = positions.to(self.device)

        # Encode positions
        encoded_pos = self.position_encoding(positions)

        # Pass through density MLP
        mlp_output = self.density_mlp(encoded_pos)  # shape [batch, 16]

        # First output is log-space density -> convert to positive density
        density = torch.exp(mlp_output[:, 0:1])

        features = mlp_output  # shape [batch, 16]

        return density, features

    def query_color(
        self,
        features: torch.Tensor,  # now expected shape [batch, 16]
        directions: torch.Tensor
    ) -> torch.Tensor:

        # Ensure inputs on same device
        if features.device != self.device:
            features = features.to(self.device)
        if directions.device != self.device:
            directions = directions.to(self.device)

        # Encode view directions
        encoded_dirs = self.view_encoding(directions)  # shape [batch, 16]

        # Concatenate features (16) and encoded directions (16) -> 32
        color_input = torch.cat([features, encoded_dirs], dim=-1)  # 16 + 16 = 32

        # Pass through color MLP
        rgb = self.color_mlp(color_input)

        # Sigmoid activation for LDR images (Section 5.4)
        rgb = torch.sigmoid(rgb)

        return rgb

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: query density and color

        Args:
            positions: 3D positions [batch_size, 3]
            directions: View directions [batch_size, 3]

        Returns:
            rgb: [batch_size, 3]
            density: [batch_size, 1]
        """
        density, features = self.query_density(positions)
        rgb = self.query_color(features, directions)
        return rgb, density


class VolumetricRenderer(nn.Module):
    """
    Volumetric rendering using ray marching
    Based on Appendix E: "Accelerated NeRF Ray Marching"
    """

    def __init__(
        self,
        near: float = 2.0,
        far: float = 6.0,
        n_samples: int = 64,
        n_importance: int = 64,
        device: str = "cuda"
    ):
        super().__init__()

        self.near = near
        self.far = far
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.device = torch.device(device if not isinstance(device, torch.device) else device)

    def sample_points_along_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        n_samples: int,
        perturb: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays
        Returns:
            points: [batch, n_samples, 3], z_vals: [batch, n_samples]
        """
        batch_size = rays_o.shape[0]
        device = rays_o.device  # use device of input tensors to avoid mismatches

        # Linearly sample in depth
        t_vals = torch.linspace(0.0, 1.0, n_samples, device=device)
        z_vals = near * (1 - t_vals) + far * t_vals
        z_vals = z_vals.expand(batch_size, n_samples)

        # Perturb sampling positions
        if perturb:
            mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
            lower = torch.cat([z_vals[:, :1], mids], dim=-1)
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand

        # Compute 3D points
        points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]

        return points, z_vals

    def volume_rendering(
        self,
        rgb: torch.Tensor,
        density: torch.Tensor,
        z_vals: torch.Tensor,
        white_background: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Volume rendering equation (from original NeRF paper)
        """
        # Compute distances between samples
        device = rgb.device
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([
            dists,
            torch.full_like(dists[:, :1], 1e10, device=device)  # Last distance is large
        ], dim=-1)

        # Compute alpha values
        alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)

        # Compute transmittance
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones_like(alpha[:, :1], device=device),
                1.0 - alpha[:, :-1] + 1e-10
            ], dim=-1),
            dim=-1
        )

        # Compute weights
        weights = alpha * transmittance

        # Render RGB
        rgb_map = torch.sum(weights[:, :, None] * rgb, dim=1)

        # Render depth
        depth_map = torch.sum(weights * z_vals, dim=1)

        # Accumulated alpha
        acc_map = torch.sum(weights, dim=1)

        # Add background
        if white_background:
            rgb_map = rgb_map + (1.0 - acc_map[:, None])

        return rgb_map, depth_map, acc_map

    def render_rays(
        self,
        model: InstantNGPNetwork,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        perturb: bool = True,
        white_background: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Render rays through the scene
        """
        batch_size = rays_o.shape[0]

        # Sample points along rays
        points, z_vals = self.sample_points_along_rays(
            rays_o, rays_d,
            self.near, self.far,
            self.n_samples,
            perturb=perturb
        )

        # Flatten points for network query
        points_flat = points.reshape(-1, 3)

        # Normalize points to [0, 1] (assuming scene is in [-1, 1])
        points_normalized = (points_flat + 1.0) / 2.0
        points_normalized = torch.clamp(points_normalized, 0.0, 1.0)

        # Expand directions for all samples
        dirs_flat = rays_d[:, None, :].expand_as(points).reshape(-1, 3)

        # Query model (model will move inputs to its device internally)
        rgb_flat, density_flat = model(points_normalized, dirs_flat)

        # Reshape outputs
        rgb = rgb_flat.reshape(batch_size, self.n_samples, 3)
        density = density_flat.reshape(batch_size, self.n_samples, 1)

        # Volume rendering
        rgb_map, depth_map, acc_map = self.volume_rendering(
            rgb, density, z_vals, white_background
        )

        return rgb_map, depth_map, acc_map


# Test the model
def test_model():
    """Test the Instant-NGP model"""
    print("Testing Instant-NGP NeRF Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024

    # Create model
    model = InstantNGPNetwork(
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=14,  # Small for testing
        base_resolution=16,
        finest_resolution=512,
        density_n_neurons=64,
        density_n_hidden_layers=1,
        color_n_neurons=64,
        color_n_hidden_layers=2,
        device=device
    )

    print(f"\n✓ Model created on device: {device}")

    # Test forward pass
    positions = torch.rand(batch_size, 3, device=device)
    directions = torch.randn(batch_size, 3, device=device)
    directions = F.normalize(directions, dim=-1)

    rgb, density = model(positions, directions)

    print(f"✓ Forward pass:")
    print(f"  - RGB shape: {rgb.shape}, range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    print(f"  - Density shape: {density.shape}, range: [{density.min():.3f}, {density.max():.3f}]")

    # Test renderer
    renderer = VolumetricRenderer(
        near=2.0,
        far=6.0,
        n_samples=64,
        device=device
    )

    rays_o = torch.zeros(batch_size, 3, device=device)
    rays_d = torch.randn(batch_size, 3, device=device)
    rays_d = F.normalize(rays_d, dim=-1)

    rgb_rendered, depth, acc = renderer.render_rays(
        model, rays_o, rays_d, perturb=False
    )

    print(f"\n✓ Rendering:")
    print(f"  - RGB shape: {rgb_rendered.shape}")
    print(f"  - Depth shape: {depth.shape}")
    print(f"  - Acc shape: {acc.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    hash_params = sum(p.numel() for p in model.position_encoding.parameters())
    mlp_params = total_params - hash_params

    print(f"\n Parameters:")
    print(f"  - Hash encoding: {hash_params:,}")
    print(f"  - MLPs: {mlp_params:,}")
    print(f"  - Total: {total_params:,} ({total_params/1e6:.2f}M)")

    print(f"\n All tests passed!")


if __name__ == "__main__":
    test_model()
