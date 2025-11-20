"""
Multiresolution Hash Encoding Implementation
Based on Section 3: "Multiresolution Hash Encoding"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class MultiresolutionHashEncoding(nn.Module):
    """
    Multiresolution hash encoding as described in Section 3.
    
    Key features:
    - Multiple resolution levels (L=16)
    - Spatial hash function for collision handling
    - Trilinear interpolation for continuity
    - Automatic adaptivity through gradient averaging
    """
    
    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 2048,
        device: str = "cuda"     # <<< UPDATED FOR UBUNTU (was CPU/MPS)
    ):
        super().__init__()
        
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.device = device
        
        # Compute growth factor b (Equation 3)
        self.per_level_scale = np.exp(
            (np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1)
        )
        
        # Hash table size
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Prime numbers for spatial hash (Equation 4)
        self.primes = [1, 2654435761, 805459861]
        
        # Embedding tables for all levels
        self.embeddings = nn.ParameterList([
            nn.Parameter(
                torch.empty(self.hashmap_size, n_features_per_level, device=device)
            )
            for _ in range(n_levels)
        ])
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize hash table entries"""
        for embedding in self.embeddings:
            nn.init.uniform_(embedding, -1e-4, 1e-4)
    
    def spatial_hash(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Spatial hash function (Equation 4)
        """
        hash_val = torch.zeros(coords.shape[0], dtype=torch.long, device=self.device)
        
        for i in range(3):
            hash_val ^= (coords[:, i] * self.primes[i])
        
        return hash_val % self.hashmap_size
    
    def get_resolution(self, level: int) -> int:
        """Grid resolution at level (Equation 2)"""
        return int(np.floor(self.base_resolution * (self.per_level_scale ** level)))
    
    def trilinear_interpolation(self, features, weights):
        """
        Trilinear interpolation for voxel corners.
        """
        c000 = features[:, 0]
        c001 = features[:, 1]
        c010 = features[:, 2]
        c011 = features[:, 3]
        c100 = features[:, 4]
        c101 = features[:, 5]
        c110 = features[:, 6]
        c111 = features[:, 7]
        
        wx, wy, wz = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        c00 = c000 * (1 - wx) + c100 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        
        return c0 * (1 - wz) + c1 * wz
    
    def forward(self, positions):
        """
        Encode 3D positions using multiresolution hash encoding.
        """
        batch_size = positions.shape[0]
        encoded_features = []
        
        for level in range(self.n_levels):
            resolution = self.get_resolution(level)
            scaled_pos = positions * resolution
            
            voxel_min = torch.floor(scaled_pos).long()
            voxel_max = voxel_min + 1
            weights = scaled_pos - voxel_min.float()
            
            # Compute 8 voxel corners
            corners = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        corner = torch.stack([
                            voxel_min[:, 0] + i,
                            voxel_min[:, 1] + j,
                            voxel_min[:, 2] + k
                        ], dim=-1)
                        corners.append(corner)
            
            corners = torch.stack(corners, dim=1)
            
            # Direct indexing vs hashing
            max_entries = (resolution + 1) ** 3
            
            if max_entries <= self.hashmap_size:
                indices = (
                    corners[..., 0] * (resolution + 1) ** 2 +
                    corners[..., 1] * (resolution + 1) +
                    corners[..., 2]
                )
            else:
                indices = torch.zeros(batch_size, 8, dtype=torch.long, device=self.device)
                for i in range(8):
                    indices[:, i] = self.spatial_hash(corners[:, i])
            
            indices = torch.clamp(indices, 0, self.hashmap_size - 1)
            
            corner_features = self.embeddings[level][indices]
            interpolated = self.trilinear_interpolation(corner_features, weights)
            
            encoded_features.append(interpolated)
        
        encoded = torch.cat(encoded_features, dim=-1)
        return encoded
    
    @property
    def output_dim(self):
        return self.n_levels * self.n_features_per_level


# Unit Tests
def test_hash_encoding():
    print("Testing Multiresolution Hash Encoding...")
    
    device = "cuda"
    batch_size = 1000
    
    encoding = MultiresolutionHashEncoding(
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=14,
        base_resolution=16,
        finest_resolution=512,
        device=device
    )
    
    positions = torch.rand(batch_size, 3, device=device)
    encoded = encoding(positions)
    
    print(f"✓ Input shape: {positions.shape}")
    print(f"✓ Output shape: {encoded.shape}")
    print(f"✓ Expected dim: {encoding.output_dim}")
    assert encoded.shape == (batch_size, encoding.output_dim)
    
    loss = encoded.sum()
    loss.backward()
    
    print("✓ Gradients OK")
    
    coords = torch.randint(0, 1000, (100, 3), device=device)
    hashes = encoding.spatial_hash(coords)
    
    assert hashes.min() >= 0
    assert hashes.max() < encoding.hashmap_size
    
    print("✓ Hash OK")
    print("All tests passed!")


if __name__ == "__main__":
    test_hash_encoding()
