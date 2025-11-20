"""
NeRF Synthetic Dataset Loader
Handles loading images, camera poses, and generating rays for training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import os


class NeRFDataset(Dataset):
    """
    NeRF Synthetic Dataset Loader
    """

    def __init__(
        self,
        data_root: str,
        scene: str,
        split: str = "train",
        img_wh: Tuple[int, int] = (800, 800),
        white_background: bool = True,
        device: str = "cuda"    # <<< UPDATED FOR UBUNTU 
    ):
        """
        Args:
            data_root: Path to nerf_dataset folder
            scene: Scene name (lego, hotdog, ship)
            split: Dataset split (train, val, test)
            img_wh: Image dimensions
            white_background: Composite on white background
            device: Device to load tensors to
        """
        self.data_root = Path(data_root)
        self.scene = scene
        self.split = split
        self.img_w, self.img_h = img_wh
        self.white_background = white_background
        self.device = device

        # Scene path
        self.scene_dir = self.data_root / scene

        # Load transforms & images
        self.load_data()

        # Precompute rays
        self.generate_all_rays()

    def load_data(self):
        """Load camera parameters and image paths"""
        transform_file = self.scene_dir / f"transforms_{self.split}.json"

        with open(transform_file, 'r') as f:
            meta = json.load(f)

        # Intrinsics
        self.camera_angle_x = meta['camera_angle_x']
        self.focal = 0.5 * self.img_w / np.tan(0.5 * self.camera_angle_x)

        # Load images + poses
        self.image_paths = []
        self.c2w_matrices = []

        for frame in meta['frames']:
            img_path = self.scene_dir / f"{frame['file_path']}.png"
            self.image_paths.append(img_path)

            c2w = np.array(frame['transform_matrix'], dtype=np.float32)
            self.c2w_matrices.append(c2w)

        self.c2w_matrices = np.stack(self.c2w_matrices, axis=0)

        print(f"Loaded {len(self.image_paths)} images for {self.split} split")
        print(f"Camera focal length: {self.focal:.2f}")
        print(f"Camera FOV: {np.degrees(self.camera_angle_x):.2f}°")

    def load_image(self, idx: int) -> torch.Tensor:
        """Load and preprocess image"""
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        img = img.resize((self.img_w, self.img_h), Image.LANCZOS)
        img = np.array(img) / 255.0  # Normalize

        if img.shape[2] == 4:
            alpha = img[..., 3:4]
            rgb = img[..., :3]

            if self.white_background:
                rgb = rgb * alpha + (1 - alpha)
            else:
                rgb = rgb * alpha
        else:
            rgb = img[..., :3]

        return torch.from_numpy(rgb).float()

    def get_rays(self, c2w: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for a given camera pose"""
        i, j = np.meshgrid(
            np.arange(self.img_w, dtype=np.float32),
            np.arange(self.img_h, dtype=np.float32),
            indexing='xy'
        )

        dirs = np.stack([
            (i - self.img_w * 0.5) / self.focal,
            -(j - self.img_h * 0.5) / self.focal,
            -np.ones_like(i)
        ], axis=-1)

        rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

        rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)

        return torch.from_numpy(rays_o).float(), torch.from_numpy(rays_d).float()

    def generate_all_rays(self):
        """Pre-generate all rays"""
        print(f"Generating rays for {len(self.image_paths)} images...")

        self.all_rays_o = []
        self.all_rays_d = []
        self.all_rgbs = []

        for idx in range(len(self.image_paths)):
            rgb = self.load_image(idx)
            rays_o, rays_d = self.get_rays(self.c2w_matrices[idx])

            self.all_rays_o.append(rays_o.reshape(-1, 3))
            self.all_rays_d.append(rays_d.reshape(-1, 3))
            self.all_rgbs.append(rgb.reshape(-1, 3))

        self.all_rays_o = torch.cat(self.all_rays_o, dim=0)
        self.all_rays_d = torch.cat(self.all_rays_d, dim=0)
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)

        print(f"Generated {len(self.all_rays_o):,} rays")

    def __len__(self):
        return len(self.all_rays_o)

    def __getitem__(self, idx):
        return {
            'ray_o': self.all_rays_o[idx],
            'ray_d': self.all_rays_d[idx],
            'rgb': self.all_rgbs[idx]
        }

    def get_image_data(self, idx):
        rgb = self.load_image(idx)
        rays_o, rays_d = self.get_rays(self.c2w_matrices[idx])

        return {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'rgb': rgb,
            'c2w': torch.from_numpy(self.c2w_matrices[idx]).float()
        }


def create_dataloaders(
    data_root: str,
    scene: str,
    batch_size: int = 4096,
    img_wh: Tuple[int, int] = (800, 800),
    num_workers: int = 0,
    device: str = "cuda"   # <<< UPDATED
):

    train_dataset = NeRFDataset(
        data_root=data_root,
        scene=scene,
        split="train",
        img_wh=img_wh,
        white_background=True,
        device=device
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_dataset = NeRFDataset(
        data_root=data_root,
        scene=scene,
        split="val",
        img_wh=img_wh,
        white_background=True,
        device=device
    )

    test_dataset = NeRFDataset(
        data_root=data_root,
        scene=scene,
        split="test",
        img_wh=img_wh,
        white_background=True,
        device=device
    )

    return train_loader, val_dataset, test_dataset


def test_dataset():
    """Test dataset loading"""
    print("Testing NeRF Dataset Loader...")

    data_root = "/home/DSE425/Desktop/Robotics_Project_Final/nerf_synthetic"   # <<< UPDATED PATH
    scene = "lego"

    dataset = NeRFDataset(
        data_root=data_root,
        scene=scene,
        split="train",
        img_wh=(400, 400),
        device="cuda"
    )

    print(f"✓ Dataset length: {len(dataset):,} rays")
    print(f"✓ Number of images: {len(dataset.image_paths)}")

    sample = dataset[0]
    print(f"\n✓ Sample ray shapes are correct")

    img_data = dataset.get_image_data(0)
    print(f"✓ Full image loading OK")


if __name__ == "__main__":
    test_dataset()
