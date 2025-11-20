#!/usr/bin/env python3
"""
Master Evaluation + Rendering Script for Instant-NGP NeRF
- Ubuntu/CUDA-ready
- Safe numpy->tensor conversions (no read-only warnings)
- Full-dataset evaluation (PSNR), image saving, depth maps
- Novel-view rendering
- 360-degree turntable MP4 rendering
Supports scenes: lego, hotdog, ship
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Import your project modules (assumes these files are in PYTHONPATH / same folder)
from instant_ngp_config import get_lego_config, get_hotdog_config, get_ship_config
from instant_ngp_model import InstantNGPNetwork, VolumetricRenderer
from nerf_dataset import NeRFDataset

# -------------------------
# Helpers
# -------------------------
def parse_device_arg(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == 'cuda':
        return torch.device("cuda")
    elif device_arg == 'cpu':
        return torch.device("cpu")
    else:
        return torch.device(device_arg)


def to_cpu_tensor(x: Union[np.ndarray, torch.Tensor], dtype=torch.float32) -> torch.Tensor:
    """Return a CPU torch tensor (copying numpy to ensure writable backing)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device='cpu')
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy()).to(dtype=dtype, device='cpu')
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


# -------------------------
# Evaluator class
# -------------------------
class InstantNGPEvaluator:
    def __init__(self, checkpoint_path: Union[str, Path], config, device: torch.device):
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        # Build model & renderer (match settings used during training)
        self.model = InstantNGPNetwork(
            n_levels=config.hash_encoding.n_levels,
            n_features_per_level=config.hash_encoding.n_features_per_level,
            log2_hashmap_size=config.hash_encoding.log2_hashmap_size,
            base_resolution=config.hash_encoding.base_resolution,
            finest_resolution=config.hash_encoding.finest_resolution,
            density_n_neurons=config.mlp.density_n_neurons,
            density_n_hidden_layers=config.mlp.density_n_hidden_layers,
            color_n_neurons=config.mlp.color_n_neurons,
            color_n_hidden_layers=config.mlp.color_n_hidden_layers,
            device=self.device
        ).to(self.device)

        # Use more samples for high-quality evaluation
        self.renderer = VolumetricRenderer(
            near=config.ray_marching.near_plane,
            far=config.ray_marching.far_plane,
            n_samples=128,
            device=self.device
        )

        self.load_checkpoint()

    def load_checkpoint(self):
        print(f"Loading checkpoint: {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        step = ckpt.get('global_step', 'unknown')
        print(f"✓ Loaded checkpoint (step={step})")
        if 'best_psnr' in ckpt:
            print(f"✓ Best PSNR in checkpoint: {ckpt['best_psnr']:.2f} dB")

    @torch.no_grad()
    def render_image(self,
                     rays_o: Union[np.ndarray, torch.Tensor],
                     rays_d: Union[np.ndarray, torch.Tensor],
                     chunk_size: int = 4096,
                     show_progress: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render entire image given flattened rays (H*W, 3).
        Returns CPU tensors: rgb [H*W,3], depth [H*W]
        """
        # Convert to torch on device (safe copies for numpy)
        if not isinstance(rays_o, torch.Tensor):
            rays_o_t = torch.from_numpy(rays_o.copy()).to(device=self.device, dtype=torch.float32)
        else:
            rays_o_t = rays_o.to(device=self.device, dtype=torch.float32)

        if not isinstance(rays_d, torch.Tensor):
            rays_d_t = torch.from_numpy(rays_d.copy()).to(device=self.device, dtype=torch.float32)
        else:
            rays_d_t = rays_d.to(device=self.device, dtype=torch.float32)

        n_rays = rays_o_t.shape[0]
        rgb_chunks = []
        depth_chunks = []

        iterator = range(0, n_rays, chunk_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Rendering", unit="chunk")

        for i in iterator:
            o_chunk = rays_o_t[i:i + chunk_size]
            d_chunk = rays_d_t[i:i + chunk_size]

            rgb_chunk, depth_chunk, _ = self.renderer.render_rays(
                self.model, o_chunk, d_chunk, perturb=False, white_background=True
            )

            rgb_chunks.append(rgb_chunk.cpu())
            depth_chunks.append(depth_chunk.cpu())

        rgb = torch.cat(rgb_chunks, dim=0)   # [H*W,3] CPU
        depth = torch.cat(depth_chunks, dim=0)  # [H*W] CPU

        return rgb, depth

    def evaluate_dataset(self, dataset: NeRFDataset, output_dir: Path, save_images: bool = True) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        n_images = len(dataset.image_paths)
        psnrs = []

        print(f"Evaluating {n_images} images (split={dataset.split}) ...")
        for idx in tqdm(range(n_images), desc="Images"):
            data = dataset.get_image_data(idx)
            H = self.config.dataset.image_height
            W = self.config.dataset.image_width

            rays_o = data['rays_o'].reshape(-1, 3)
            rays_d = data['rays_d'].reshape(-1, 3)
            target_rgb_raw = data['rgb'].reshape(-1, 3)

            pred_rgb_flat, pred_depth_flat = self.render_image(rays_o, rays_d, chunk_size=4096, show_progress=False)

            pred_rgb = pred_rgb_flat.reshape(H, W, 3)       # torch CPU
            pred_depth = pred_depth_flat.reshape(H, W)      # torch CPU

            # Convert target to CPU torch tensor safely
            target_rgb = to_cpu_tensor(target_rgb_raw, dtype=torch.float32).reshape(H, W, 3)

            # Metric (MSE->PSNR)
            mse = F.mse_loss(pred_rgb, target_rgb)
            psnr = -10.0 * torch.log10(mse + 1e-10)
            psnrs.append(psnr.item())

            if save_images:
                pred_np = (pred_rgb.numpy() * 255.0).astype(np.uint8)
                target_np = (target_rgb.numpy() * 255.0).astype(np.uint8)
                depth_np = pred_depth.numpy()
                depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                depth_img = (depth_norm * 255.0).astype(np.uint8)

                Image.fromarray(pred_np).save(output_dir / f'{dataset.split}_{idx:03d}_pred.png')
                Image.fromarray(target_np).save(output_dir / f'{dataset.split}_{idx:03d}_target.png')
                Image.fromarray(depth_img).save(output_dir / f'{dataset.split}_{idx:03d}_depth.png')

                if idx < 5:
                    self.save_comparison(pred_np / 255.0, target_np / 255.0, depth_norm, output_dir / f'{dataset.split}_{idx:03d}_comparison.png', psnr.item())

        results = {
            'mean_psnr': float(np.mean(psnrs)) if psnrs else 0.0,
            'std_psnr': float(np.std(psnrs)) if psnrs else 0.0,
            'min_psnr': float(np.min(psnrs)) if psnrs else 0.0,
            'max_psnr': float(np.max(psnrs)) if psnrs else 0.0,
            'per_image_psnr': psnrs
        }

        with open(output_dir / f'{dataset.split}_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def save_comparison(self, pred: np.ndarray, target: np.ndarray, depth: np.ndarray, save_path: Path, psnr: float):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)

        axes[0, 0].imshow(target)
        axes[0, 0].set_title('Target (GT)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(pred)
        axes[0, 1].set_title(f'Prediction (PSNR: {psnr:.2f} dB)')
        axes[0, 1].axis('off')

        diff = np.abs(target - pred)
        im = axes[1, 0].imshow(diff, cmap='hot')
        axes[1, 0].set_title('Absolute Difference')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])

        im = axes[1, 1].imshow(depth, cmap='viridis')
        axes[1, 1].set_title('Depth Map')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def render_novel_view(self, c2w: np.ndarray, chunk_size: int = 4096) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a single novel view defined by a 4x4 camera-to-world matrix.
        Returns (rgb_cpu_tensor [H,W,3], depth_cpu_tensor [H,W])
        """
        focal = 0.5 * self.config.dataset.image_width / np.tan(0.5 * self.config.dataset.camera_angle_x)
        W = self.config.dataset.image_width
        H = self.config.dataset.image_height

        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], axis=-1)

        rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)

        # safe copy -> torch
        rays_o_t = torch.from_numpy(rays_o.copy()).float().reshape(-1, 3)
        rays_d_t = torch.from_numpy(rays_d.copy()).float().reshape(-1, 3)

        rgb_flat, depth_flat = self.render_image(rays_o_t, rays_d_t, chunk_size=chunk_size, show_progress=True)

        rgb = rgb_flat.reshape(H, W, 3)
        depth = depth_flat.reshape(H, W)

        return rgb, depth

    def render_turntable_video(self, output_path: Path, n_frames: int = 120, radius: float = 4.0):
        """
        Render a 360-degree turntable MP4 and save to output_path.
        """
        W = self.config.dataset.image_width
        H = self.config.dataset.image_height

        cam_angle = self.config.dataset.camera_angle_x
        focal = 0.5 * W / np.tan(0.5 * cam_angle)

        frames = []
        print("Rendering turntable video...")

        for t in tqdm(range(n_frames), desc="Video frames"):
            theta = 2 * math.pi * (t / n_frames)
            cam_x = radius * math.cos(theta)
            cam_y = radius * math.sin(theta)
            cam_z = 0.5

            # Build c2w: look at origin
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 3] = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

            forward = -c2w[:3, 3]
            forward = forward / np.linalg.norm(forward)
            tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(tmp, forward)
            right = right / (np.linalg.norm(right) + 1e-9)
            up = np.cross(forward, right)
            c2w[:3, :3] = np.stack([right, up, forward], axis=1)

            rgb, _ = self.render_novel_view(c2w, chunk_size=4096)
            frame = (rgb.numpy() * 255.0).astype(np.uint8)
            frames.append(frame)

        # Save MP4
        imageio.mimwrite(str(output_path), frames, fps=30, quality=8)
        print(f"Saved video to: {output_path}")

# -------------------------
# CLI Main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Master evaluation & rendering for Instant-NGP")
    p.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (pt)')
    p.add_argument('--scene', type=str, default='lego', choices=['lego', 'hotdog', 'ship'])
    p.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    p.add_argument('--dataset_root', type=str, default='/home/DSE425/Desktop/Robotics_Project_Final/nerf_synthetic', help='Dataset root path')
    p.add_argument('--output_dir', type=str, default='./evaluation', help='Output directory')
    p.add_argument('--device', type=str, default='auto', help='Device: auto/cuda/cpu')
    p.add_argument('--render_video', action='store_true', help='Render 360-degree turntable video after evaluation')
    p.add_argument('--video_frames', type=int, default=120, help='Frames for output video')
    p.add_argument('--video_radius', type=float, default=4.0, help='Camera radius for turntable')

    args = p.parse_args()

    device = parse_device_arg(args.device)
    print(f"Using device: {device}")

    # Load scene config
    if args.scene == 'lego':
        config = get_lego_config()
    elif args.scene == 'hotdog':
        config = get_hotdog_config()
    else:
        config = get_ship_config()

    # Ensure dataset path
    config.dataset.data_root = args.dataset_root
    config.device = device

    # Build evaluator
    evaluator = InstantNGPEvaluator(
        checkpoint_path=args.checkpoint,
        config=config,
        device=device
    )

    # Load dataset
    print(f"Loading dataset (scene={args.scene}, split={args.split}) from {args.dataset_root}")
    dataset = NeRFDataset(
        data_root=config.dataset.data_root,
        scene=args.scene,
        split=args.split,
        img_wh=(config.dataset.image_width, config.dataset.image_height),
        white_background=True,
        device=device
    )

    out_dir = Path(args.output_dir) / args.scene / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate & save images/metrics
    results = evaluator.evaluate_dataset(dataset, out_dir, save_images=True)

    print("\n" + "="*80)
    print(f"EVALUATION RESULTS ({args.scene} / {args.split})")
    print("="*80)
    print(f"Mean PSNR: {results['mean_psnr']:.2f} ± {results['std_psnr']:.2f} dB")
    print(f"Min PSNR:  {results['min_psnr']:.2f} dB")
    print(f"Max PSNR:  {results['max_psnr']:.2f} dB")
    print(f"Saved outputs to: {out_dir}")
    print("="*80)

    # Optionally render video
    if args.render_video:
        video_path = Path(args.output_dir) / args.scene / f"turntable_{args.video_frames}f.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator.render_turntable_video(video_path, n_frames=args.video_frames, radius=args.video_radius)


if __name__ == "__main__":
    main()
