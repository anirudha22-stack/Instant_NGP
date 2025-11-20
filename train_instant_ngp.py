"""
Training Script for Instant-NGP NeRF
Implements the training procedure from Section 4 and 5.4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import time
import json
from tqdm import tqdm
from typing import Dict, Optional
import matplotlib.pyplot as plt
from PIL import Image

# Import our modules
from instant_ngp_config import InstantNGPConfig, get_lego_config
from hash_encoding import MultiresolutionHashEncoding
from instant_ngp_model import InstantNGPNetwork, VolumetricRenderer
from nerf_dataset import NeRFDataset, create_dataloaders


class InstantNGPTrainer:
    """
    Trainer for Instant-NGP NeRF
    Implements training loop with all optimizations from the paper
    """
    
    def __init__(self, config: InstantNGPConfig):
        self.config = config
        self.device = config.device
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.save_config()
        
        # Initialize model
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
        
        # Initialize renderer
        self.renderer = VolumetricRenderer(
            near=config.ray_marching.near_plane,
            far=config.ray_marching.far_plane,
            n_samples=64,
            device=self.device
        )
        
        # Initialize optimizer (Adam with paper settings)
        self.optimizer = optim.Adam(
            [
                {
                    'params': self.model.position_encoding.parameters(),
                    'lr': config.training.learning_rate,
                    'betas': (config.training.beta1, config.training.beta2),
                    'eps': config.training.epsilon
                },
                {
                    'params': list(self.model.density_mlp.parameters()) + 
                              list(self.model.color_mlp.parameters()),
                    'lr': config.training.learning_rate,
                    'betas': (config.training.beta1, config.training.beta2),
                    'eps': config.training.epsilon,
                    'weight_decay': config.training.weight_decay
                }
            ]
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[20000, 30000, 40000],
            gamma=config.training.learning_rate_decay
        )
        
        # Training state
        self.global_step = 0
        self.train_losses = []
        self.val_psnrs = []
        self.best_psnr = 0.0
        
        print(f"Trainer initialized. Output dir: {self.output_dir}")
    
    def save_config(self):
        config_path = self.output_dir / "config.json"
        config_dict = {
            'scene': self.config.dataset.scene,
            'n_levels': self.config.hash_encoding.n_levels,
            'n_features_per_level': self.config.hash_encoding.n_features_per_level,
            'log2_hashmap_size': self.config.hash_encoding.log2_hashmap_size,
            'learning_rate': self.config.training.learning_rate,
            'batch_size': self.config.training.batch_size,
            'n_training_steps': self.config.training.n_training_steps
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, indent=2, fp=f)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        rays_o = batch['ray_o'].to(self.device)
        rays_d = batch['ray_d'].to(self.device)
        target_rgb = batch['rgb'].to(self.device)
        
        pred_rgb, depth, acc = self.renderer.render_rays(
            self.model, rays_o, rays_d,
            perturb=True,
            white_background=True
        )
        
        loss = F.mse_loss(pred_rgb, target_rgb)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self, val_dataset: NeRFDataset, n_images: int = 5) -> float:
        self.model.eval()

        psnrs = []

        for img_idx in range(min(n_images, len(val_dataset.image_paths))):
            img_data = val_dataset.get_image_data(img_idx)

            H = self.config.dataset.image_height
            W = self.config.dataset.image_width

            rays_o = img_data['rays_o'].reshape(-1, 3).to(self.device)
            rays_d = img_data['rays_d'].reshape(-1, 3).to(self.device)

            target_rgb_flat = img_data['rgb'].reshape(-1, 3).to(torch.float32).cpu()

            chunk_size = 4096
            pred_rgb_chunks = []

            for i in range(0, rays_o.shape[0], chunk_size):
                rays_o_chunk = rays_o[i:i + chunk_size]
                rays_d_chunk = rays_d[i:i + chunk_size]

                pred_rgb_chunk, _, _ = self.renderer.render_rays(
                    self.model, rays_o_chunk, rays_d_chunk,
                    perturb=False,
                    white_background=True
                )

                pred_rgb_chunks.append(pred_rgb_chunk.cpu())

            pred_rgb_flat = torch.cat(pred_rgb_chunks, dim=0)

            if target_rgb_flat.shape != pred_rgb_flat.shape:
                try:
                    target_rgb_flat = target_rgb_flat.reshape(H * W, 3)
                except:
                    raise RuntimeError("Target shape mismatch.")

            mse = F.mse_loss(pred_rgb_flat, target_rgb_flat)
            psnr = -10.0 * torch.log10(mse + 1e-10)
            psnrs.append(psnr.item())

        return float(np.mean(psnrs)) if psnrs else 0.0
    
    def save_image_comparison(self, pred, target, step):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        pred = np.clip(pred, 0, 1)
        target = np.clip(target, 0, 1)
        
        axes[0].imshow(target)
        axes[0].set_title('Target')
        axes[0].axis('off')
        
        axes[1].imshow(pred)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        diff = np.abs(target - pred)
        axes[2].imshow(diff)
        axes[2].set_title('Absolute Difference')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f'comparison_step_{step:06d}.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
        
        pred_img = (pred * 255).astype(np.uint8)
        Image.fromarray(pred_img).save(
            self.output_dir / f'render_step_{step:06d}.png'
        )
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_psnrs': self.val_psnrs,
            'best_psnr': self.best_psnr
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_step_{self.global_step:06d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (PSNR: {self.best_psnr:.2f} dB)")
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        if self.train_losses:
            axes[0].plot(self.train_losses)
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
        
        if self.val_psnrs:
            steps = np.arange(len(self.val_psnrs)) * self.config.training.validation_interval
            axes[1].plot(steps, self.val_psnrs)
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('PSNR (dB)')
            axes[1].set_title('Validation PSNR')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150)
        plt.close()
    
    def train(self, train_loader: DataLoader, val_dataset: NeRFDataset):
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        pbar = tqdm(total=self.config.training.n_training_steps, desc="Training")
        
        epoch = 0
        while self.global_step < self.config.training.n_training_steps:
            epoch += 1
            
            for batch in train_loader:
                loss = self.train_step(batch)
                self.train_losses.append(loss)
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                if self.global_step % self.config.training.validation_interval == 0:
                    psnr = self.validate(val_dataset)
                    self.val_psnrs.append(psnr)
                    
                    is_best = psnr > self.best_psnr
                    if is_best:
                        self.best_psnr = psnr
                    
                    print(f"\nStep {self.global_step}: Val PSNR = {psnr:.2f} dB")
                    
                    self.plot_training_curves()
                    
                    if self.global_step % self.config.training.checkpoint_interval == 0:
                        self.save_checkpoint(is_best=is_best)
                
                self.global_step += 1
                
                if self.global_step >= self.config.training.n_training_steps:
                    break
        
        pbar.close()
        
        print("\n" + "="*80)
        print("FINAL VALIDATION")
        print("="*80)
        
        final_psnr = self.validate(val_dataset, n_images=10)
        print(f"Final PSNR: {final_psnr:.2f} dB")
        
        self.save_checkpoint(is_best=final_psnr > self.best_psnr)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Final PSNR: {final_psnr:.2f} dB")
        print(f"Output directory: {self.output_dir}")


def main():
    """Main function to run training"""
    
    config = get_lego_config()
    
    # UPDATED FOR UBUNTU
    config.dataset.data_root = "/home/DSE425/Desktop/Robotics_Project_Final/nerf_synthetic"
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config.print_config()
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    train_loader, val_dataset, test_dataset = create_dataloaders(
        data_root=config.dataset.data_root,
        scene=config.dataset.scene,
        batch_size=config.training.n_rays_per_batch,
        img_wh=(config.dataset.image_width, config.dataset.image_height),
        device=config.device
    )
    
    trainer = InstantNGPTrainer(config)
    
    trainer.train(train_loader, val_dataset)


if __name__ == "__main__":
    main()
