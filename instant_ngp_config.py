"""
Instant-NGP Configuration (Updated for Ubuntu CUDA)
Based on "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


# ------------------------------------------------------------------------------------
# Hash Encoding Configuration
# ------------------------------------------------------------------------------------

@dataclass
class HashEncodingConfig:
    """Multiresolution hash encoding parameters (Table 1 in paper)"""

    n_levels: int = 16                          # L
    n_features_per_level: int = 2               # F
    log2_hashmap_size: int = 19                 # T = 2^19
    base_resolution: int = 16                   # N_min
    finest_resolution: int = 2048               # N_max (scene dependent)

    @property
    def per_level_scale(self) -> float:
        """Compute b from eq.(3)"""
        return np.exp(
            (np.log(self.finest_resolution) - np.log(self.base_resolution))
            / (self.n_levels - 1)
        )

    @property
    def total_hash_size(self) -> int:
        return 2 ** self.log2_hashmap_size

    def get_resolution_at_level(self, level: int) -> int:
        return int(np.floor(self.base_resolution * (self.per_level_scale ** level)))


# ------------------------------------------------------------------------------------
# MLP Configuration
# ------------------------------------------------------------------------------------

@dataclass
class MLPConfig:
    """MLP network architecture (Section 5.4)"""

    density_n_neurons: int = 64
    density_n_hidden_layers: int = 1

    color_n_neurons: int = 64
    color_n_hidden_layers: int = 2

    activation: str = "relu"
    density_activation: str = "exponential"
    color_activation: str = "sigmoid"


# ------------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    learning_rate: float = 1e-2
    learning_rate_decay: float = 0.33
    decay_step: int = 20000
    decay_interval: int = 10000

    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1e-15

    weight_decay: float = 1e-6

    batch_size: int = 2**18
    n_rays_per_batch: int = 8192

    n_training_steps: int = 10000
    loss_type: str = "l2"

    checkpoint_interval: int = 2000
    validation_interval: int = 500


# ------------------------------------------------------------------------------------
# Ray Marching Configuration
# ------------------------------------------------------------------------------------

@dataclass
class RayMarchingConfig:
    """Ray marching parameters (Appendix E)"""

    step_size: float = np.sqrt(3) / 1024
    near_plane: float = 2.0
    far_plane: float = 6.0

    occupancy_grid_resolution: int = 128
    occupancy_grid_levels: int = 1
    occupancy_update_interval: int = 16
    occupancy_decay: float = 0.95
    occupancy_threshold: float = 0.01

    transmittance_threshold: float = 1e-4
    max_samples_per_ray: int = 1024


# ------------------------------------------------------------------------------------
# Dataset Configuration
# ------------------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Dataset configuration for NeRF synthetic scenes"""

    # *** UPDATED FOR UBUNTU ***
    data_root: str = "/home/DSE425/Desktop/Robotics_Project_Final/nerf_synthetic"

    scene: str = "lego"                    # Options: lego, hotdog, ship
    image_width: int = 800
    image_height: int = 800
    camera_angle_x: float = 0.6911         # From transforms

    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    use_alpha: bool = True

    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


# ------------------------------------------------------------------------------------
# Full Instant-NGP Configuration
# ------------------------------------------------------------------------------------

@dataclass
class InstantNGPConfig:

    hash_encoding: HashEncodingConfig = field(default_factory=HashEncodingConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ray_marching: RayMarchingConfig = field(default_factory=RayMarchingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # *** UPDATED FOR UBUNTU CUDA ***
    device: str = "cuda"

    seed: int = 42
    output_dir: str = "./outputs"
    experiment_name: str = "instant_ngp_lego"

    def __post_init__(self):
        # Adjust finest resolution (original paper uses 2048 × scene scale)
        self.hash_encoding.finest_resolution = 2048

        # Auto-update experiment name
        self.experiment_name = f"instant_ngp_{self.dataset.scene}"

    def print_config(self):
        """Pretty print configuration"""
        print("=" * 80)
        print("INSTANT-NGP CONFIGURATION")
        print("=" * 80)

        print("\n Hash Encoding:")
        print(f"  Levels: {self.hash_encoding.n_levels}")
        print(f"  Features/level: {self.hash_encoding.n_features_per_level}")
        print(f"  Hash table size: 2^{self.hash_encoding.log2_hashmap_size} = "
              f"{self.hash_encoding.total_hash_size:,}")
        print(f"  Base resolution: {self.hash_encoding.base_resolution}")
        print(f"  Finest resolution: {self.hash_encoding.finest_resolution}")
        print(f"  Per-level scale: {self.hash_encoding.per_level_scale:.4f}")

        print("\n MLP:")
        print(f"  Density MLP: {self.mlp.density_n_hidden_layers}×{self.mlp.density_n_neurons}")
        print(f"  Color MLP: {self.mlp.color_n_hidden_layers}×{self.mlp.color_n_neurons}")
        print(f"  Activation: {self.mlp.activation}")

        print("\n Training:")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Batch size: {self.training.batch_size:,}")
        print(f"  Rays per batch: {self.training.n_rays_per_batch:,}")
        print(f"  Steps: {self.training.n_training_steps}")
        print(f"  Loss: {self.training.loss_type}")

        print("\n Ray Marching:")
        print(f"  Step size: {self.ray_marching.step_size:.6f}")
        print(f"  Near/Far: {self.ray_marching.near_plane} / {self.ray_marching.far_plane}")
        print(f"  Occupancy grid: {self.ray_marching.occupancy_grid_resolution}³")

        print("\n Dataset:")
        print(f"  Scene: {self.dataset.scene}")
        print(f"  Resolution: {self.dataset.image_width}×{self.dataset.image_height}")
        print(f"  Data root: {self.dataset.data_root}")

        print("\n System:")
        print(f"  Device: {self.device}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Experiment: {self.experiment_name}")
        print("=" * 80)


# ------------------------------------------------------------------------------------
# Predefined Scene Configs
# ------------------------------------------------------------------------------------

def get_lego_config() -> InstantNGPConfig:
    cfg = InstantNGPConfig()
    cfg.dataset.scene = "lego"
    return cfg

def get_hotdog_config() -> InstantNGPConfig:
    cfg = InstantNGPConfig()
    cfg.dataset.scene = "hotdog"
    return cfg

def get_ship_config() -> InstantNGPConfig:
    cfg = InstantNGPConfig()
    cfg.dataset.scene = "ship"
    return cfg


# ------------------------------------------------------------------------------------
# Example Run
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    config = get_lego_config()
    config.print_config()

