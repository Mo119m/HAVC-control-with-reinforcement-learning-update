"""
PPO Collection Script - Optimized Version

This script trains a PPO agent and collects trajectory data for HVAC control.
Uses TrajectoryRecorder callback to save all interactions.

Key Features:
- Configurable PPO training
- Trajectory recording with callbacks
- Robust error handling
- Resume from checkpoint support
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# Assuming BEAR environment is available
try:
    from BEAR.Env.env_building import BuildingEnvReal
    from BEAR.Utils.utils_building import ParameterGenerator
    BEAR_AVAILABLE = True
except ImportError:
    BEAR_AVAILABLE = False
    print("Warning: BEAR environment not available")

from recorder_v2 import TrajectoryRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_default_data_root():
    """Get default data root path relative to this script"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "BEAR", "Data"))


def plot_training_results(trajectory_path: str, save_dir: str):
    """
    绘制 PPO 训练结果图表

    生成的图表:
    1. Episode Reward 曲线
    2. Reward 分布直方图
    3. 滑动平均 Reward
    """
    try:
        with open(trajectory_path, 'r') as f:
            data = json.load(f)

        if not data:
            logger.warning("No trajectory data to plot")
            return

        rewards = [d.get('reward', 0) for d in data]
        steps = list(range(len(rewards)))

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PPO Training Results', fontsize=14, fontweight='bold')

        # 1. Reward 曲线
        ax1 = axes[0, 0]
        ax1.plot(steps, rewards, alpha=0.3, color='blue', linewidth=0.5)
        # 滑动平均
        window = min(100, len(rewards) // 10) if len(rewards) > 10 else 1
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), smoothed, color='red', linewidth=2, label=f'MA-{window}')
            ax1.legend()
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('Step Rewards')
        ax1.grid(True, alpha=0.3)

        # 2. Reward 分布
        ax2 = axes[0, 1]
        ax2.hist(rewards, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
        ax2.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
        ax2.axvline(np.median(rewards), color='green', linestyle='--', label=f'Median: {np.median(rewards):.2f}')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Count')
        ax2.set_title('Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 累积 Reward
        ax3 = axes[1, 0]
        cumsum = np.cumsum(rewards)
        ax3.plot(steps, cumsum, color='green', linewidth=1.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Cumulative Reward')
        ax3.set_title('Cumulative Reward')
        ax3.grid(True, alpha=0.3)

        # 4. 统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
        Training Statistics
        ─────────────────────
        Total Steps: {len(rewards):,}

        Reward Stats:
          Mean:   {np.mean(rewards):.4f}
          Std:    {np.std(rewards):.4f}
          Min:    {np.min(rewards):.4f}
          Max:    {np.max(rewards):.4f}
          Median: {np.median(rewards):.4f}

        Percentiles:
          25th:   {np.percentile(rewards, 25):.4f}
          75th:   {np.percentile(rewards, 75):.4f}
          90th:   {np.percentile(rewards, 90):.4f}
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存图表
        plot_path = Path(save_dir) / "ppo_training_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Training plots saved to: {plot_path}")

    except Exception as e:
        logger.warning(f"Failed to plot training results: {e}")


@dataclass
class CollectionConfig:
    """Configuration for PPO training and data collection"""
    # Environment
    building: str = "OfficeSmall"
    weather: str = "Hot_Dry"
    location: str = "Tucson"
    data_root: str = ""  # Will be set in __post_init__

    # Training
    total_steps: int = 500000
    n_envs: int = 1
    seed: int = 42

    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Saving
    save_dir: str = "./runs_officesmall_hotdry"
    checkpoint_freq: int = 50000

    # Resume
    resume_from: Optional[str] = None

    def __post_init__(self):
        if not self.data_root:
            self.data_root = _get_default_data_root()

    def validate(self) -> None:
        """Validate configuration"""
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")


def create_environment(config: CollectionConfig):
    """
    Create vectorized BEAR environment.
    
    Args:
        config: Collection configuration
        
    Returns:
        Vectorized environment
    """
    if not BEAR_AVAILABLE:
        raise ImportError("BEAR environment not available")
    
    logger.info(f"Creating environment: {config.building}, {config.weather}, {config.location}")
    
    param = ParameterGenerator(
        config.building,
        config.weather,
        config.location,
        root=config.data_root
    )
    
    env = make_vec_env(
        lambda: BuildingEnvReal(param),
        n_envs=config.n_envs
    )
    
    # Log action space
    logger.info(f"Action space: {env.action_space}")
    try:
        logger.info(f"Action low: {env.envs[0].action_space.low}")
        logger.info(f"Action high: {env.envs[0].action_space.high}")
    except (AttributeError, IndexError):
        pass
    
    return env


def create_model(
    env,
    config: CollectionConfig,
    resume_from: Optional[str] = None
):
    """
    Create or load PPO model.
    
    Args:
        env: Environment
        config: Collection configuration
        resume_from: Path to checkpoint to resume from
        
    Returns:
        PPO model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = PPO.load(
            resume_from,
            env=env,
            device=device
        )
    else:
        logger.info("Creating new PPO model")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            verbose=1,
            seed=config.seed,
            device=device,
        )
    
    return model


def setup_callbacks(config: CollectionConfig):
    """
    Setup training callbacks.
    
    Args:
        config: Collection configuration
        
    Returns:
        Combined callback list
    """
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = []
    
    # Trajectory recorder
    traj_path = save_dir / "ppo_trajectory.json"
    recorder = TrajectoryRecorder(
        save_path=str(traj_path),
        verbose=1,
        auto_save_interval=10000  # Save every 10k steps
    )
    callbacks.append(recorder)
    
    # Checkpoint saver
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_model"
    )
    callbacks.append(checkpoint_callback)
    
    return CallbackList(callbacks)


def save_config(config: CollectionConfig, save_dir: Path) -> None:
    """Save configuration to JSON"""
    config_dict = {
        "building": config.building,
        "weather": config.weather,
        "location": config.location,
        "total_steps": config.total_steps,
        "n_envs": config.n_envs,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
    }
    
    config_path = save_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Saved config to {config_path}")


def main():
    """Main training loop"""
    
    # Create configuration (can be loaded from file or CLI in production)
    config = CollectionConfig(
        building=os.getenv("BUILDING", "OfficeSmall"),
        weather=os.getenv("WEATHER", "Hot_Dry"),
        location=os.getenv("LOCATION", "Tucson"),
        total_steps=int(os.getenv("TOTAL_STEPS", "500000")),
        save_dir=os.getenv("SAVE_DIR", "./runs_officesmall_hotdry"),
    )
    
    # Validate
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return
    
    logger.info("Starting PPO training with trajectory collection")
    logger.info(f"Configuration: {config}")
    
    # Create save directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, save_dir)
    
    try:
        # Create environment
        env = create_environment(config)
        
        # Create model
        model = create_model(env, config, resume_from=config.resume_from)
        
        # Setup callbacks
        callback = setup_callbacks(config)
        
        # Train
        logger.info(f"Training for {config.total_steps} steps")
        model.learn(
            total_timesteps=config.total_steps,
            callback=callback,
            progress_bar=True  # Show progress bar
        )
        
        # Save final model
        model_path = save_dir / "ppo_final.zip"
        model.save(str(model_path))
        logger.info(f"Saved final model to {model_path}")

        # 绘制训练结果图表
        trajectory_path = save_dir / "ppo_trajectory.json"
        if trajectory_path.exists():
            plot_training_results(str(trajectory_path), str(save_dir))

        logger.info("Training complete!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save partial model
        if 'model' in locals():
            partial_path = save_dir / "ppo_interrupted.zip"
            model.save(str(partial_path))
            logger.info(f"Saved partial model to {partial_path}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if 'env' in locals():
            env.close()
        
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
