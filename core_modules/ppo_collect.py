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
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

from core_modules.recorder_v2 import TrajectoryRecorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_default_data_root():
    """Get default data root path relative to this script"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "BEAR", "Data"))


class MetricsCallback(CheckpointCallback):
    """
    Callback to collect PPO training metrics for plotting.
    Collect PPO training metrics for plotting.
    """
    def __init__(self, save_freq, save_path, name_prefix="rl_model", verbose=0):
        super().__init__(save_freq, save_path, name_prefix, verbose)
        self.metrics = {
            'timesteps': [],
            'ep_rew_mean': [],
            'ep_len_mean': [],
            'explained_variance': [],
            'approx_kl': [],
            'clip_fraction': [],
            'entropy_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'std': [],
        }
        self._last_logged_step = 0

    def _on_rollout_end(self) -> bool:
        """Called at the end of a rollout."""
        # Get metrics from logger
        if len(self.model.ep_info_buffer) > 0:
            self.metrics['timesteps'].append(self.num_timesteps)

            # Episode metrics
            ep_rew_mean = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            ep_len_mean = np.mean([ep_info['l'] for ep_info in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0

            self.metrics['ep_rew_mean'].append(ep_rew_mean)
            self.metrics['ep_len_mean'].append(ep_len_mean)

        return True

    def _on_step(self) -> bool:
        """Called at each step."""
        # Try to get training metrics from the model's logger
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            # Access the logger's name_to_value dict which contains recent logged values
            if hasattr(self.model.logger, 'name_to_value'):
                logs = self.model.logger.name_to_value

                # Only record if we have new data (check if step changed significantly)
                if self.num_timesteps - self._last_logged_step >= self.model.n_steps:
                    self._last_logged_step = self.num_timesteps

                    # Extract metrics
                    if 'train/explained_variance' in logs:
                        self.metrics['explained_variance'].append(logs['train/explained_variance'])
                    if 'train/approx_kl' in logs:
                        self.metrics['approx_kl'].append(logs['train/approx_kl'])
                    if 'train/clip_fraction' in logs:
                        self.metrics['clip_fraction'].append(logs['train/clip_fraction'])
                    if 'train/entropy_loss' in logs:
                        self.metrics['entropy_loss'].append(logs['train/entropy_loss'])
                    if 'train/policy_gradient_loss' in logs:
                        self.metrics['policy_loss'].append(logs['train/policy_gradient_loss'])
                    if 'train/value_loss' in logs:
                        self.metrics['value_loss'].append(logs['train/value_loss'])
                    if 'train/std' in logs:
                        self.metrics['std'].append(logs['train/std'])

        return super()._on_step()

    def _on_training_end(self) -> None:
        """Called at the end of training - save metrics."""
        # Save metrics to JSON
        metrics_path = Path(self.save_path).parent / "training_metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump({k: [float(v) if isinstance(v, (int, float, np.floating)) else v
                              for v in vals] for k, vals in self.metrics.items()}, f, indent=2)
            logger.info(f"📊 Training metrics saved to: {metrics_path}")
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")


def plot_training_results(trajectory_path: str, save_dir: str, metrics: dict = None):
    """
    Plot PPO training results.

    Generated charts:
    1. Episode Reward Mean (ep_rew_mean)
    2. Episode Length Mean (ep_len_mean)
    3. Explained Variance
    4. Approx KL & Clip Fraction
    5. Entropy & Std
    6. Policy Loss & Value Loss
    7. Step Rewards
    8. Reward Distribution
    """
    try:
        with open(trajectory_path, 'r') as f:
            data = json.load(f)

        # Try to load metrics from file if not provided
        if metrics is None:
            metrics_path = Path(save_dir) / "training_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                logger.info(f"Loaded training metrics from {metrics_path}")

        if not data:
            logger.warning("No trajectory data to plot")
            return

        rewards = [d.get('reward', 0) for d in data]
        steps = list(range(len(rewards)))

        # Create a larger 4x2 figure
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle('PPO Training Results - Comprehensive Metrics', fontsize=16, fontweight='bold')

        # 1. Episode Reward Mean
        ax1 = axes[0, 0]
        if metrics and metrics.get('ep_rew_mean'):
            ax1.plot(metrics['timesteps'][:len(metrics['ep_rew_mean'])], metrics['ep_rew_mean'],
                    color='blue', linewidth=1.5, label='ep_rew_mean')
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Episode Reward Mean')
            ax1.set_title('Episode Reward Mean (Training)')
            ax1.legend()
        else:
            # Fallback: use trajectory rewards with smoothing
            window = min(1000, len(rewards) // 10) if len(rewards) > 10 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=1.5)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Reward (Smoothed)')
            ax1.set_title('Episode Reward Mean (Smoothed)')
        ax1.grid(True, alpha=0.3)

        # 2. Episode Length Mean
        ax2 = axes[0, 1]
        if metrics and metrics.get('ep_len_mean'):
            ax2.plot(metrics['timesteps'][:len(metrics['ep_len_mean'])], metrics['ep_len_mean'],
                    color='green', linewidth=1.5)
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Episode Length')
            ax2.set_title('Episode Length Mean')
        else:
            ax2.text(0.5, 0.5, 'No ep_len_mean data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Episode Length Mean (N/A)')
        ax2.grid(True, alpha=0.3)

        # 3. Explained Variance
        ax3 = axes[1, 0]
        if metrics and metrics.get('explained_variance'):
            ax3.plot(metrics['explained_variance'], color='purple', linewidth=1.5)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Explained Variance')
            ax3.set_title('Explained Variance (ideal: close to 1)')
        else:
            ax3.text(0.5, 0.5, 'No explained_variance data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Explained Variance (N/A)')
        ax3.grid(True, alpha=0.3)

        # 4. Approx KL & Clip Fraction
        ax4 = axes[1, 1]
        if metrics and (metrics.get('approx_kl') or metrics.get('clip_fraction')):
            if metrics.get('approx_kl'):
                ax4.plot(metrics['approx_kl'], color='red', linewidth=1.5, label='approx_kl')
            if metrics.get('clip_fraction'):
                ax4_twin = ax4.twinx()
                ax4_twin.plot(metrics['clip_fraction'], color='orange', linewidth=1.5, label='clip_fraction')
                ax4_twin.set_ylabel('Clip Fraction', color='orange')
                ax4_twin.legend(loc='upper right')
            ax4.set_xlabel('Update')
            ax4.set_ylabel('Approx KL', color='red')
            ax4.set_title('Approx KL & Clip Fraction')
            ax4.legend(loc='upper left')
        else:
            ax4.text(0.5, 0.5, 'No KL/Clip data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Approx KL & Clip Fraction (N/A)')
        ax4.grid(True, alpha=0.3)

        # 5. Entropy & Std
        ax5 = axes[2, 0]
        if metrics and (metrics.get('entropy_loss') or metrics.get('std')):
            if metrics.get('entropy_loss'):
                ax5.plot(metrics['entropy_loss'], color='teal', linewidth=1.5, label='entropy_loss')
            if metrics.get('std'):
                ax5_twin = ax5.twinx()
                ax5_twin.plot(metrics['std'], color='brown', linewidth=1.5, label='std')
                ax5_twin.set_ylabel('Std', color='brown')
                ax5_twin.legend(loc='upper right')
            ax5.set_xlabel('Update')
            ax5.set_ylabel('Entropy Loss', color='teal')
            ax5.set_title('Entropy Loss & Std')
            ax5.legend(loc='upper left')
        else:
            ax5.text(0.5, 0.5, 'No Entropy/Std data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Entropy & Std (N/A)')
        ax5.grid(True, alpha=0.3)

        # 6. Policy Loss & Value Loss
        ax6 = axes[2, 1]
        if metrics and (metrics.get('policy_loss') or metrics.get('value_loss')):
            if metrics.get('policy_loss'):
                ax6.plot(metrics['policy_loss'], color='blue', linewidth=1.5, label='policy_loss')
            if metrics.get('value_loss'):
                ax6_twin = ax6.twinx()
                ax6_twin.plot(metrics['value_loss'], color='red', linewidth=1.5, label='value_loss')
                ax6_twin.set_ylabel('Value Loss', color='red')
                ax6_twin.legend(loc='upper right')
            ax6.set_xlabel('Update')
            ax6.set_ylabel('Policy Loss', color='blue')
            ax6.set_title('Policy Loss & Value Loss')
            ax6.legend(loc='upper left')
        else:
            ax6.text(0.5, 0.5, 'No Loss data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Policy Loss & Value Loss (N/A)')
        ax6.grid(True, alpha=0.3)

        # 7. Step Rewards (raw)
        ax7 = axes[3, 0]
        ax7.plot(steps, rewards, alpha=0.3, color='blue', linewidth=0.5)
        window = min(100, len(rewards) // 10) if len(rewards) > 10 else 1
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax7.plot(range(window-1, len(rewards)), smoothed, color='red', linewidth=2, label=f'MA-{window}')
            ax7.legend()
        ax7.set_xlabel('Step')
        ax7.set_ylabel('Reward')
        ax7.set_title('Step Rewards (Raw + Smoothed)')
        ax7.grid(True, alpha=0.3)

        # 8. Statistics Summary
        ax8 = axes[3, 1]
        ax8.axis('off')
        stats_text = f"""
        PPO Training Statistics
        ═══════════════════════════════════

        Trajectory Data:
          Total Steps:     {len(rewards):,}
          Mean Reward:     {np.mean(rewards):.4f}
          Std Reward:      {np.std(rewards):.4f}
          Min Reward:      {np.min(rewards):.4f}
          Max Reward:      {np.max(rewards):.4f}

        Percentiles:
          25th:  {np.percentile(rewards, 25):.4f}
          50th:  {np.percentile(rewards, 50):.4f}
          75th:  {np.percentile(rewards, 75):.4f}
          90th:  {np.percentile(rewards, 90):.4f}
        """
        ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save figure
        plot_path = Path(save_dir) / "ppo_training_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Training plots saved to: {plot_path}")

        # Save metrics data
        if metrics:
            metrics_path = Path(save_dir) / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({k: [float(v) if isinstance(v, (int, float, np.floating)) else v for v in vals]
                          for k, vals in metrics.items()}, f, indent=2)
            logger.info(f"📊 Training metrics saved to: {metrics_path}")

    except Exception as e:
        logger.warning(f"Failed to plot training results: {e}")
        import traceback
        traceback.print_exc()


@dataclass
class CollectionConfig:
    """Configuration for PPO training and data collection"""
    # Environment
    building: str = "OfficeSmall"
    weather: str = "Hot_Dry"
    location: str = "Tucson"
    data_root: str = ""  # Will be set in __post_init__

    # Training
    total_steps: int = 100000
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
        Tuple of (callback_list, metrics_callback)
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

    # Metrics callback (collects PPO training metrics for plotting)
    metrics_callback = MetricsCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_model",
        verbose=1
    )
    callbacks.append(metrics_callback)

    return CallbackList(callbacks), metrics_callback


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
        total_steps=int(os.getenv("TOTAL_STEPS", "100000")),
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
        callback, metrics_callback = setup_callbacks(config)

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

        # Plot training results (using collected metrics)
        trajectory_path = save_dir / "ppo_trajectory.json"
        if trajectory_path.exists():
            plot_training_results(
                str(trajectory_path),
                str(save_dir),
                metrics=metrics_callback.metrics
            )

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
