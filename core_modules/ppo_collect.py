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
from typing import Optional
from dataclasses import dataclass

import torch
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


@dataclass
class CollectionConfig:
    """Configuration for PPO training and data collection"""
    # Environment
    building: str = "OfficeSmall"
    weather: str = "Hot_Dry"
    location: str = "Tucson"
    data_root: str = "./BEAR/Data/"
    
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
            callback=callback
        )
        
        # Save final model
        model_path = save_dir / "ppo_final.zip"
        model.save(str(model_path))
        logger.info(f"Saved final model to {model_path}")
        
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
