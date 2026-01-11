"""
Progressive On-policy Training Script

This script implements a progressive transition from offline to on-policy training:
1. Start with existing rollout data (offline)
2. Train for 1 epoch
3. Generate new episodes with updated policy
4. Mix old and new data with increasing weight on new data
5. Gradually discard old data and train on fresher data
6. Repeat until fully on-policy

This balances data efficiency with theoretical correctness.
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_rollout(config_path: str, num_episodes: int, building: str, climate: str, location: str) -> bool:
    """Run LLM rollout to generate new episodes"""
    logger.info(f"🎬 Running rollout: {num_episodes} episodes")

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "BUILDING": building,
        "CLIMATE": climate,
        "LOCATION": location,
        "NUM_EPISODES": str(num_episodes),
    }

    try:
        result = subprocess.run(
            ["python", "core_modules/main_pipeline.py", "--config", config_path, "--stage", "rollout"],
            env=env,
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Rollout failed: {e}")
        return False


def run_finetune(config_path: str, clip_eps: float, epochs: int = 1) -> bool:
    """Run fine-tuning with specified clip epsilon"""
    logger.info(f"🎓 Running fine-tuning: clip_eps={clip_eps}, epochs={epochs}")

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CLIP_EPS": str(clip_eps),
        "EPOCHS": str(epochs),
    }

    try:
        result = subprocess.run(
            ["python", "core_modules/main_pipeline.py", "--config", config_path, "--stage", "finetune"],
            env=env,
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Fine-tuning failed: {e}")
        return False


def progressive_training(config_path: str):
    """
    Run progressive on-policy training according to schedule.

    Schedule:
    - Epoch 1: Train on initial 50 episodes (clip=0.20)
    - Refresh 1: Add 10 new episodes (total 60), weight new 2x (clip=0.18)
    - Epoch 2: Train on mixed data
    - Refresh 2: Add 10 more, keep only last 20 episodes (clip=0.15)
    - Epoch 3: Train on recent 20 episodes
    - Refresh 3: Add 5 more, keep only last 10 episodes (clip=0.12)
    - Epoch 4: Train on most recent 10 episodes (most on-policy)
    """
    logger.info("=" * 70)
    logger.info("PROGRESSIVE ON-POLICY TRAINING")
    logger.info("=" * 70)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    if not config.get("use_progressive_onpolicy", False):
        logger.error("Progressive on-policy training not enabled in config!")
        logger.error("Set 'use_progressive_onpolicy': true in config file")
        return False

    schedule = config.get("refresh_schedule", [])
    initial_clip = config.get("initial_clip_eps", 0.20)
    building = config["building"]
    climate = config["weather"]
    location = config["location"]

    logger.info(f"\nConfiguration:")
    logger.info(f"  Building: {building}")
    logger.info(f"  Climate: {climate}")
    logger.info(f"  Location: {location}")
    logger.info(f"  Refresh schedule: {len(schedule)} refreshes")
    logger.info("")

    # Phase 0: Initial data should already exist (50 episodes)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 0: Verify initial data exists")
    logger.info("=" * 70)
    rollout_file = Path(config["base_dir"]) / config.get("llm_rollout_dir", "03_llm_rollout") / "llm_rollout.json"
    if not rollout_file.exists():
        logger.error(f"Initial rollout data not found: {rollout_file}")
        logger.error("Please run Stage 3 (rollout) first to generate initial data")
        return False

    with open(rollout_file) as f:
        data = json.load(f)
    initial_episodes = len(set(entry.get("episode", 0) for entry in data))
    logger.info(f"✅ Found initial data: {len(data)} steps, {initial_episodes} episodes")

    # Phase 1: Train on initial data
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Train on initial offline data")
    logger.info("=" * 70)
    if not run_finetune(config_path, clip_eps=initial_clip, epochs=1):
        logger.error("Phase 1 failed")
        return False
    logger.info("✅ Phase 1 completed")

    # Progressive refreshes
    for refresh_idx, refresh_config in enumerate(schedule, 1):
        after_epoch = refresh_config["after_epoch"]
        new_episodes = refresh_config["new_episodes"]
        clip_eps = refresh_config["clip_eps"]

        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE {refresh_idx + 1}: Refresh {refresh_idx} (after epoch {after_epoch})")
        logger.info("=" * 70)

        # Step 1: Generate new episodes with current policy
        logger.info(f"\nStep 1: Generate {new_episodes} new episodes")
        if not run_rollout(config_path, new_episodes, building, climate, location):
            logger.error(f"Refresh {refresh_idx} rollout failed")
            return False

        # Step 2: Train on mixed data
        logger.info(f"\nStep 2: Train with clip_eps={clip_eps}")
        if not run_finetune(config_path, clip_eps=clip_eps, epochs=1):
            logger.error(f"Refresh {refresh_idx} fine-tuning failed")
            return False

        logger.info(f"✅ Refresh {refresh_idx} completed")

    logger.info("\n" + "=" * 70)
    logger.info("🎉 PROGRESSIVE TRAINING COMPLETED!")
    logger.info("=" * 70)
    logger.info("\nModel has been progressively transitioned to on-policy training")
    logger.info(f"Final model saved in: {config['base_dir']}/04_finetuning/final_model")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run progressive on-policy training")
    parser.add_argument("--config", type=str, default="config_enhanced.json",
                        help="Path to configuration file")
    args = parser.parse_args()

    success = progressive_training(args.config)
    sys.exit(0 if success else 1)
