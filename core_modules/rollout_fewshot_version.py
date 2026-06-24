"""
LLM Rollout with Few-Shot Examples - Optimized Version

Improvements:
1. Type hints and dataclasses
2. Comprehensive error handling
3. Configuration management
4. Better logging
5. Validation functions
6. Checkpoint saving
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3.common.env_util import make_vec_env

try:
    from BEAR.Utils.utils_building import ParameterGenerator
    from BEAR.Env.env_building import BuildingEnvReal
    BEAR_AVAILABLE = True
except ImportError:
    BEAR_AVAILABLE = False
    print("Warning: BEAR not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

def _get_default_data_root():
    """Get default data root path relative to this script"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "BEAR", "Data"))


@dataclass
class RolloutConfig:
    """Configuration for LLM rollout"""
    # Environment
    building: str = "OfficeSmall"
    climate: str = "Hot_Dry"
    location: str = "Tucson"
    target: float = 22.0
    data_root: str = ""  # Will be set in __post_init__

    # Rollout
    max_steps: int = 200
    num_episodes: int = 1  # Number of episodes to run
    episode_offset_stride: int = 0  # >0: episode i starts at weather window i*stride
    hist_keep: int = 6
    hist_lines_in_prompt: int = 3

    # LLM
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.3

    # Few-shot
    fewshot_json: Optional[str] = None
    k_fewshot: int = 3
    fewshot_alpha: float = 0.6

    # Output
    save_path: str = "./outputs/llm_rollout.json"
    save_interval: int = 50

    def __post_init__(self):
        """Load from environment variables"""
        # Set default data_root if not provided
        if not self.data_root:
            self.data_root = _get_default_data_root()

        self.building = os.getenv("BUILDING", self.building)
        self.climate = os.getenv("CLIMATE", self.climate)
        self.location = os.getenv("LOCATION", self.location)
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.save_path = os.getenv("SAVE_PATH", self.save_path)

        if os.getenv("MAX_STEPS"):
            self.max_steps = int(os.getenv("MAX_STEPS"))
        if os.getenv("NUM_EPISODES"):
            self.num_episodes = int(os.getenv("NUM_EPISODES"))
        if os.getenv("EPISODE_OFFSET_STRIDE"):
            self.episode_offset_stride = int(os.getenv("EPISODE_OFFSET_STRIDE"))
        if os.getenv("FEWSHOT_JSON"):
            self.fewshot_json = os.getenv("FEWSHOT_JSON")
        if os.getenv("K_FEWSHOT"):
            self.k_fewshot = int(os.getenv("K_FEWSHOT"))
        if os.getenv("FEWSHOT_ALPHA"):
            self.fewshot_alpha = float(os.getenv("FEWSHOT_ALPHA"))


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_rollout_results(trajectory: List[Dict], save_path: str):
    """
    绘制 LLM Rollout 结果图表

    生成的图表:
    1. Step-by-step Reward
    2. Reward 分布
    3. 累积 Reward
    4. 室内温度变化 (如果有)
    """
    try:
        if not trajectory:
            logger.warning("No trajectory data to plot")
            return

        rewards = [d.get('reward', 0) for d in trajectory]
        steps = list(range(len(rewards)))

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('LLM Rollout Results', fontsize=14, fontweight='bold')

        # 1. Step Rewards
        ax1 = axes[0, 0]
        ax1.plot(steps, rewards, color='blue', linewidth=1, alpha=0.7)
        # 滑动平均
        window = min(20, len(rewards) // 5) if len(rewards) > 5 else 1
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(rewards)), smoothed, color='red', linewidth=2, label=f'MA-{window}')
            ax1.legend()
        ax1.axhline(y=np.mean(rewards), color='green', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(rewards):.2f}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('Step-by-Step Rewards')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Reward 分布
        ax2 = axes[0, 1]
        ax2.hist(rewards, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax2.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax2.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Count')
        ax2.set_title('Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 累积 Reward
        ax3 = axes[1, 0]
        cumsum = np.cumsum(rewards)
        ax3.plot(steps, cumsum, color='green', linewidth=1.5)
        ax3.fill_between(steps, cumsum, alpha=0.3, color='green')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Cumulative Reward')
        ax3.set_title(f'Cumulative Reward (Total: {cumsum[-1]:.2f})')
        ax3.grid(True, alpha=0.3)

        # 4. 统计信息
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
        LLM Rollout Statistics
        ──────────────────────────
        Total Steps: {len(rewards):,}

        Reward Stats:
          Mean:   {np.mean(rewards):.4f}
          Std:    {np.std(rewards):.4f}
          Min:    {np.min(rewards):.4f}
          Max:    {np.max(rewards):.4f}
          Total:  {np.sum(rewards):.4f}

        Percentiles:
          25th:   {np.percentile(rewards, 25):.4f}
          50th:   {np.percentile(rewards, 50):.4f}
          75th:   {np.percentile(rewards, 75):.4f}
        """
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()

        # 保存
        plot_path = Path(save_path).parent / "rollout_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"📊 Rollout plots saved to: {plot_path}")

    except Exception as e:
        logger.warning(f"Failed to plot rollout results: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def extract_outside_temp(obs: List[float]) -> float:
    """
    Extract outside temperature from observation.
    
    Args:
        obs: Observation array (3n+2 structure)
        
    Returns:
        Outside temperature
    """
    try:
        # obs structure: [temps(n), outside(1), ghi(n), ground(1), occupancy(n)]
        n = (len(obs) - 2) // 3
        return float(obs[n])
    except Exception as e:
        logger.warning(f"Failed to extract outside temp: {e}")
        return 0.0


def validate_action(action: List[float], n_zones: int) -> Tuple[bool, str]:
    """
    Validate action array.
    
    Args:
        action: Action array
        n_zones: Expected number of zones
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(action, list):
        return False, "Action is not a list"
    
    if len(action) != n_zones:
        return False, f"Action length {len(action)} != {n_zones}"
    
    for i, a in enumerate(action):
        try:
            val = float(a)
            if not -1.05 <= val <= 1.05:
                return False, f"Action[{i}] = {val} out of range [-1, 1]"
        except (ValueError, TypeError):
            return False, f"Action[{i}] is not a number"
    
    return True, ""


def save_checkpoint(
    logs: List[Dict],
    save_path: str,
    step: int
) -> None:
    """
    Save checkpoint of rollout logs.
    
    Args:
        logs: List of log entries
        save_path: Path to save checkpoint
        step: Current step number
    """
    checkpoint_path = save_path.replace('.json', f'_step{step}.json')
    
    try:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


# ============================================================================
# Main Rollout
# ============================================================================

def run_single_episode(config: RolloutConfig, episode_num: int = 0) -> List[Dict]:
    """
    Run a single episode of LLM rollout with few-shot examples.

    Args:
        config: Rollout configuration
        episode_num: Episode number (for logging)

    Returns:
        List of log entries for this episode
    """
    # Import here to avoid circular dependency
    from prompt_builder_control import build_prompt, zone_count_from_obs
    from llm_agent_colab import call_llm, parse_actions_with_validation
    from few_shot_auto import (
        load_examples, select_examples,
        format_few_shot_block, inject_few_shot,
        SelectionConfig
    )
    
    # Create output directory
    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
    
    # Load few-shot examples if provided
    ex_dataset = None
    if config.fewshot_json:
        try:
            ex_dataset = load_examples(config.fewshot_json)
            logger.info(f"Loaded {len(ex_dataset)} few-shot examples")
        except Exception as e:
            logger.warning(f"Failed to load few-shot examples: {e}")
    
    # Create environment
    if not BEAR_AVAILABLE:
        raise RuntimeError("BEAR environment not available")
    
    param = ParameterGenerator(
        config.building,
        config.climate,
        config.location,
        root=config.data_root,
        target=config.target
    )
    
    vec_env = make_vec_env(lambda: BuildingEnvReal(param), n_envs=1)
    env = vec_env.envs[0]

    logger.info(f"Environment action space: {env.action_space.low} to {env.action_space.high}")

    # Reset environment
    reset_ret = env.reset()
    obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

    # Multi-window diversity: start each episode at a different weather window so
    # the rollout (and the distilled training data) covers varied conditions
    # rather than the single deterministic window epochs=0.
    offset = (episode_num * config.episode_offset_stride) % (env.length_of_weather - 1) \
        if config.episode_offset_stride > 0 else 0
    if offset > 0:
        env.epochs = offset
        T_initial = env.X_new
        ghi_repeated = np.full(T_initial.shape, env.ghi[env.epochs])
        occ_repeated = np.full(T_initial.shape, env.Occupower / 1000)
        env.state = np.concatenate((
            T_initial,
            env.OutTemp[env.epochs].reshape(-1),
            ghi_repeated,
            env.GroundTemp[env.epochs].reshape(-1),
            occ_repeated,
        ), axis=0)
        obs = env.state
        logger.info(f"[Episode {episode_num+1}] weather window offset: {offset}")

    obs = np.array(obs).flatten().tolist()
    
    # Initialize history and logs
    history = deque(maxlen=config.hist_keep)
    logs = []
    
    # Main loop
    for step in range(config.max_steps):
        logger.info(f"[Episode {episode_num+1}] Step {step+1}/{config.max_steps}")
        
        # Get number of zones
        n_zones = zone_count_from_obs(obs)
        
        # Build prompt
        try:
            prompt = build_prompt(
                obs=obs,
                building=config.building,
                location=config.location,
                climate=config.climate,
                target=config.target,
                round_idx=step + 1,
                history=list(history)
            )
        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            break
        
        # Inject few-shot examples
        few_block = None
        if ex_dataset:
            try:
                selection_cfg = SelectionConfig(
                    k=config.k_fewshot,
                    alpha=config.fewshot_alpha
                )
                examples = select_examples(
                    ex_dataset,
                    current_obs=obs,
                    config=selection_cfg,
                    building=config.building,
                    climate=config.climate,
                    location=config.location
                )
                few_block = format_few_shot_block(examples, target=config.target, n=n_zones)
                prompt = inject_few_shot(prompt, few_block)
            except Exception as e:
                logger.warning(f"Failed to inject few-shot: {e}")
        
        # Call LLM
        try:
            raw_text = call_llm(
                prompt,
                n_actions=n_zones,
                model_name=config.model_name,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raw_text = ""
        
        # Parse actions
        action_unit, meta = parse_actions_with_validation(raw_text, n_zones)
        
        if action_unit is None:
            logger.warning("Failed to parse actions, using zero action")
            action_unit = [0.0] * n_zones
            meta = {"parsed_from": "fallback_zero", "warnings": ["parse_failed"]}
        
        # Validate action
        is_valid, error_msg = validate_action(action_unit, n_zones)
        if not is_valid:
            logger.warning(f"Invalid action: {error_msg}")
            action_unit = [0.0] * n_zones
        
        logger.debug(f"Action: {np.round(action_unit, 3).tolist()}")
        
        # Environment step
        try:
            step_ret = env.step(action_unit)
            
            if len(step_ret) == 5:
                obs_next, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            else:
                obs_next, reward, done, info = step_ret
            
            obs_next = np.array(obs_next).flatten().tolist()
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            break
        
        # Update history
        history.append({
            "step": step + 1,
            "action": [float(x) for x in action_unit],
            "reward": float(reward),
            "env_temp": extract_outside_temp(obs),
            "obs_before": obs,
            "obs_after": obs_next,
            "power": (info or {}).get("power", None)
        })
        
        # Log entry
        log_entry = {
            "step": step,
            "prompt": prompt,
            "few_shot": few_block or "",
            "llm_raw": raw_text,
            "parsed_from": (meta or {}).get("parsed_from", "unknown"),
            "action_unit": [float(x) for x in action_unit],
            "action_env": action_unit,  # Same as action_unit
            "reward": float(reward),
            "done": bool(done),
            "obs": obs,
            "next_obs": obs_next,
            "env_temp": extract_outside_temp(obs),
            "used_fallback": action_unit == [0.0] * n_zones
        }
        
        logs.append(log_entry)
        
        # Save checkpoint
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(logs, config.save_path, step + 1)
        
        # Update observation
        obs = obs_next
        
        # Check if done
        if done:
            logger.info(f"[Episode {episode_num+1}] Episode ended at step {step+1}")
            break

    # Print episode summary
    if logs:
        rewards = [log["reward"] for log in logs]
        logger.info(f"\n[Episode {episode_num+1}] Summary:")
        logger.info(f"  Total steps: {len(logs)}")
        logger.info(f"  Mean reward: {np.mean(rewards):.2f}")
        logger.info(f"  Total reward: {np.sum(rewards):.2f}")

    return logs


def run_rollout(config: RolloutConfig) -> List[Dict]:
    """
    Run multiple episodes of LLM rollout and merge results.

    Args:
        config: Rollout configuration

    Returns:
        Merged list of log entries from all episodes
    """
    num_episodes = config.num_episodes
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting Multi-Episode LLM Rollout")
    logger.info(f"{'='*60}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Steps per episode: {config.max_steps}")
    logger.info(f"Total expected steps: {num_episodes * config.max_steps}")
    logger.info(f"{'='*60}\n")

    # Check for Drive backup path
    drive_backup_path = os.getenv("DRIVE_BACKUP_PATH")
    if drive_backup_path:
        logger.info(f"💾 Drive backup enabled: {drive_backup_path}")
        os.makedirs(drive_backup_path, exist_ok=True)

    # Try to load existing progress
    all_logs = []
    episode_stats = []
    completed_episodes = set()

    # Check if there's existing data to resume from
    if os.path.exists(config.save_path):
        try:
            with open(config.save_path, 'r') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list) and len(existing_data) > 0:
                all_logs = existing_data
                completed_episodes = set(entry.get('episode', 0) for entry in existing_data)
                logger.info(f"📂 Resuming from existing data:")
                logger.info(f"   - Loaded {len(all_logs)} steps")
                logger.info(f"   - Completed episodes: {sorted(completed_episodes)}")
        except Exception as e:
            logger.warning(f"Failed to load existing data: {e}")

    for episode_idx in range(num_episodes):
        # Skip if already completed
        if episode_idx in completed_episodes:
            logger.info(f"\n⏭️  Episode {episode_idx + 1} already completed, skipping...")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode_idx + 1}/{num_episodes}")
        logger.info(f"{'='*60}\n")

        try:
            # Run single episode
            episode_logs = run_single_episode(config, episode_num=episode_idx)

            # Add episode number to each log entry
            for log_entry in episode_logs:
                log_entry["episode"] = episode_idx

            # Collect stats
            if episode_logs:
                rewards = [log["reward"] for log in episode_logs]
                episode_stats.append({
                    "episode": episode_idx,
                    "steps": len(episode_logs),
                    "mean_reward": np.mean(rewards),
                    "total_reward": np.sum(rewards),
                    "min_reward": np.min(rewards),
                    "max_reward": np.max(rewards)
                })

            # Merge logs
            all_logs.extend(episode_logs)
            completed_episodes.add(episode_idx)

            logger.info(f"\n✅ Episode {episode_idx + 1} completed: {len(episode_logs)} steps")

            # IMMEDIATELY save to local after each episode
            try:
                os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
                with open(config.save_path, "w", encoding="utf-8") as f:
                    json.dump(all_logs, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Saved to local: {config.save_path}")
            except Exception as e:
                logger.error(f"Failed to save locally: {e}")

            # IMMEDIATELY backup to Drive after each episode
            if drive_backup_path:
                try:
                    import shutil
                    drive_file = os.path.join(drive_backup_path, "llm_rollout.json")
                    shutil.copy2(config.save_path, drive_file)

                    # Also save episode-specific backup
                    drive_episode_file = os.path.join(
                        drive_backup_path,
                        f"llm_rollout_ep{episode_idx+1}.json"
                    )
                    with open(drive_episode_file, "w", encoding="utf-8") as f:
                        json.dump(episode_logs, f, ensure_ascii=False, indent=2)

                    logger.info(f"☁️  Backed up to Drive: {drive_file}")
                    logger.info(f"☁️  Episode backup: {drive_episode_file}")
                except Exception as e:
                    logger.error(f"Failed to backup to Drive: {e}")

        except Exception as e:
            logger.error(f"❌ Episode {episode_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()

            # Save current progress even on failure
            if all_logs:
                try:
                    with open(config.save_path, "w", encoding="utf-8") as f:
                        json.dump(all_logs, f, ensure_ascii=False, indent=2)
                    logger.info(f"💾 Saved progress before failure")

                    if drive_backup_path:
                        import shutil
                        drive_file = os.path.join(drive_backup_path, "llm_rollout.json")
                        shutil.copy2(config.save_path, drive_file)
                        logger.info(f"☁️  Backed up to Drive before failure")
                except:
                    pass

            # Continue with next episode

    # Save merged results
    logger.info(f"\n{'='*60}")
    logger.info(f"All Episodes Completed")
    logger.info(f"{'='*60}\n")

    if all_logs:
        # Save merged trajectory
        try:
            os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
            with open(config.save_path, "w", encoding="utf-8") as f:
                json.dump(all_logs, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved merged rollout: {config.save_path}")
        except Exception as e:
            logger.error(f"Failed to save merged rollout: {e}")

        # Print overall summary
        all_rewards = [log["reward"] for log in all_logs]
        logger.info(f"\n📊 Overall Rollout Summary:")
        logger.info(f"  Total episodes: {num_episodes}")
        logger.info(f"  Total steps: {len(all_logs):,}")
        logger.info(f"  Mean reward: {np.mean(all_rewards):.4f}")
        logger.info(f"  Std reward: {np.std(all_rewards):.4f}")
        logger.info(f"  Min reward: {np.min(all_rewards):.4f}")
        logger.info(f"  Max reward: {np.max(all_rewards):.4f}")
        logger.info(f"  Total reward: {np.sum(all_rewards):.4f}")

        # Print per-episode stats
        if episode_stats:
            logger.info(f"\n📈 Per-Episode Statistics:")
            for stats in episode_stats:
                logger.info(
                    f"  Episode {stats['episode']+1}: "
                    f"{stats['steps']} steps, "
                    f"mean={stats['mean_reward']:.2f}, "
                    f"total={stats['total_reward']:.2f}"
                )

        # Plot results
        try:
            plot_rollout_results(all_logs, config.save_path)
        except Exception as e:
            logger.warning(f"Failed to plot results: {e}")

    else:
        logger.warning("No logs collected from any episode!")

    return all_logs


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM rollout with few-shot")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--building", type=str, help="Building type")
    parser.add_argument("--climate", type=str, help="Climate type")
    parser.add_argument("--max_steps", type=int, help="Maximum steps")
    parser.add_argument("--fewshot_json", type=str, help="Few-shot examples JSON")
    parser.add_argument("--output", type=str, help="Output path")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = RolloutConfig(**config_dict)
    else:
        config = RolloutConfig()
    
    # Override with CLI args
    if args.building:
        config.building = args.building
    if args.climate:
        config.climate = args.climate
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.fewshot_json:
        config.fewshot_json = args.fewshot_json
    if args.output:
        config.save_path = args.output
    
    logger.info("Starting LLM rollout...")
    logger.info(f"Config: {config}")
    
    try:
        logs = run_rollout(config)
        logger.info(f"Rollout completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
