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
import numpy as np

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

@dataclass
class RolloutConfig:
    """Configuration for LLM rollout"""
    # Environment
    building: str = "OfficeSmall"
    climate: str = "Hot_Dry"
    location: str = "Tucson"
    target: float = 22.0
    data_root: str = "./BEAR/Data/"
    
    # Rollout
    max_steps: int = 200
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
        self.building = os.getenv("BUILDING", self.building)
        self.climate = os.getenv("CLIMATE", self.climate)
        self.location = os.getenv("LOCATION", self.location)
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.save_path = os.getenv("SAVE_PATH", self.save_path)
        
        if os.getenv("MAX_STEPS"):
            self.max_steps = int(os.getenv("MAX_STEPS"))
        if os.getenv("FEWSHOT_JSON"):
            self.fewshot_json = os.getenv("FEWSHOT_JSON")
        if os.getenv("K_FEWSHOT"):
            self.k_fewshot = int(os.getenv("K_FEWSHOT"))
        if os.getenv("FEWSHOT_ALPHA"):
            self.fewshot_alpha = float(os.getenv("FEWSHOT_ALPHA"))


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

def run_rollout(config: RolloutConfig) -> List[Dict]:
    """
    Run LLM rollout with few-shot examples.
    
    Args:
        config: Rollout configuration
        
    Returns:
        List of log entries
    """
    # Import here to avoid circular dependency
    from prompt_builder_control import build_prompt, zone_count_from_obs
    from llm_agent_colab import call_llm, parse_actions_with_validation
    from few_shot_auto import (
        load_examples, select_examples,
        format_few_shot_block, inject_few_shot
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
    obs = np.array(obs).flatten().tolist()
    
    # Initialize history and logs
    history = deque(maxlen=config.hist_keep)
    logs = []
    
    # Main loop
    for step in range(config.max_steps):
        logger.info(f"Step {step+1}/{config.max_steps}")
        
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
                history=list(history),
                history_lines=config.hist_lines_in_prompt
            )
        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            break
        
        # Inject few-shot examples
        few_block = None
        if ex_dataset:
            try:
                examples = select_examples(
                    ex_dataset,
                    current_obs=obs,
                    k=config.k_fewshot,
                    alpha=config.fewshot_alpha,
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
            logger.info(f"Episode ended at step {step+1}")
            break
    
    # Save final results
    try:
        with open(config.save_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved final rollout: {config.save_path}")
    except Exception as e:
        logger.error(f"Failed to save final rollout: {e}")
    
    # Print summary
    if logs:
        rewards = [log["reward"] for log in logs]
        logger.info(f"\nRollout Summary:")
        logger.info(f"  Total steps: {len(logs)}")
        logger.info(f"  Mean reward: {np.mean(rewards):.2f}")
        logger.info(f"  Std reward: {np.std(rewards):.2f}")
        logger.info(f"  Min reward: {np.min(rewards):.2f}")
        logger.info(f"  Max reward: {np.max(rewards):.2f}")
        logger.info(f"  Total reward: {np.sum(rewards):.2f}")
    
    return logs


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
