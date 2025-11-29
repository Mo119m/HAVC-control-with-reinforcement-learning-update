"""
Self-Distillation Data Preparation

This module filters high-reward trajectories from LLM rollout for fine-tuning.
This implements the "self-distillation" approach where the LLM learns from its
own successful behaviors.

自我蒸馏核心逻辑：
1. LLM生成大量轨迹 (llm_rollout.json)
2. 筛选reward高的轨迹 (这个模块)
3. 用筛选后的数据微调LLM
4. 微调后的LLM性能提升
"""

import os
import json
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for self-distillation data preparation"""
    # Input
    llm_rollout_json: str = "./outputs/llm_rollout.json"
    
    # Filtering
    min_reward_percentile: float = 0.5  # Keep top 50% by reward
    reward_q_low: float = 0.05  # Remove bottom 5%
    reward_q_high: float = 0.99  # Remove top 1% (outliers)
    
    # Output
    output_json: str = "./outputs/distillation_data.json"
    
    # Validation
    check_action_validity: bool = True
    check_parse_success: bool = True
    
    def __post_init__(self):
        """Load from environment"""
        self.llm_rollout_json = os.getenv("LLM_ROLLOUT_JSON", self.llm_rollout_json)
        self.output_json = os.getenv("DISTILLATION_OUTPUT", self.output_json)
        
        if os.getenv("MIN_REWARD_PERCENTILE"):
            self.min_reward_percentile = float(os.getenv("MIN_REWARD_PERCENTILE"))


def load_llm_rollout(path: str) -> List[Dict]:
    """
    Load LLM rollout trajectory.
    
    Args:
        path: Path to llm_rollout.json
        
    Returns:
        List of trajectory entries
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM rollout not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} entries from {path}")
    return data


def is_valid_entry(entry: Dict, check_actions: bool = True) -> Tuple[bool, str]:
    """
    Validate a trajectory entry.
    
    Args:
        entry: Trajectory entry
        check_actions: Whether to validate actions
        
    Returns:
        Tuple of (is_valid, reason)
    """
    # Required fields
    required = ["prompt", "action_unit", "reward", "done", "obs", "next_obs"]
    for field in required:
        if field not in entry:
            return False, f"Missing field: {field}"
    
    # Check if parsing succeeded
    parsed_from = entry.get("parsed_from", "")
    if parsed_from in ["failed", "fallback_zero"]:
        return False, f"Parse failed: {parsed_from}"
    
    # Check if used fallback
    if entry.get("used_fallback", False):
        return False, "Used fallback action"
    
    # Check actions
    if check_actions:
        actions = entry.get("action_unit", [])
        if not isinstance(actions, list) or len(actions) == 0:
            return False, "Invalid actions"
        
        # Check action range
        for a in actions:
            try:
                val = float(a)
                if not -1.05 <= val <= 1.05:
                    return False, f"Action out of range: {val}"
            except (ValueError, TypeError):
                return False, "Action not numeric"
    
    return True, ""


def filter_by_reward(
    data: List[Dict],
    min_percentile: float = 0.5,
    q_low: float = 0.05,
    q_high: float = 0.99
) -> List[Dict]:
    """
    Filter trajectories by reward.
    
    Strategy:
    1. Remove outliers (below q_low or above q_high)
    2. Keep trajectories with reward >= min_percentile
    
    Args:
        data: List of trajectory entries
        min_percentile: Minimum reward percentile to keep
        q_low: Lower quantile for outlier removal
        q_high: Upper quantile for outlier removal
        
    Returns:
        Filtered list
    """
    if len(data) == 0:
        return []
    
    # Extract rewards
    rewards = np.array([d["reward"] for d in data])
    
    # Remove outliers
    r_low = np.quantile(rewards, q_low)
    r_high = np.quantile(rewards, q_high)
    
    logger.info(f"Reward range: [{r_low:.2f}, {r_high:.2f}]")
    
    data_filtered = [
        d for d in data
        if r_low <= d["reward"] <= r_high
    ]
    
    logger.info(f"After outlier removal: {len(data_filtered)} entries")
    
    # Filter by percentile
    if len(data_filtered) > 0:
        rewards_filtered = np.array([d["reward"] for d in data_filtered])
        threshold = np.quantile(rewards_filtered, min_percentile)
        
        logger.info(f"Reward threshold (p={min_percentile}): {threshold:.2f}")
        
        data_final = [
            d for d in data_filtered
            if d["reward"] >= threshold
        ]
        
        logger.info(f"After reward filtering: {len(data_final)} entries")
        
        return data_final
    
    return data_filtered


def prepare_distillation_data(config: DistillationConfig) -> List[Dict]:
    """
    Prepare self-distillation training data.
    
    This function implements the core self-distillation logic:
    1. Load LLM-generated trajectories
    2. Validate entries (parsing success, action validity)
    3. Filter by reward (keep high-performing trajectories)
    4. Save filtered data for fine-tuning
    
    Args:
        config: Distillation configuration
        
    Returns:
        List of filtered trajectory entries
    """
    logger.info("="*60)
    logger.info("Self-Distillation Data Preparation")
    logger.info("="*60)
    
    # Load LLM rollout
    data = load_llm_rollout(config.llm_rollout_json)
    
    # Validate entries
    valid_data = []
    validation_stats = {
        "total": len(data),
        "valid": 0,
        "invalid_parse": 0,
        "invalid_action": 0,
        "invalid_other": 0
    }
    
    for entry in data:
        is_valid, reason = is_valid_entry(
            entry,
            check_actions=config.check_action_validity
        )
        
        if is_valid:
            valid_data.append(entry)
            validation_stats["valid"] += 1
        else:
            if "parse" in reason.lower() or "fallback" in reason.lower():
                validation_stats["invalid_parse"] += 1
            elif "action" in reason.lower():
                validation_stats["invalid_action"] += 1
            else:
                validation_stats["invalid_other"] += 1
    
    logger.info(f"\nValidation Statistics:")
    logger.info(f"  Total entries: {validation_stats['total']}")
    logger.info(f"  Valid: {validation_stats['valid']}")
    logger.info(f"  Invalid (parse): {validation_stats['invalid_parse']}")
    logger.info(f"  Invalid (action): {validation_stats['invalid_action']}")
    logger.info(f"  Invalid (other): {validation_stats['invalid_other']}")
    
    if len(valid_data) == 0:
        logger.error("No valid entries found!")
        return []
    
    # Filter by reward
    filtered_data = filter_by_reward(
        valid_data,
        min_percentile=config.min_reward_percentile,
        q_low=config.reward_q_low,
        q_high=config.reward_q_high
    )
    
    if len(filtered_data) == 0:
        logger.error("No entries passed reward filtering!")
        return []
    
    # Statistics
    rewards = [d["reward"] for d in filtered_data]
    logger.info(f"\nFinal Dataset Statistics:")
    logger.info(f"  Number of entries: {len(filtered_data)}")
    logger.info(f"  Mean reward: {np.mean(rewards):.2f}")
    logger.info(f"  Std reward: {np.std(rewards):.2f}")
    logger.info(f"  Min reward: {np.min(rewards):.2f}")
    logger.info(f"  Max reward: {np.max(rewards):.2f}")
    
    # Save
    os.makedirs(os.path.dirname(config.output_json) or ".", exist_ok=True)
    
    with open(config.output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nSaved to: {config.output_json}")
    
    return filtered_data


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare self-distillation data from LLM rollout"
    )
    parser.add_argument(
        "--llm_rollout",
        type=str,
        help="Path to LLM rollout JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for filtered data"
    )
    parser.add_argument(
        "--min_percentile",
        type=float,
        default=0.5,
        help="Minimum reward percentile to keep (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = DistillationConfig()
    
    # Override with CLI args
    if args.llm_rollout:
        config.llm_rollout_json = args.llm_rollout
    if args.output:
        config.output_json = args.output
    if args.min_percentile:
        config.min_reward_percentile = args.min_percentile
    
    logger.info(f"Config: {config}")
    
    # Prepare data
    try:
        data = prepare_distillation_data(config)
        
        if len(data) > 0:
            logger.info("\n" + "="*60)
            logger.info("SUCCESS: Self-distillation data prepared!")
            logger.info("="*60)
            logger.info(f"Next step: Fine-tune with {config.output_json}")
            return 0
        else:
            logger.error("Failed to prepare data")
            return 1
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
