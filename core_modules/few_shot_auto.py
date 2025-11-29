"""
Few-Shot Example Selection - Optimized Version

This module selects high-quality few-shot examples based on:
- Similarity to current state (weighted Euclidean distance)
- Historical reward (higher is better)
- Configurable alpha for balancing both factors

Key Features:
- Weighted feature similarity
- Reward-based filtering
- Automatic prompt injection
- Type-safe implementation
"""

import json
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    """Configuration for few-shot example selection"""
    k: int = 3  # Number of examples to select
    alpha: float = 0.6  # Weight for similarity (1-alpha for reward)
    weights: Optional[List[float]] = None  # Feature weights
    
    def __post_init__(self):
        """Validate configuration"""
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0.0, 1.0]")


def zone_count_from_obs(obs: List[float]) -> int:
    """
    Calculate number of zones from observation vector.
    
    Args:
        obs: Observation vector (3n+2 structure)
        
    Returns:
        Number of zones
    """
    try:
        n = (len(obs) - 2) // 3
        return max(1, n)
    except Exception:
        return 1


def extract_env_terms(obs: List[float], n: int) -> Dict[str, float]:
    """
    Extract environmental terms from observation.
    
    Args:
        obs: Observation vector
        n: Number of zones
        
    Returns:
        Dictionary with environmental features
    """
    def safe_get(idx: int, default: float = 0.0) -> float:
        try:
            return float(obs[idx]) if idx < len(obs) else default
        except (ValueError, TypeError, IndexError):
            return default
    
    outside = safe_get(n)
    
    # GHI average
    ghi_vals = [safe_get(i) for i in range(n + 1, 2 * n + 1)]
    ghi_avg = sum(ghi_vals) / len(ghi_vals) if ghi_vals else 0.0
    
    # Ground temperature
    ground = safe_get(2 * n + 1)
    
    # Occupancy sum
    occ_vals = [safe_get(i) for i in range(2 * n + 2, 3 * n + 2)]
    occ_sum_kw = sum(occ_vals) if occ_vals else 0.0
    
    return {
        "outside": outside,
        "ghi_avg": ghi_avg,
        "ground": ground,
        "occ_sum_kw": occ_sum_kw
    }


def featurize_observation(
    obs: List[float],
    n_override: Optional[int] = None
) -> List[float]:
    """
    Convert observation to feature vector for similarity computation.
    
    Feature order: room_temps + outside_temp + ghi_avg + ground_temp + occ_sum
    
    Args:
        obs: Observation vector
        n_override: Force specific number of zones
        
    Returns:
        Feature vector
    """
    n = n_override if n_override is not None else zone_count_from_obs(obs)
    
    # Extract room temperatures
    temps = [float(x) for x in obs[:n]] if len(obs) >= n else []
    
    # Extract environmental terms
    env = extract_env_terms(obs, n)
    
    # Combine into feature vector
    features = temps + [
        env["outside"],
        env["ghi_avg"],
        env["ground"],
        env["occ_sum_kw"]
    ]
    
    return features


def weighted_euclidean_distance(
    a: List[float],
    b: List[float],
    weights: Optional[List[float]] = None
) -> float:
    """
    Compute weighted Euclidean distance between two feature vectors.
    
    Args:
        a: First feature vector
        b: Second feature vector
        weights: Optional weights for each dimension
        
    Returns:
        Weighted Euclidean distance
    """
    length = min(len(a), len(b))
    
    if weights is None:
        weights = [1.0] * length
    
    sum_sq = 0.0
    for i in range(length):
        w = weights[i] if i < len(weights) else 1.0
        diff = a[i] - b[i]
        sum_sq += w * diff * diff
    
    return math.sqrt(sum_sq)


def compute_similarity(
    feat_a: List[float],
    feat_b: List[float],
    weights: Optional[List[float]] = None
) -> float:
    """
    Compute similarity score (higher is more similar).
    
    Similarity = 1 / (1 + distance)
    
    Args:
        feat_a: First feature vector
        feat_b: Second feature vector
        weights: Optional feature weights
        
    Returns:
        Similarity score in (0, 1]
    """
    distance = weighted_euclidean_distance(feat_a, feat_b, weights)
    return 1.0 / (1.0 + distance)


def load_examples(json_path: str) -> List[Dict]:
    """
    Load few-shot examples from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of example dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is malformed
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of examples")
        
        logger.info(f"Loaded {len(data)} examples from {json_path}")
        return data
        
    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {json_path}: {e}")
        raise ValueError(f"Malformed JSON: {e}")


def select_examples(
    dataset: List[Dict],
    current_obs: List[float],
    config: Optional[SelectionConfig] = None,
    building: Optional[str] = None,
    climate: Optional[str] = None,
    location: Optional[str] = None,
) -> List[Dict]:
    """
    Select k best examples based on similarity and reward.
    
    Scoring: score = alpha * similarity + (1 - alpha) * normalized_reward
    
    Args:
        dataset: List of example dictionaries
        current_obs: Current observation
        config: Selection configuration
        building: Filter by building type
        climate: Filter by climate
        location: Filter by location
        
    Returns:
        List of k selected examples (sorted by score)
    """
    if not dataset:
        logger.warning("Empty dataset")
        return []
    
    # Initialize config
    if config is None:
        config = SelectionConfig()
    
    # Determine zone count and feature weights
    n = zone_count_from_obs(current_obs)
    
    if config.weights is None:
        # Default weights: temps (1.0) > outside (0.5) > ghi (0.2) > ground (0.2) > occ (0.1)
        weights = [1.0] * n + [0.5, 0.2, 0.2, 0.1]
    else:
        weights = config.weights
    
    # Featurize current observation
    current_feat = featurize_observation(current_obs, n_override=n)
    
    # Filter by metadata if specified
    def matches_filters(example: Dict) -> bool:
        if building and example.get("building") != building:
            return False
        if climate and example.get("climate") != climate:
            return False
        if location and example.get("location") != location:
            return False
        return True
    
    pool = [e for e in dataset if matches_filters(e)]
    
    if not pool:
        logger.warning("No examples match filters, using full dataset")
        pool = dataset
    
    logger.info(f"Filtered pool size: {len(pool)} examples")
    
    # Extract rewards for normalization
    rewards = [float(e.get("reward", 0.0)) for e in pool]
    reward_min = min(rewards)
    reward_max = max(rewards)
    reward_span = reward_max - reward_min if reward_max > reward_min else 1.0
    
    # Score each example
    scored = []
    for example in pool:
        # Compute similarity
        example_feat = featurize_observation(example.get("obs", []), n_override=n)
        sim = compute_similarity(current_feat, example_feat, weights)
        
        # Normalize reward to [0, 1]
        reward = float(example.get("reward", 0.0))
        reward_norm = (reward - reward_min) / reward_span
        
        # Combined score
        score = config.alpha * sim + (1.0 - config.alpha) * reward_norm
        
        scored.append((score, example))
    
    # Sort by score (descending) and take top k
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [example for _, example in scored[:config.k]]
    
    logger.info(f"Selected {len(selected)} examples with scores: "
                f"{[round(s, 3) for s, _ in scored[:config.k]]}")
    
    return selected


def format_few_shot_block(
    examples: List[Dict],
    target: float,
    n: int
) -> str:
    """
    Format selected examples into readable text block.
    
    Args:
        examples: List of example dictionaries
        target: Target temperature
        n: Number of zones
        
    Returns:
        Formatted text block
    """
    if not examples:
        return "No examples available."
    
    lines = ["Auto-selected examples (similar & high-reward):"]
    
    for example in examples:
        obs = example.get("obs", [])
        actions = example.get("actions", [])
        reward = example.get("reward", 0.0)
        
        # Format room temperatures
        rooms = [f"{float(x):.1f}" for x in obs[:n]]
        room_str = ", ".join(rooms)
        
        # Format actions
        acts = [f"{float(a):.1f}" for a in actions[:n]] if actions else ["0.0"] * n
        act_str = ", ".join(acts)
        
        # Create example line
        lines.append(
            f"- Example (reward={reward:.3g}): "
            f"Rooms: {room_str}; Target: {target:.1f} -> "
            f"Actions: {act_str}"
        )
    
    return "\n".join(lines)


def inject_few_shot(prompt: str, fewshot_block: str) -> str:
    """
    Inject few-shot examples into prompt at appropriate location.
    
    Searches for "History Action And Feedback Reference:" anchor.
    If not found, appends to end of prompt.
    
    Args:
        prompt: Original prompt
        fewshot_block: Formatted few-shot examples
        
    Returns:
        Prompt with few-shot examples injected
    """
    anchor = "History Action And Feedback Reference:"
    
    # Try to find anchor
    idx = prompt.find(anchor)
    
    if idx == -1:
        # No anchor found, append to end
        return prompt.rstrip() + "\n\n" + fewshot_block.strip() + "\n"
    
    # Find insertion point (after anchor line)
    lines = prompt.splitlines()
    insert_at = len(lines)  # Default to end
    
    for i, line in enumerate(lines):
        if line.strip().startswith(anchor):
            insert_at = i + 2  # Insert after anchor + 1 blank line
            break
    
    # Insert few-shot block
    new_lines = (
        lines[:insert_at] + 
        ["", fewshot_block.strip(), ""] + 
        lines[insert_at:]
    )
    
    return "\n".join(new_lines)


# ============================================================================
# Testing and Validation
# ============================================================================

def validate_examples(examples: List[Dict]) -> Tuple[bool, str]:
    """
    Validate structure of example dictionaries.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not examples:
        return False, "Empty example list"
    
    required_fields = ["obs", "actions", "reward"]
    
    for i, ex in enumerate(examples):
        # Check required fields
        for field in required_fields:
            if field not in ex:
                return False, f"Example {i} missing field: {field}"
        
        # Check obs is list
        if not isinstance(ex["obs"], list):
            return False, f"Example {i}: obs must be a list"
        
        # Check actions is list
        if not isinstance(ex["actions"], list):
            return False, f"Example {i}: actions must be a list"
        
        # Check reward is numeric
        try:
            float(ex["reward"])
        except (ValueError, TypeError):
            return False, f"Example {i}: reward must be numeric"
    
    return True, "Valid"


if __name__ == "__main__":
    # Test with sample data
    sample_obs = [22.5, 23.0, 21.8, 30.0, 500, 450, 480, 20.0, 0.5, 0.3, 0.4]
    
    sample_dataset = [
        {
            "obs": [22.0, 22.5, 22.3, 30.0, 500, 480, 490, 20.0, 0.4, 0.3, 0.3],
            "actions": [0.1, -0.2, 0.0],
            "reward": -5.2,
            "building": "OfficeSmall",
            "climate": "Hot_Dry"
        },
        {
            "obs": [25.0, 24.5, 24.8, 32.0, 550, 530, 540, 21.0, 0.6, 0.5, 0.5],
            "actions": [-0.5, -0.4, -0.6],
            "reward": -12.5,
            "building": "OfficeSmall",
            "climate": "Hot_Dry"
        }
    ]
    
    print("Testing few-shot selection:")
    print(f"Sample observation: {sample_obs[:5]}...")
    print(f"Dataset size: {len(sample_dataset)}")
    
    valid, msg = validate_examples(sample_dataset)
    print(f"Validation: {valid} - {msg}")
    
    if valid:
        selected = select_examples(
            sample_dataset,
            sample_obs,
            config=SelectionConfig(k=2, alpha=0.6)
        )
        print(f"\nSelected {len(selected)} examples")
        
        block = format_few_shot_block(selected, target=22.0, n=3)
        print(f"\nFormatted block:\n{block}")
