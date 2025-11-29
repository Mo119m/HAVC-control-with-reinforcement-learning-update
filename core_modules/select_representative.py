"""
Representative Sample Selection - Optimized Version

This module selects high-quality, diverse samples for few-shot learning:
1. Pre-filter by reward (top percentile)
2. Cluster by room temperatures for diversity
3. Select top-N from each cluster

Key Features:
- Reward-based filtering
- KMeans clustering for diversity
- Configurable selection parameters
- Clean output format
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    """Configuration for sample selection"""
    preselect: int = 2000  # Pre-filter top N by reward
    clusters: int = 12  # Number of clusters
    n_per_cluster: int = 20  # Samples per cluster
    min_reward_percentile: float = 0.5  # Minimum reward percentile (0-1)


def obs_split(obs: List[float]) -> Tuple[int, List[float]]:
    """
    Split observation into zones and extract room temperatures.
    
    Args:
        obs: Observation vector (3n+2 structure)
        
    Returns:
        Tuple of (n_zones, room_temperatures)
    """
    n = max(1, (len(obs) - 2) // 3)
    temps = list(obs[:n]) if len(obs) >= n else []
    return n, temps


def validate_action(
    action: Any,
    n_zones: int,
    min_val: float = -1.0,
    max_val: float = 1.0,
    tolerance: float = 0.05
) -> Tuple[bool, Optional[List[float]]]:
    """
    Validate and normalize action to [-1, 1] range.
    
    Args:
        action: Action to validate (can be scalar or list)
        n_zones: Expected number of zones
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        tolerance: Tolerance for slightly out-of-range values
        
    Returns:
        Tuple of (is_valid, normalized_action)
    """
    # Convert to list
    if not isinstance(action, list):
        action_list = [action]
    else:
        action_list = list(action)
    
    # Handle single value broadcast
    if len(action_list) == 1 and n_zones > 1:
        action_list = action_list * n_zones
    
    # Check length
    if len(action_list) != n_zones:
        logger.debug(f"Action length {len(action_list)} != n_zones {n_zones}")
        return False, None
    
    # Validate and clip values
    try:
        action_float = [float(a) for a in action_list]
        
        # Check if values are reasonable
        for a in action_float:
            if abs(a) > (max_val + tolerance):
                logger.debug(f"Action value {a} exceeds bounds")
                return False, None
        
        # Clip to valid range
        action_clipped = [
            max(min_val, min(max_val, a)) 
            for a in action_float
        ]
        
        return True, action_clipped
    
    except (ValueError, TypeError) as e:
        logger.debug(f"Action validation error: {e}")
        return False, None


def create_clustering_features(
    temps_list: List[List[float]]
) -> np.ndarray:
    """
    Create normalized feature matrix for clustering.
    
    Only uses room temperatures, standardized to zero mean and unit variance.
    
    Args:
        temps_list: List of room temperature vectors
        
    Returns:
        Standardized feature matrix (n_samples, n_features)
    """
    if not temps_list:
        raise ValueError("Empty temperatures list")
    
    # Find minimum length (handle variable-length observations)
    n_min = min(len(t) for t in temps_list)
    
    if n_min == 0:
        raise ValueError("All temperature vectors are empty")
    
    # Stack and truncate to minimum length
    temps_array = np.stack([
        np.asarray(t[:n_min], dtype=np.float32) 
        for t in temps_list
    ], axis=0)
    
    # Standardize (z-score normalization)
    mean = temps_array.mean(axis=0, keepdims=True)
    std = temps_array.std(axis=0, keepdims=True) + 1e-6
    
    normalized = (temps_array - mean) / std
    
    logger.debug(f"Created feature matrix: {normalized.shape}")
    
    return normalized


def load_and_filter_trajectory(
    traj_path: Path,
    config: SelectionConfig
) -> List[Tuple[List[float], List[float], List[float], float]]:
    """
    Load trajectory and filter valid samples.
    
    Args:
        traj_path: Path to trajectory JSON
        config: Selection configuration
        
    Returns:
        List of tuples (obs, temps, action, reward)
    """
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    
    logger.info(f"Loading trajectory from {traj_path}")
    
    with open(traj_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)
    
    if not trajectory:
        raise ValueError("Empty trajectory")
    
    logger.info(f"Loaded {len(trajectory)} transitions")
    
    # Extract valid candidates
    candidates = []
    
    for i, step in enumerate(trajectory):
        # Extract fields
        obs = step.get("obs") or step.get("obs_before")
        action = step.get("action")
        reward = step.get("reward", 0.0)
        
        if obs is None or action is None:
            continue
        
        # Parse observation
        n_zones, temps = obs_split(obs)
        
        # Validate action
        is_valid, action_normalized = validate_action(action, n_zones)
        
        if not is_valid:
            continue
        
        # Apply minimum reward filter
        if reward < config.min_reward_percentile:
            continue
        
        candidates.append((obs, temps, action_normalized, float(reward)))
    
    logger.info(f"Found {len(candidates)} valid candidates")
    
    if not candidates:
        raise ValueError("No valid candidates after filtering")
    
    return candidates


def select_representative_samples(
    candidates: List[Tuple[List[float], List[float], List[float], float]],
    config: SelectionConfig
) -> List[Tuple[List[float], List[float], List[float], float]]:
    """
    Select representative samples using reward + clustering.
    
    Args:
        candidates: List of (obs, temps, action, reward) tuples
        config: Selection configuration
        
    Returns:
        Selected samples
    """
    # Step 1: Sort by reward and take top preselect
    candidates_sorted = sorted(
        candidates, 
        key=lambda x: x[3],  # reward
        reverse=True
    )
    
    pool = candidates_sorted[:min(len(candidates_sorted), config.preselect)]
    logger.info(f"Pre-selected top {len(pool)} by reward")
    
    # Step 2: Cluster by temperatures
    temps_list = [sample[1] for sample in pool]
    features = create_clustering_features(temps_list)
    
    n_clusters = min(config.clusters, len(pool))
    
    if n_clusters <= 1:
        logger.warning("Only 1 cluster, selecting top N samples")
        return pool[:config.n_per_cluster]
    
    logger.info(f"Clustering into {n_clusters} clusters")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(features)
    
    # Step 3: Select top N from each cluster
    selected = []
    
    for cluster_id in range(n_clusters):
        # Get samples in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        cluster_samples = [pool[i] for i in cluster_indices]
        
        # Sort by reward within cluster
        cluster_samples_sorted = sorted(
            cluster_samples,
            key=lambda x: x[3],
            reverse=True
        )
        
        # Take top N
        selected.extend(
            cluster_samples_sorted[:config.n_per_cluster]
        )
    
    logger.info(f"Selected {len(selected)} samples total")
    
    return selected


def format_output(
    samples: List[Tuple[List[float], List[float], List[float], float]],
    building: str,
    climate: str,
    location: str,
    decimals: int = 1
) -> List[Dict[str, Any]]:
    """
    Format selected samples for output.
    
    Args:
        samples: List of (obs, temps, action, reward) tuples
        building: Building type
        climate: Climate type
        location: Location name
        decimals: Decimal places for actions
        
    Returns:
        List of formatted dictionaries
    """
    formatted = []
    
    for obs, _, action, reward in samples:
        formatted.append({
            "obs": obs,
            "actions": [
                round(float(a), decimals) 
                for a in action
            ],
            "reward": float(reward),
            "building": building,
            "climate": climate,
            "location": location,
        })
    
    return formatted


def main(args: argparse.Namespace) -> None:
    """Main selection pipeline"""
    
    # Initialize configuration
    config = SelectionConfig(
        preselect=args.preselect,
        clusters=args.clusters,
        n_per_cluster=args.n_per_cluster,
        min_reward_percentile=args.min_reward_percentile
    )
    
    logger.info(f"Selection config: {config}")
    
    # Load and filter trajectory
    traj_path = Path(args.traj)
    candidates = load_and_filter_trajectory(traj_path, config)
    
    # Select representative samples
    selected = select_representative_samples(candidates, config)
    
    # Format output
    output = format_output(
        selected,
        building=args.building,
        climate=args.climate,
        location=args.location,
        decimals=args.decimals
    )
    
    # Save to file
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "few_shot_examples_structured.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(output)} examples to {out_path}")
    
    # Print statistics
    rewards = [s["reward"] for s in output]
    print("\n" + "=" * 50)
    print("Selection Summary")
    print("=" * 50)
    print(f"Total candidates: {len(candidates)}")
    print(f"Pre-selected pool: {config.preselect}")
    print(f"Clusters: {config.clusters}")
    print(f"Per cluster: {config.n_per_cluster}")
    print(f"Final output: {len(output)}")
    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(rewards):.3f}")
    print(f"  Std:  {np.std(rewards):.3f}")
    print(f"  Min:  {np.min(rewards):.3f}")
    print(f"  Max:  {np.max(rewards):.3f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select representative samples for few-shot learning"
    )
    
    # Input/output
    parser.add_argument(
        "--traj",
        type=str,
        default="runs_officesmall_hotdry/ppo_trajectory.json",
        help="Path to trajectory JSON"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="fs_out",
        help="Output directory"
    )
    
    # Selection parameters
    parser.add_argument(
        "--preselect",
        type=int,
        default=2000,
        help="Pre-select top N by reward"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=12,
        help="Number of clusters"
    )
    parser.add_argument(
        "--n_per_cluster",
        type=int,
        default=20,
        help="Samples per cluster"
    )
    parser.add_argument(
        "--min_reward_percentile",
        type=float,
        default=0.5,
        help="Minimum reward percentile (0-1)"
    )
    
    # Metadata
    parser.add_argument(
        "--building",
        type=str,
        default="OfficeSmall",
        help="Building type"
    )
    parser.add_argument(
        "--climate",
        type=str,
        default="Hot_Dry",
        help="Climate type"
    )
    parser.add_argument(
        "--location",
        type=str,
        default="Tucson",
        help="Location name"
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Decimal places for actions"
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logger.error(f"Selection failed: {e}", exc_info=True)
        exit(1)
