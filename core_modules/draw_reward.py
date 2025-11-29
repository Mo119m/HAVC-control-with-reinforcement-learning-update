"""
Reward Visualization - Optimized Version

This module plots reward curves from trajectory files for performance comparison.

Key Features:
- Multi-file comparison
- Configurable plot parameters
- Statistical analysis
- Clean visualization
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.size'] = 10


def load_trajectory_rewards(
    path: str
) -> Tuple[List[int], List[float], str]:
    """
    Load rewards from trajectory JSON file.
    
    Args:
        path: Path to trajectory JSON
        
    Returns:
        Tuple of (steps, rewards, filename)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    logger.info(f"Loading: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    
    # Extract step and reward
    steps = []
    rewards = []
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        
        step = item.get("step", i)
        reward = item.get("reward")
        
        if reward is not None:
            steps.append(step)
            rewards.append(float(reward))
    
    filename = os.path.basename(path)
    
    logger.info(f"Loaded {len(rewards)} rewards from {filename}")
    
    return steps, rewards, filename


def create_comparison_dataframe(
    paths: List[str],
    first_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Create DataFrame for multi-file comparison.
    
    Args:
        paths: List of trajectory file paths
        first_n: Limit to first N steps (None for all)
        
    Returns:
        DataFrame with columns: file, step, reward
    """
    records = []
    
    for path in paths:
        try:
            steps, rewards, filename = load_trajectory_rewards(path)
            
            # Limit if requested
            if first_n is not None:
                steps = steps[:first_n]
                rewards = rewards[:first_n]
            
            # Add to records
            for step, reward in zip(steps, rewards):
                records.append({
                    "file": filename,
                    "step": step,
                    "reward": reward
                })
        
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
    
    return pd.DataFrame(records)


def plot_rewards(
    df: pd.DataFrame,
    output_path: str = "rewards_plot.png",
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    title: str = "Reward Comparison",
    smooth_window: Optional[int] = None,
    show_stats: bool = True
) -> str:
    """
    Plot reward curves with optional smoothing.
    
    Args:
        df: DataFrame with file, step, reward columns
        output_path: Output file path
        y_min: Y-axis minimum
        y_max: Y-axis maximum
        title: Plot title
        smooth_window: Moving average window (None to disable)
        show_stats: Show mean/std as shaded area
        
    Returns:
        Path to saved plot
    """
    if df.empty:
        raise ValueError("Empty DataFrame")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each file
    for filename, group in df.groupby("file"):
        group_sorted = group.sort_values("step")
        steps = group_sorted["step"].values
        rewards = group_sorted["reward"].values
        
        # Apply smoothing if requested
        if smooth_window and smooth_window > 1:
            rewards_smooth = pd.Series(rewards).rolling(
                window=smooth_window, 
                center=True, 
                min_periods=1
            ).mean().values
            
            # Plot both raw (transparent) and smooth
            ax.plot(
                steps, 
                rewards, 
                alpha=0.2, 
                linewidth=0.5
            )
            ax.plot(
                steps, 
                rewards_smooth, 
                label=filename, 
                marker="o", 
                markersize=3, 
                linewidth=1.5
            )
        else:
            # Plot raw
            ax.plot(
                steps, 
                rewards, 
                label=filename, 
                marker="o", 
                markersize=3, 
                linewidth=1.5
            )
        
        # Add statistics overlay
        if show_stats and len(rewards) > 1:
            mean = np.mean(rewards)
            std = np.std(rewards)
            ax.axhline(
                mean, 
                linestyle="--", 
                alpha=0.3, 
                linewidth=1
            )
            ax.fill_between(
                [steps.min(), steps.max()],
                mean - std,
                mean + std,
                alpha=0.1
            )
    
    # Formatting
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    
    # Set y-limits if specified
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved plot to {output_path}")
    
    return output_path


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each file.
    
    Args:
        df: DataFrame with file, step, reward columns
        
    Returns:
        DataFrame with statistics
    """
    stats = df.groupby("file")["reward"].agg([
        ("count", "count"),
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("q25", lambda x: x.quantile(0.25)),
        ("median", "median"),
        ("q75", lambda x: x.quantile(0.75)),
    ]).round(3)
    
    return stats


def extract_and_plot_rewards(
    paths: List[str],
    first_n: Optional[int] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    output_png: str = "rewards_plot.png",
    smooth_window: Optional[int] = None,
    return_dataframe: bool = False,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Main function: load, plot, and analyze rewards.
    
    Args:
        paths: List of trajectory JSON paths
        first_n: Limit to first N steps
        y_min: Y-axis minimum
        y_max: Y-axis maximum
        output_png: Output file path
        smooth_window: Moving average window
        return_dataframe: Whether to return DataFrame
        
    Returns:
        Tuple of (plot_path, dataframe)
    """
    # Load data
    df = create_comparison_dataframe(paths, first_n=first_n)
    
    if df.empty:
        raise ValueError("No data loaded")
    
    logger.info(f"Total records: {len(df)}")
    
    # Plot
    plot_path = plot_rewards(
        df,
        output_path=output_png,
        y_min=y_min,
        y_max=y_max,
        smooth_window=smooth_window
    )
    
    # Compute statistics
    stats = compute_statistics(df)
    print("\n" + "=" * 50)
    print("Reward Statistics")
    print("=" * 50)
    print(stats)
    print("=" * 50)
    
    if return_dataframe:
        return plot_path, df
    
    return plot_path, None


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot reward curves from trajectory files"
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="Trajectory JSON files"
    )
    parser.add_argument(
        "--first_n",
        type=int,
        help="Limit to first N steps"
    )
    parser.add_argument(
        "--y_min",
        type=float,
        help="Y-axis minimum"
    )
    parser.add_argument(
        "--y_max",
        type=float,
        help="Y-axis maximum"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rewards_plot.png",
        help="Output PNG file"
    )
    parser.add_argument(
        "--smooth",
        type=int,
        help="Moving average window"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Reward Comparison",
        help="Plot title"
    )
    
    args = parser.parse_args()
    
    try:
        plot_path, _ = extract_and_plot_rewards(
            paths=args.files,
            first_n=args.first_n,
            y_min=args.y_min,
            y_max=args.y_max,
            output_png=args.output,
            smooth_window=args.smooth,
        )
        print(f"\nPlot saved to: {plot_path}")
    
    except Exception as e:
        logger.error(f"Plotting failed: {e}", exc_info=True)
        exit(1)
