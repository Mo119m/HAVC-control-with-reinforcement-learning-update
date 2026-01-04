"""
Pipeline Visualization Module - Complete Analysis and Reporting

This module creates comprehensive visualizations for each pipeline stage:
1. PPO Training Analysis
2. Few-shot Selection Analysis
3. LLM Rollout Analysis
4. Self-Distillation Data Analysis
5. Fine-tuning Progress
6. Final Comparison Report

Usage:
    python visualization_pipeline.py --pipeline_dir ./pipeline_output --output_dir ./reports
"""

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class StageResults:
    """Container for stage results"""
    name: str
    trajectory_path: Optional[str] = None
    metrics_path: Optional[str] = None
    model_path: Optional[str] = None
    plots_dir: Optional[str] = None


class PipelineVisualizer:
    """Complete pipeline visualization system"""

    def __init__(self, pipeline_dir: str, output_dir: str = None):
        """
        Initialize visualizer.

        Args:
            pipeline_dir: Root directory of pipeline output
            output_dir: Directory to save visualization reports
        """
        self.pipeline_dir = Path(pipeline_dir)
        self.output_dir = Path(output_dir) if output_dir else self.pipeline_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage directories
        self.stages = {
            "ppo": self.pipeline_dir / "01_ppo_training",
            "fewshot": self.pipeline_dir / "02_few_shot_samples",
            "rollout": self.pipeline_dir / "03_llm_rollout",
            "finetune": self.pipeline_dir / "04_finetuning",
            "eval": self.pipeline_dir / "05_evaluation",
        }

        logger.info(f"Pipeline dir: {self.pipeline_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def load_trajectory(self, path: Path) -> List[Dict]:
        """Load trajectory JSON file"""
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []

        with open(path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.warning(f"Expected list, got {type(data)}")
            return []

        return data

    def load_metrics(self, path: Path) -> Dict:
        """Load metrics JSON file"""
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return {}

        with open(path, 'r') as f:
            return json.load(f)

    def extract_rewards(self, trajectory: List[Dict]) -> np.ndarray:
        """Extract rewards from trajectory"""
        return np.array([step.get('reward', 0) for step in trajectory])

    # ========== Stage 1: PPO Training ==========

    def visualize_ppo_training(self, save: bool = True) -> Optional[str]:
        """
        Create comprehensive PPO training analysis.

        Returns:
            Path to saved plot
        """
        logger.info("Visualizing PPO training...")

        stage_dir = self.stages["ppo"]
        traj_path = stage_dir / "ppo_trajectory.json"
        metrics_path = stage_dir / "training_metrics.json"

        if not traj_path.exists():
            logger.error("PPO trajectory not found")
            return None

        # Load data
        trajectory = self.load_trajectory(traj_path)
        metrics = self.load_metrics(metrics_path)
        rewards = self.extract_rewards(trajectory)

        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Reward curve (cumulative)
        ax1 = fig.add_subplot(gs[0, :2])
        cumulative = np.cumsum(rewards)
        ax1.plot(cumulative, linewidth=2, color='#2ecc71', alpha=0.8)
        ax1.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='#2ecc71')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('PPO Training - Cumulative Reward', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Reward per step
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(rewards, linewidth=1, color='#3498db', alpha=0.6)
        # Moving average
        if len(rewards) > 50:
            window = min(50, len(rewards) // 10)
            ma = pd.Series(rewards).rolling(window=window).mean()
            ax2.plot(ma, linewidth=2, color='#e74c3c', label=f'MA({window})')
            ax2.legend()
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.set_title('Step Reward', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Reward distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(rewards, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(rewards), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax3.axvline(np.median(rewards), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Rolling statistics
        ax4 = fig.add_subplot(gs[1, 1])
        if len(rewards) > 100:
            window = min(100, len(rewards) // 5)
            rolling_mean = pd.Series(rewards).rolling(window=window).mean()
            rolling_std = pd.Series(rewards).rolling(window=window).std()

            ax4.plot(rolling_mean, label='Mean', linewidth=2, color='#e74c3c')
            ax4.fill_between(
                range(len(rolling_mean)),
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.3,
                color='#e74c3c',
                label='±1 Std'
            )
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Reward')
            ax4.set_title(f'Rolling Statistics (window={window})', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Training metrics (if available)
        if metrics and 'ep_rew_mean' in metrics:
            ax5 = fig.add_subplot(gs[1, 2])
            timesteps = metrics.get('timesteps', [])
            ep_rew = metrics.get('ep_rew_mean', [])
            if timesteps and ep_rew:
                ax5.plot(timesteps, ep_rew, linewidth=2, color='#1abc9c', marker='o', markersize=4)
                ax5.set_xlabel('Timesteps')
                ax5.set_ylabel('Episode Reward Mean')
                ax5.set_title('Training Progress', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)

        # 6. Statistics table
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        stats_text = f"""
        PPO Training Statistics:

        Total Steps: {len(rewards):,}
        Total Reward: {np.sum(rewards):,.2f}

        Reward Statistics:
        - Mean: {np.mean(rewards):.2f}
        - Median: {np.median(rewards):.2f}
        - Std: {np.std(rewards):.2f}
        - Min: {np.min(rewards):.2f}
        - Max: {np.max(rewards):.2f}
        - Q1: {np.percentile(rewards, 25):.2f}
        - Q3: {np.percentile(rewards, 75):.2f}

        Performance:
        - Best Episode: {np.max(rewards):.2f} (step {np.argmax(rewards)})
        - Worst Episode: {np.min(rewards):.2f} (step {np.argmin(rewards)})
        - Final 100 steps avg: {np.mean(rewards[-100:]):.2f}
        """

        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Stage 1: PPO Training Analysis', fontsize=16, fontweight='bold', y=0.995)

        if save:
            output_path = self.output_dir / "01_ppo_training_analysis.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close()
            return str(output_path)

        return None

    # ========== Stage 2: Few-shot Selection ==========

    def visualize_fewshot_selection(self, save: bool = True) -> Optional[str]:
        """
        Visualize few-shot sample selection analysis.

        Returns:
            Path to saved plot
        """
        logger.info("Visualizing few-shot selection...")

        stage_dir = self.stages["fewshot"]
        fewshot_path = stage_dir / "few_shot_examples_structured.json"

        if not fewshot_path.exists():
            logger.error("Few-shot examples not found")
            return None

        # Load data
        with open(fewshot_path, 'r') as f:
            fewshot_data = json.load(f)

        if not isinstance(fewshot_data, list):
            logger.warning("Unexpected fewshot data format")
            return None

        # Extract rewards
        rewards = []
        for example in fewshot_data:
            if isinstance(example, dict):
                reward = example.get('reward') or example.get('total_reward')
                if reward is not None:
                    rewards.append(float(reward))

        rewards = np.array(rewards)

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Stage 2: Few-Shot Selection Analysis', fontsize=16, fontweight='bold')

        # 1. Reward distribution
        axes[0, 0].hist(rewards, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(rewards), color='r', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(rewards):.2f}')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Selected Examples - Reward Distribution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plot
        axes[0, 1].boxplot(rewards, vert=True)
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Reward Statistics (Box Plot)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Sorted rewards
        sorted_rewards = np.sort(rewards)
        axes[1, 0].plot(sorted_rewards, linewidth=2, color='#2ecc71')
        axes[1, 0].fill_between(range(len(sorted_rewards)), sorted_rewards, alpha=0.3, color='#2ecc71')
        axes[1, 0].set_xlabel('Example Index (sorted)')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Sorted Rewards', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Statistics
        axes[1, 1].axis('off')
        stats_text = f"""
        Few-Shot Selection Statistics:

        Total Examples: {len(rewards)}

        Reward Statistics:
        - Mean: {np.mean(rewards):.2f}
        - Median: {np.median(rewards):.2f}
        - Std: {np.std(rewards):.2f}
        - Min: {np.min(rewards):.2f}
        - Max: {np.max(rewards):.2f}

        Quality Metrics:
        - Top 10%: {np.percentile(rewards, 90):.2f}
        - Bottom 10%: {np.percentile(rewards, 10):.2f}
        - IQR: {np.percentile(rewards, 75) - np.percentile(rewards, 25):.2f}
        """

        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center', transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()

        if save:
            output_path = self.output_dir / "02_fewshot_selection_analysis.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close()
            return str(output_path)

        return None

    # ========== Stage 3: LLM Rollout ==========

    def visualize_llm_rollout(self, save: bool = True) -> Optional[str]:
        """
        Visualize LLM rollout analysis.

        Returns:
            Path to saved plot
        """
        logger.info("Visualizing LLM rollout...")

        stage_dir = self.stages["rollout"]
        rollout_path = stage_dir / "llm_rollout.json"

        if not rollout_path.exists():
            logger.error("LLM rollout not found")
            return None

        # Load data
        trajectory = self.load_trajectory(rollout_path)
        rewards = self.extract_rewards(trajectory)

        # Analyze parsing success
        parse_success = []
        parse_methods = []
        for step in trajectory:
            if isinstance(step, dict):
                parsed_from = step.get('parsed_from', 'unknown')
                parse_success.append(parsed_from != 'failed')
                parse_methods.append(parsed_from)

        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Cumulative reward
        ax1 = fig.add_subplot(gs[0, :2])
        cumulative = np.cumsum(rewards)
        ax1.plot(cumulative, linewidth=2, color='#e74c3c', alpha=0.8)
        ax1.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='#e74c3c')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('LLM Rollout - Cumulative Reward', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Step rewards
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(rewards, linewidth=1, color='#9b59b6', alpha=0.6)
        if len(rewards) > 20:
            ma = pd.Series(rewards).rolling(window=20).mean()
            ax2.plot(ma, linewidth=2, color='#e74c3c', label='MA(20)')
            ax2.legend()
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.set_title('Step Reward', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Parsing success rate
        ax3 = fig.add_subplot(gs[1, 0])
        success_rate = np.mean(parse_success) * 100
        ax3.bar(['Success', 'Failed'],
               [sum(parse_success), len(parse_success) - sum(parse_success)],
               color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Count')
        ax3.set_title(f'Action Parsing ({success_rate:.1f}% success)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Parsing methods
        ax4 = fig.add_subplot(gs[1, 1])
        from collections import Counter
        method_counts = Counter(parse_methods)
        methods = list(method_counts.keys())
        counts = list(method_counts.values())
        ax4.barh(methods, counts, color='#3498db', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Count')
        ax4.set_title('Parsing Methods Used', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 5. Reward distribution
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.hist(rewards, bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
        ax5.axvline(np.mean(rewards), color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(rewards):.2f}')
        ax5.set_xlabel('Reward')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Statistics
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')

        stats_text = f"""
        LLM Rollout Statistics:

        Total Steps: {len(rewards):,}
        Total Reward: {np.sum(rewards):,.2f}

        Reward Statistics:
        - Mean: {np.mean(rewards):.2f}
        - Median: {np.median(rewards):.2f}
        - Std: {np.std(rewards):.2f}
        - Min: {np.min(rewards):.2f}
        - Max: {np.max(rewards):.2f}

        Parsing Success:
        - Success Rate: {success_rate:.1f}%
        - Successful: {sum(parse_success)} / {len(parse_success)}
        - Failed: {len(parse_success) - sum(parse_success)}
        """

        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.suptitle('Stage 3: LLM Rollout Analysis (Before Fine-tuning)',
                    fontsize=16, fontweight='bold', y=0.995)

        if save:
            output_path = self.output_dir / "03_llm_rollout_analysis.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close()
            return str(output_path)

        return None

    # ========== Stage 4: Self-Distillation ==========

    def visualize_self_distillation(self, save: bool = True) -> Optional[str]:
        """
        Visualize self-distillation data filtering analysis.

        Returns:
            Path to saved plot
        """
        logger.info("Visualizing self-distillation...")

        stage_dir = self.stages["rollout"]
        original_path = stage_dir / "llm_rollout.json"
        distilled_path = stage_dir / "distillation_data.json"

        if not original_path.exists() or not distilled_path.exists():
            logger.error("Distillation data not found")
            return None

        # Load data
        original_traj = self.load_trajectory(original_path)
        distilled_traj = self.load_trajectory(distilled_path)

        original_rewards = self.extract_rewards(original_traj)
        distilled_rewards = self.extract_rewards(distilled_traj)

        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Before vs After comparison
        ax1 = fig.add_subplot(gs[0, :])
        x1 = np.arange(len(original_rewards))
        x2 = np.arange(len(distilled_rewards))

        ax1.scatter(x1, original_rewards, alpha=0.3, s=20, color='gray', label='Original (all data)')
        ax1.scatter(x2, distilled_rewards, alpha=0.6, s=30, color='#2ecc71',
                   label='After Distillation (filtered)')
        ax1.axhline(np.mean(original_rewards), color='gray', linestyle='--',
                   linewidth=2, label=f'Original Mean: {np.mean(original_rewards):.2f}')
        ax1.axhline(np.mean(distilled_rewards), color='#2ecc71', linestyle='--',
                   linewidth=2, label=f'Distilled Mean: {np.mean(distilled_rewards):.2f}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('Self-Distillation: Data Filtering Effect', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. Distribution comparison
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist([original_rewards, distilled_rewards], bins=40,
                label=['Original', 'Distilled'], color=['gray', '#2ecc71'],
                alpha=0.6, edgecolor='black')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution Comparison', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Box plot comparison
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.boxplot([original_rewards, distilled_rewards], labels=['Original', 'Distilled'])
        ax3.set_ylabel('Reward')
        ax3.set_title('Statistical Comparison', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative rewards
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(np.cumsum(original_rewards), linewidth=2, color='gray',
                label='Original', alpha=0.7)
        ax4.plot(np.cumsum(distilled_rewards), linewidth=2, color='#2ecc71',
                label='Distilled')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Cumulative Reward')
        ax4.set_title('Cumulative Performance', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Statistics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        retention_rate = len(distilled_rewards) / len(original_rewards) * 100
        improvement = (np.mean(distilled_rewards) - np.mean(original_rewards)) / abs(np.mean(original_rewards)) * 100

        stats_text = f"""
        Self-Distillation Statistics:

        Data Filtering:
        - Original Data: {len(original_rewards)} steps
        - After Filtering: {len(distilled_rewards)} steps
        - Retention Rate: {retention_rate:.1f}%
        - Removed: {len(original_rewards) - len(distilled_rewards)} steps ({100-retention_rate:.1f}%)

        Quality Improvement:
        - Original Mean Reward: {np.mean(original_rewards):.2f}
        - Distilled Mean Reward: {np.mean(distilled_rewards):.2f}
        - Improvement: {improvement:+.1f}%

        Original Data:                          Distilled Data:
        - Median: {np.median(original_rewards):.2f}                          - Median: {np.median(distilled_rewards):.2f}
        - Std: {np.std(original_rewards):.2f}                             - Std: {np.std(distilled_rewards):.2f}
        - Range: [{np.min(original_rewards):.2f}, {np.max(original_rewards):.2f}]              - Range: [{np.min(distilled_rewards):.2f}, {np.max(distilled_rewards):.2f}]
        """

        ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.suptitle('Stage 4: Self-Distillation Analysis', fontsize=16, fontweight='bold', y=0.995)

        if save:
            output_path = self.output_dir / "04_self_distillation_analysis.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close()
            return str(output_path)

        return None

    # ========== Stage 5: Fine-tuning ==========

    def visualize_finetuning(self, save: bool = True) -> Optional[str]:
        """
        Visualize fine-tuning progress.

        Note: This assumes the fine-tuning script saves training logs

        Returns:
            Path to saved plot
        """
        logger.info("Visualizing fine-tuning...")

        stage_dir = self.stages["finetune"]

        # Check for existing plot
        existing_plot = stage_dir / "finetune_training_curves.png"
        if existing_plot.exists():
            logger.info(f"Fine-tuning plot already exists: {existing_plot}")
            # Copy to reports
            if save:
                import shutil
                output_path = self.output_dir / "05_finetuning_progress.png"
                shutil.copy(existing_plot, output_path)
                logger.info(f"Copied to: {output_path}")
                return str(output_path)

        logger.warning("Fine-tuning plot not found - will be generated during training")
        return None

    # ========== Stage 6: Final Comparison ==========

    def visualize_final_comparison(self, save: bool = True) -> Optional[str]:
        """
        Create comprehensive final comparison across all stages.

        Returns:
            Path to saved plot
        """
        logger.info("Creating final comparison...")

        # Load all trajectories
        trajectories = {}

        ppo_path = self.stages["ppo"] / "ppo_trajectory.json"
        if ppo_path.exists():
            trajectories['PPO (Expert)'] = self.load_trajectory(ppo_path)

        llm_path = self.stages["rollout"] / "llm_rollout.json"
        if llm_path.exists():
            trajectories['LLM (Before FT)'] = self.load_trajectory(llm_path)

        ft_path = self.stages["eval"] / "finetuned_rollout.json"
        if ft_path.exists():
            trajectories['LLM (After FT)'] = self.load_trajectory(ft_path)

        if len(trajectories) < 2:
            logger.error("Not enough trajectories for comparison")
            return None

        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        colors = {'PPO (Expert)': '#2ecc71',
                 'LLM (Before FT)': '#e74c3c',
                 'LLM (After FT)': '#3498db'}

        # Extract rewards
        all_rewards = {}
        for name, traj in trajectories.items():
            all_rewards[name] = self.extract_rewards(traj)

        # 1. Cumulative rewards comparison
        ax1 = fig.add_subplot(gs[0, :])
        for name, rewards in all_rewards.items():
            cumulative = np.cumsum(rewards)
            ax1.plot(cumulative, linewidth=2.5, label=name, color=colors.get(name, 'gray'), alpha=0.8)
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('Cumulative Reward', fontsize=12)
        ax1.set_title('Cumulative Reward Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # 2. Step rewards comparison (first 200 steps)
        ax2 = fig.add_subplot(gs[1, 0])
        max_len = min(200, min(len(r) for r in all_rewards.values()))
        for name, rewards in all_rewards.items():
            ax2.plot(rewards[:max_len], linewidth=1.5, label=name,
                    color=colors.get(name, 'gray'), alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.set_title(f'Step Reward (First {max_len} steps)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Distribution comparison
        ax3 = fig.add_subplot(gs[1, 1])
        data_for_boxplot = [rewards for rewards in all_rewards.values()]
        labels_for_boxplot = list(all_rewards.keys())
        bp = ax3.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
        for patch, name in zip(bp['boxes'], labels_for_boxplot):
            patch.set_facecolor(colors.get(name, 'gray'))
            patch.set_alpha(0.7)
        ax3.set_ylabel('Reward')
        ax3.set_title('Reward Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')

        # 4. Mean reward comparison (bar chart)
        ax4 = fig.add_subplot(gs[1, 2])
        means = [np.mean(rewards) for rewards in all_rewards.values()]
        bars = ax4.bar(range(len(means)), means,
                      color=[colors.get(name, 'gray') for name in all_rewards.keys()],
                      alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_xticks(range(len(means)))
        ax4.set_xticklabels(list(all_rewards.keys()), rotation=15, ha='right')
        ax4.set_ylabel('Mean Reward')
        ax4.set_title('Average Performance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 5. Rolling average comparison
        ax5 = fig.add_subplot(gs[2, :2])
        window = 20
        for name, rewards in all_rewards.items():
            if len(rewards) > window:
                ma = pd.Series(rewards).rolling(window=window).mean()
                ax5.plot(ma, linewidth=2, label=name, color=colors.get(name, 'gray'))
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Reward')
        ax5.set_title(f'Moving Average Comparison (window={window})', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Statistics table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')

        # Calculate improvements
        stats_lines = ["Performance Summary:\n"]

        for name, rewards in all_rewards.items():
            stats_lines.append(f"\n{name}:")
            stats_lines.append(f"  Mean: {np.mean(rewards):.2f}")
            stats_lines.append(f"  Total: {np.sum(rewards):.2f}")
            stats_lines.append(f"  Steps: {len(rewards)}")

        # Calculate improvements if we have the data
        if 'LLM (Before FT)' in all_rewards and 'LLM (After FT)' in all_rewards:
            before_mean = np.mean(all_rewards['LLM (Before FT)'])
            after_mean = np.mean(all_rewards['LLM (After FT)'])
            improvement = (after_mean - before_mean) / abs(before_mean) * 100
            stats_lines.append(f"\nFine-tuning Improvement:")
            stats_lines.append(f"  {improvement:+.1f}%")

        if 'PPO (Expert)' in all_rewards and 'LLM (After FT)' in all_rewards:
            ppo_mean = np.mean(all_rewards['PPO (Expert)'])
            llm_mean = np.mean(all_rewards['LLM (After FT)'])
            vs_ppo = (llm_mean - ppo_mean) / abs(ppo_mean) * 100
            stats_lines.append(f"\nLLM vs PPO:")
            stats_lines.append(f"  {vs_ppo:+.1f}%")

        stats_text = "\n".join(stats_lines)

        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax6.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

        plt.suptitle('Final Comparison: All Methods', fontsize=16, fontweight='bold', y=0.995)

        if save:
            output_path = self.output_dir / "06_final_comparison.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {output_path}")
            plt.close()
            return str(output_path)

        return None

    def generate_all_reports(self):
        """Generate all visualization reports"""
        logger.info("="*60)
        logger.info("Generating All Visualization Reports")
        logger.info("="*60)

        results = {}

        # Stage 1: PPO Training
        try:
            results['ppo'] = self.visualize_ppo_training(save=True)
        except Exception as e:
            logger.error(f"Failed to visualize PPO: {e}")
            results['ppo'] = None

        # Stage 2: Few-shot Selection
        try:
            results['fewshot'] = self.visualize_fewshot_selection(save=True)
        except Exception as e:
            logger.error(f"Failed to visualize few-shot: {e}")
            results['fewshot'] = None

        # Stage 3: LLM Rollout
        try:
            results['rollout'] = self.visualize_llm_rollout(save=True)
        except Exception as e:
            logger.error(f"Failed to visualize LLM rollout: {e}")
            results['rollout'] = None

        # Stage 4: Self-Distillation
        try:
            results['distillation'] = self.visualize_self_distillation(save=True)
        except Exception as e:
            logger.error(f"Failed to visualize distillation: {e}")
            results['distillation'] = None

        # Stage 5: Fine-tuning
        try:
            results['finetuning'] = self.visualize_finetuning(save=True)
        except Exception as e:
            logger.error(f"Failed to visualize fine-tuning: {e}")
            results['finetuning'] = None

        # Stage 6: Final Comparison
        try:
            results['comparison'] = self.visualize_final_comparison(save=True)
        except Exception as e:
            logger.error(f"Failed to create final comparison: {e}")
            results['comparison'] = None

        # Summary
        logger.info("="*60)
        logger.info("Report Generation Summary")
        logger.info("="*60)

        for stage, path in results.items():
            if path:
                logger.info(f"✅ {stage:15s}: {path}")
            else:
                logger.info(f"❌ {stage:15s}: Failed or not available")

        logger.info("="*60)
        logger.info(f"All reports saved to: {self.output_dir}")
        logger.info("="*60)

        return results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate pipeline visualization reports")
    parser.add_argument("--pipeline_dir", type=str, default="./pipeline_output",
                       help="Pipeline output directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Report output directory (default: pipeline_dir/reports)")
    parser.add_argument("--stage", type=str, default="all",
                       choices=["all", "ppo", "fewshot", "rollout", "distillation", "finetuning", "comparison"],
                       help="Which stage to visualize")

    args = parser.parse_args()

    # Create visualizer
    visualizer = PipelineVisualizer(args.pipeline_dir, args.output_dir)

    # Generate reports
    if args.stage == "all":
        visualizer.generate_all_reports()
    elif args.stage == "ppo":
        visualizer.visualize_ppo_training(save=True)
    elif args.stage == "fewshot":
        visualizer.visualize_fewshot_selection(save=True)
    elif args.stage == "rollout":
        visualizer.visualize_llm_rollout(save=True)
    elif args.stage == "distillation":
        visualizer.visualize_self_distillation(save=True)
    elif args.stage == "finetuning":
        visualizer.visualize_finetuning(save=True)
    elif args.stage == "comparison":
        visualizer.visualize_final_comparison(save=True)


if __name__ == "__main__":
    main()
