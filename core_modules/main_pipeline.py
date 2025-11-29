"""
Main Pipeline - Complete HVAC-LLM Workflow

This script runs the complete pipeline:
1. PPO Training → 2. Sample Selection → 3. LLM Rollout → 4. Fine-tuning → 5. Evaluation

Usage:
    python main_pipeline.py --config config.json
    python main_pipeline.py --stage ppo
    python main_pipeline.py --stage all
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    # Environment
    building: str = "OfficeSmall"
    weather: str = "Hot_Dry"
    location: str = "Tucson"
    
    # Directories
    data_root: str = "./BEAR/Data/"
    base_dir: str = "./pipeline_output"
    ppo_dir: str = "01_ppo_training"
    samples_dir: str = "02_few_shot_samples"
    llm_rollout_dir: str = "03_llm_rollout"
    finetune_dir: str = "04_finetuning"
    eval_dir: str = "05_evaluation"
    
    # PPO Training
    ppo_total_steps: int = 500000
    ppo_checkpoint_freq: int = 50000
    
    # Sample Selection
    preselect: int = 2000
    clusters: int = 12
    n_per_cluster: int = 20
    min_reward_percentile: float = 0.5
    
    # LLM Inference
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7
    k_fewshot: int = 3
    fewshot_alpha: float = 0.6
    
    # Fine-tuning
    finetune_epochs: int = 4
    finetune_lr: float = 1e-5
    finetune_batch_size: int = 1
    finetune_grad_accum: int = 8
    
    # Evaluation
    eval_episodes: int = 10
    
    def get_paths(self):
        """Get all relevant paths"""
        base = Path(self.base_dir)
        
        return {
            "ppo_trajectory": base / self.ppo_dir / "ppo_trajectory.json",
            "ppo_model": base / self.ppo_dir / "ppo_final.zip",
            "ppo_checkpoints": base / self.ppo_dir / "checkpoints",
            
            "fewshot_json": base / self.samples_dir / "few_shot_examples_structured.json",
            
            "llm_rollout_trajectory": base / self.llm_rollout_dir / "llm_rollout.json",
            
            "finetune_model": base / self.finetune_dir / "final_model",
            "finetune_checkpoints": base / self.finetune_dir / "checkpoints",
            
            "eval_results": base / self.eval_dir / "results.json",
            "eval_plots": base / self.eval_dir / "comparison_plot.png",
        }
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Saved config to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load config from JSON"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class Pipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = config.get_paths()
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        base = Path(self.config.base_dir)
        
        dirs = [
            base / self.config.ppo_dir,
            base / self.config.ppo_dir / "checkpoints",
            base / self.config.samples_dir,
            base / self.config.llm_rollout_dir,
            base / self.config.finetune_dir,
            base / self.config.eval_dir,
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {d}")
    
    def run_stage(self, stage: str) -> bool:
        """
        Run a specific pipeline stage.
        
        Args:
            stage: One of ['ppo', 'select', 'rollout', 'finetune', 'eval', 'all']
            
        Returns:
            True if successful
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running stage: {stage.upper()}")
        logger.info('='*60)
        
        if stage == "all":
            stages = ["ppo", "select", "rollout", "finetune", "eval"]
            for s in stages:
                if not self.run_stage(s):
                    logger.error(f"Stage {s} failed, stopping pipeline")
                    return False
            return True
        
        if stage == "ppo":
            return self._run_ppo_training()
        elif stage == "select":
            return self._run_sample_selection()
        elif stage == "rollout":
            return self._run_llm_rollout()
        elif stage == "finetune":
            return self._run_finetuning()
        elif stage == "eval":
            return self._run_evaluation()
        else:
            logger.error(f"Unknown stage: {stage}")
            return False
    
    def _run_ppo_training(self) -> bool:
        """Stage 1: PPO Training"""
        logger.info("Starting PPO training...")
        
        env = {
            **os.environ,
            "BUILDING": self.config.building,
            "WEATHER": self.config.weather,
            "LOCATION": self.config.location,
            "TOTAL_STEPS": str(self.config.ppo_total_steps),
            "SAVE_DIR": str(Path(self.config.base_dir) / self.config.ppo_dir),
        }
        
        try:
            result = subprocess.run(
                ["python", "ppo_collect.py"],
                env=env,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info("PPO training completed")
            logger.debug(result.stdout)
            
            # Verify output
            if not self.paths["ppo_trajectory"].exists():
                logger.error("PPO trajectory file not found")
                return False
            
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"PPO training failed: {e}")
            logger.error(e.stderr)
            return False
    
    def _run_sample_selection(self) -> bool:
        """Stage 2: Sample Selection"""
        logger.info("Starting sample selection...")
        
        if not self.paths["ppo_trajectory"].exists():
            logger.error("PPO trajectory not found, run 'ppo' stage first")
            return False
        
        cmd = [
            "python", "select_representative.py",
            "--traj", str(self.paths["ppo_trajectory"]),
            "--out_dir", str(Path(self.config.base_dir) / self.config.samples_dir),
            "--preselect", str(self.config.preselect),
            "--clusters", str(self.config.clusters),
            "--n_per_cluster", str(self.config.n_per_cluster),
            "--building", self.config.building,
            "--climate", self.config.weather,
            "--location", self.config.location,
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info("Sample selection completed")
            logger.debug(result.stdout)
            
            # Verify output
            if not self.paths["fewshot_json"].exists():
                logger.error("Few-shot examples file not found")
                return False
            
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Sample selection failed: {e}")
            logger.error(e.stderr)
            return False
    
    def _run_llm_rollout(self) -> bool:
        """Stage 3: LLM Rollout"""
        logger.info("Starting LLM rollout...")
        
        if not self.paths["fewshot_json"].exists():
            logger.error("Few-shot examples not found, run 'select' stage first")
            return False
        
        # This would call rollout_fewshot_version.py
        logger.warning("LLM rollout script not yet optimized")
        logger.info("Skipping LLM rollout for now...")
        
        # TODO: Implement when rollout_fewshot_version.py is optimized
        # For now, create a dummy file
        dummy_trajectory = [
            {
                "step": 0,
                "prompt": "dummy",
                "action": [0.0] * 3,
                "reward": -10.0,
                "done": False
            }
        ]
        
        with open(self.paths["llm_rollout_trajectory"], "w") as f:
            json.dump(dummy_trajectory, f, indent=2)
        
        return True
    
    def _run_finetuning(self) -> bool:
        """Stage 4: Fine-tuning"""
        logger.info("Starting fine-tuning...")
        
        if not self.paths["llm_rollout_trajectory"].exists():
            logger.error("LLM rollout trajectory not found, run 'rollout' stage first")
            return False
        
        env = {
            **os.environ,
            "BASE_MODEL": self.config.model_name,
            "ROLLOUT_GLOBS": str(self.paths["llm_rollout_trajectory"]),
            "SAVE_DIR": str(Path(self.config.base_dir) / self.config.finetune_dir),
            "EPOCHS": str(self.config.finetune_epochs),
            "LR": str(self.config.finetune_lr),
        }
        
        try:
            result = subprocess.run(
                ["python", "7b_finetune_fixed.py"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hours timeout
            )
            
            logger.info("Fine-tuning completed")
            logger.debug(result.stdout)
            
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Fine-tuning failed: {e}")
            logger.error(e.stderr)
            return False
        except subprocess.TimeoutExpired:
            logger.error("Fine-tuning timed out")
            return False
    
    def _run_evaluation(self) -> bool:
        """Stage 5: Evaluation"""
        logger.info("Starting evaluation...")
        
        # Collect all trajectory files
        trajectories = []
        
        if self.paths["ppo_trajectory"].exists():
            trajectories.append(str(self.paths["ppo_trajectory"]))
        
        if self.paths["llm_rollout_trajectory"].exists():
            trajectories.append(str(self.paths["llm_rollout_trajectory"]))
        
        if not trajectories:
            logger.error("No trajectories found for evaluation")
            return False
        
        # Plot comparison
        cmd = [
            "python", "draw_reward.py",
            *trajectories,
            "--output", str(self.paths["eval_plots"]),
            "--smooth", "5",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            logger.info("Evaluation completed")
            logger.debug(result.stdout)
            
            # Save evaluation results
            results = {
                "trajectories_evaluated": len(trajectories),
                "plot_path": str(self.paths["eval_plots"]),
            }
            
            with open(self.paths["eval_results"], "w") as f:
                json.dump(results, f, indent=2)
            
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error(e.stderr)
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="HVAC-LLM Complete Pipeline"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["ppo", "select", "rollout", "finetune", "eval", "all"],
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--building",
        type=str,
        help="Building type (overrides config)"
    )
    parser.add_argument(
        "--weather",
        type=str,
        help="Weather type (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading config from {args.config}")
        config = PipelineConfig.load(args.config)
    else:
        logger.info("Using default config")
        config = PipelineConfig()
    
    # Override with CLI args
    if args.building:
        config.building = args.building
    if args.weather:
        config.weather = args.weather
    
    # Save config
    config_path = Path(config.base_dir) / "pipeline_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(config_path))
    
    # Run pipeline
    pipeline = Pipeline(config)
    
    logger.info("\n" + "="*60)
    logger.info("HVAC-LLM PIPELINE")
    logger.info("="*60)
    logger.info(f"Building: {config.building}")
    logger.info(f"Weather: {config.weather}")
    logger.info(f"Location: {config.location}")
    logger.info(f"Base directory: {config.base_dir}")
    logger.info(f"Stage: {args.stage}")
    logger.info("="*60 + "\n")
    
    success = pipeline.run_stage(args.stage)
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Results saved to: {config.base_dir}")
        
        # Print paths
        paths = config.get_paths()
        logger.info("\nKey outputs:")
        for name, path in paths.items():
            if path.exists():
                logger.info(f"  ✅ {name}: {path}")
            else:
                logger.info(f"  ⏳ {name}: {path} (pending)")
        
        sys.exit(0)
    else:
        logger.error("\n" + "="*60)
        logger.error("PIPELINE FAILED")
        logger.error("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
