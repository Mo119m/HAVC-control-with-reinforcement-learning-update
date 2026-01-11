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

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core_modules.visualization_pipeline import PipelineVisualizer
from core_modules.drive_backup import DriveBackup

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
    reports_dir: str = "reports"
    
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
    llm_rollout_episodes: int = 1  # Number of episodes to run
    llm_rollout_max_steps: int = 200  # Steps per episode
    
    # Fine-tuning
    finetune_epochs: int = 4
    finetune_lr: float = 1e-5
    finetune_batch_size: int = 1
    finetune_grad_accum: int = 8
    
    # Evaluation
    eval_episodes: int = 10

    # Drive Backup (optional)
    drive_backup_path: Optional[str] = None  # e.g., "/content/drive/MyDrive/HVAC-RL-Backup"
    resume_from_backup: bool = True  # Try to resume from backup if available

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
        # Filter out comment fields (keys starting with '_comment')
        filtered_data = {k: v for k, v in data.items() if not k.startswith('_comment')}
        return cls(**filtered_data)


class Pipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = config.get_paths()
        self._create_directories()

        # Initialize visualizer
        reports_path = Path(self.config.base_dir) / self.config.reports_dir
        self.visualizer = PipelineVisualizer(
            pipeline_dir=self.config.base_dir,
            output_dir=str(reports_path)
        )

        # Initialize Drive backup
        self.drive_backup = DriveBackup(
            drive_path=self.config.drive_backup_path,
            local_path=self.config.base_dir
        )
        self.drive_backup.print_backup_status()
    
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
            base / self.config.reports_dir,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {d}")

    def _backup_stage(self, stage_name: str, stage_dir: str):
        """Backup a stage to Drive after completion"""
        if self.drive_backup.enabled:
            self.drive_backup.backup_stage(stage_name, stage_dir)
            # Also backup reports if they exist
            reports_dir = Path(self.config.base_dir) / self.config.reports_dir
            if reports_dir.exists():
                self.drive_backup.backup_reports(str(reports_dir))

    def _try_restore_stage(self, stage_name: str, stage_dir: str) -> bool:
        """
        Try to restore a stage from Drive backup.

        Returns:
            True if successfully restored, False otherwise
        """
        if not self.config.resume_from_backup:
            return False

        if self.drive_backup.check_stage_exists(stage_name):
            logger.info(f"Found backup for {stage_name}, attempting to restore...")
            return self.drive_backup.restore_stage(stage_name, stage_dir)

        return False
    
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
        stage_name = self.config.ppo_dir
        stage_path = str(Path(self.config.base_dir) / stage_name)

        # Try to restore from backup
        if self._try_restore_stage(stage_name, stage_path):
            if self.paths["ppo_trajectory"].exists():
                logger.info("✅ PPO training restored from backup, skipping training")
                return True

        logger.info("Starting PPO training...")

        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",  # Disable output buffering for real-time logs
            "BUILDING": self.config.building,
            "WEATHER": self.config.weather,
            "LOCATION": self.config.location,
            "TOTAL_STEPS": str(self.config.ppo_total_steps),
            "SAVE_DIR": stage_path,
        }

        try:
            # Run with real-time output streaming
            result = subprocess.run(
                ["python", "core_modules/ppo_collect.py"],
                env=env,
                check=True,
                # Don't capture output - let it stream to console in real-time
                stdout=None,
                stderr=None
            )

            logger.info("PPO training completed")

            # Verify output
            if not self.paths["ppo_trajectory"].exists():
                logger.error("PPO trajectory file not found")
                return False

            # Generate visualizations
            logger.info("Generating PPO training visualizations...")
            try:
                self.visualizer.visualize_ppo_training()
                logger.info("✅ PPO visualizations saved")
            except Exception as e:
                logger.warning(f"Failed to generate PPO visualizations: {e}")

            # Backup to Drive
            self._backup_stage(stage_name, stage_path)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"PPO training failed: {e}")
            return False
    
    def _run_sample_selection(self) -> bool:
        """Stage 2: Sample Selection"""
        stage_name = self.config.samples_dir
        stage_path = str(Path(self.config.base_dir) / stage_name)

        # Try to restore from backup
        if self._try_restore_stage(stage_name, stage_path):
            if self.paths["fewshot_json"].exists():
                logger.info("✅ Sample selection restored from backup, skipping selection")
                return True

        logger.info("Starting sample selection...")

        if not self.paths["ppo_trajectory"].exists():
            logger.error("PPO trajectory not found, run 'ppo' stage first")
            return False
        
        cmd = [
            "python", "core_modules/select_representative.py",
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
            # Run with real-time output streaming
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                stdout=None,
                stderr=None
            )

            logger.info("Sample selection completed")

            # Verify output
            if not self.paths["fewshot_json"].exists():
                logger.error("Few-shot examples file not found")
                return False

            # Generate visualizations
            logger.info("Generating few-shot selection visualizations...")
            try:
                self.visualizer.visualize_fewshot_selection()
                logger.info("✅ Few-shot selection visualizations saved")
            except Exception as e:
                logger.warning(f"Failed to generate few-shot visualizations: {e}")

            # Backup to Drive
            self._backup_stage(stage_name, stage_path)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Sample selection failed: {e}")
            return False
    
    def _run_llm_rollout(self) -> bool:
        """Stage 3: LLM Rollout"""
        stage_name = self.config.llm_rollout_dir
        stage_path = str(Path(self.config.base_dir) / stage_name)

        # Try to restore from backup
        if self._try_restore_stage(stage_name, stage_path):
            if self.paths["llm_rollout_trajectory"].exists():
                logger.info("✅ LLM rollout restored from backup, skipping rollout")
                return True

        logger.info("Starting LLM rollout...")

        if not self.paths["fewshot_json"].exists():
            logger.error("Few-shot examples not found, run 'select' stage first")
            return False

        # Set environment variables for rollout script
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",  # Disable output buffering for real-time logs
            "BUILDING": self.config.building,
            "CLIMATE": self.config.weather,
            "LOCATION": self.config.location,
            "FEWSHOT_JSON": str(self.paths["fewshot_json"]),
            "SAVE_PATH": str(self.paths["llm_rollout_trajectory"]),
            "MAX_STEPS": str(self.config.llm_rollout_max_steps),
            "NUM_EPISODES": str(self.config.llm_rollout_episodes),
            "K_FEWSHOT": str(self.config.k_fewshot),
            "FEWSHOT_ALPHA": str(self.config.fewshot_alpha),
            "MODEL_NAME": self.config.model_name,
        }

        # Add Drive backup path if enabled (for real-time backup during rollout)
        if self.drive_backup.enabled:
            drive_rollout_backup = self.drive_backup.drive_path / self.config.llm_rollout_dir
            env["DRIVE_BACKUP_PATH"] = str(drive_rollout_backup)
            logger.info(f"Real-time Drive backup enabled: {drive_rollout_backup}")

        logger.info(f"Running LLM rollout with model: {self.config.model_name}")
        logger.info(f"Episodes: {self.config.llm_rollout_episodes} × {self.config.llm_rollout_max_steps} steps")
        logger.info(f"Total expected steps: {self.config.llm_rollout_episodes * self.config.llm_rollout_max_steps}")
        logger.info(f"Few-shot examples: {self.paths['fewshot_json']}")
        logger.info(f"Output: {self.paths['llm_rollout_trajectory']}")

        try:
            # Calculate timeout based on number of episodes (assume ~5 min per episode + overhead)
            timeout_per_episode = 600  # 10 minutes per episode (conservative estimate)
            timeout = max(7200, self.config.llm_rollout_episodes * timeout_per_episode)
            logger.info(f"Timeout set to {timeout//60} minutes for {self.config.llm_rollout_episodes} episodes")

            # Run with real-time output streaming
            result = subprocess.run(
                ["python", "core_modules/rollout_fewshot_version.py"],
                env=env,
                check=True,
                stdout=None,
                stderr=None,
                timeout=timeout
            )

            logger.info("LLM rollout completed")

            # Verify output
            if not self.paths["llm_rollout_trajectory"].exists():
                logger.error("LLM rollout trajectory file not found")
                return False

            # Generate visualizations
            logger.info("Generating LLM rollout visualizations...")
            try:
                self.visualizer.visualize_llm_rollout()
                logger.info("✅ LLM rollout visualizations saved")
            except Exception as e:
                logger.warning(f"Failed to generate LLM rollout visualizations: {e}")

            # Backup to Drive
            self._backup_stage(stage_name, stage_path)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"LLM rollout failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("LLM rollout timed out")
            return False
    
    def _run_finetuning(self) -> bool:
        """Stage 4: Fine-tuning"""
        stage_name = self.config.finetune_dir
        stage_path = str(Path(self.config.base_dir) / stage_name)

        # Try to restore from backup
        if self._try_restore_stage(stage_name, stage_path):
            finetune_model = Path(stage_path) / "final_model"
            if finetune_model.exists():
                logger.info("✅ Fine-tuning restored from backup, skipping training")
                return True

        logger.info("Starting fine-tuning...")

        if not self.paths["llm_rollout_trajectory"].exists():
            logger.error("LLM rollout trajectory not found, run 'rollout' stage first")
            return False
        
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",  # Disable output buffering for real-time logs
            "BASE_MODEL": self.config.model_name,
            "ROLLOUT_GLOBS": str(self.paths["llm_rollout_trajectory"]),
            "SAVE_DIR": str(Path(self.config.base_dir) / self.config.finetune_dir),
            "EPOCHS": str(self.config.finetune_epochs),
            "LR": str(self.config.finetune_lr),
        }
        
        try:
            # Run with real-time output streaming
            result = subprocess.run(
                ["python", "core_modules/7b_finetune_fixed.py"],
                env=env,
                check=True,
                stdout=None,
                stderr=None,
                timeout=7200  # 2 hours timeout
            )

            logger.info("Fine-tuning completed")

            # Generate visualizations
            logger.info("Generating fine-tuning visualizations...")
            try:
                self.visualizer.visualize_self_distillation()
                logger.info("✅ Fine-tuning visualizations saved")
            except Exception as e:
                logger.warning(f"Failed to generate fine-tuning visualizations: {e}")

            # Backup to Drive
            self._backup_stage(stage_name, stage_path)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Fine-tuning timed out")
            return False
    
    def _run_evaluation(self) -> bool:
        """Stage 5: Evaluation"""
        stage_name = self.config.eval_dir
        stage_path = str(Path(self.config.base_dir) / stage_name)

        # Try to restore from backup
        if self._try_restore_stage(stage_name, stage_path):
            if self.paths["eval_results"].exists():
                logger.info("✅ Evaluation restored from backup, skipping evaluation")
                return True

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
            "python", "core_modules/draw_reward.py",
            *trajectories,
            "--output", str(self.paths["eval_plots"]),
            "--smooth", "5",
        ]

        try:
            # Run with real-time output streaming
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                stdout=None,
                stderr=None
            )

            logger.info("Evaluation completed")

            # Save evaluation results
            results = {
                "trajectories_evaluated": len(trajectories),
                "plot_path": str(self.paths["eval_plots"]),
            }

            with open(self.paths["eval_results"], "w") as f:
                json.dump(results, f, indent=2)

            # Generate comprehensive comparison visualizations
            logger.info("Generating final comparison visualizations...")
            try:
                self.visualizer.visualize_final_comparison()
                logger.info("✅ Final comparison visualizations saved")
            except Exception as e:
                logger.warning(f"Failed to generate final comparison visualizations: {e}")

            # Backup to Drive
            self._backup_stage(stage_name, stage_path)

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation failed: {e}")
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
    parser.add_argument(
        "--drive_backup",
        type=str,
        help="Google Drive backup path (e.g., /content/drive/MyDrive/HVAC-RL-Backup)"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resuming from backup (always run all stages)"
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
    if args.drive_backup:
        config.drive_backup_path = args.drive_backup
    if args.no_resume:
        config.resume_from_backup = False
    
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

        # Print visualization reports
        reports_dir = Path(config.base_dir) / config.reports_dir
        logger.info(f"\nVisualization reports:")
        logger.info(f"  📊 Reports directory: {reports_dir}")
        if reports_dir.exists():
            report_files = sorted(reports_dir.glob("*.png"))
            for report_file in report_files:
                logger.info(f"  ✅ {report_file.name}")

        sys.exit(0)
    else:
        logger.error("\n" + "="*60)
        logger.error("PIPELINE FAILED")
        logger.error("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
