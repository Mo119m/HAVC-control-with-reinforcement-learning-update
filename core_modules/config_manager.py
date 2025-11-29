"""
Configuration Management - Centralized Config System

This module provides centralized configuration management for the HVAC-LLM project.
Supports loading from JSON files, environment variables, and command-line arguments.

Key Features:
- Hierarchical configuration (defaults < file < env < CLI)
- Validation and type checking
- Easy serialization/deserialization
- Environment-specific configs
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class LLMConfig:
    """LLM inference configuration"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.0
    hf_token: Optional[str] = None
    
    def __post_init__(self):
        # Load from environment if available
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.hf_token = os.getenv("HF_TOKEN", self.hf_token)


@dataclass
class HVACConfig:
    """HVAC environment configuration"""
    building: str = "OfficeSmall"
    climate: str = "Hot_Dry"
    location: str = "Tucson"
    target_temp: float = 22.0
    data_root: str = "./BEAR/Data/"
    
    def __post_init__(self):
        self.building = os.getenv("BUILDING", self.building)
        self.climate = os.getenv("WEATHER", self.climate)
        self.location = os.getenv("LOCATION", self.location)


@dataclass
class FewShotConfig:
    """Few-shot selection configuration"""
    json_path: str = "fs_out/few_shot_examples_structured.json"
    k: int = 3
    alpha: float = 0.6
    weights: Optional[List[float]] = None
    
    def __post_init__(self):
        self.json_path = os.getenv("FEWSHOT_JSON", self.json_path)
        if os.getenv("K_FEWSHOT"):
            self.k = int(os.getenv("K_FEWSHOT"))
        if os.getenv("FEWSHOT_ALPHA"):
            self.alpha = float(os.getenv("FEWSHOT_ALPHA"))


@dataclass
class PromptConfig:
    """Prompt construction configuration"""
    history_lines: int = 3
    enable_history: bool = True
    enable_reasoning: bool = True
    max_history_keep: int = 10


@dataclass
class TrainingConfig:
    """PPO training configuration"""
    total_steps: int = 500000
    n_envs: int = 1
    seed: int = 42
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    def __post_init__(self):
        if os.getenv("TOTAL_STEPS"):
            self.total_steps = int(os.getenv("TOTAL_STEPS"))


@dataclass
class FineTuningConfig:
    """LoRA fine-tuning configuration"""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    epochs: int = 4
    lr: float = 1e-5
    batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 1500
    
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # PPO parameters
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    kl_coef: float = 0.0
    
    def __post_init__(self):
        self.base_model = os.getenv("BASE_MODEL", self.base_model)
        if os.getenv("EPOCHS"):
            self.epochs = int(os.getenv("EPOCHS"))
        if os.getenv("LR"):
            self.lr = float(os.getenv("LR"))


@dataclass
class PathConfig:
    """File paths configuration"""
    save_dir: str = "./runs_officesmall_hotdry"
    trajectory_file: str = "ppo_trajectory.json"
    fewshot_dir: str = "fs_out"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    def __post_init__(self):
        self.save_dir = os.getenv("SAVE_DIR", self.save_dir)
        
        # Create directories
        for dir_path in [self.save_dir, self.fewshot_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @property
    def trajectory_path(self) -> str:
        return os.path.join(self.save_dir, self.trajectory_file)
    
    @property
    def fewshot_path(self) -> str:
        return os.path.join(self.fewshot_dir, "few_shot_examples_structured.json")


@dataclass
class ProjectConfig:
    """Master configuration for entire project"""
    env: Environment = Environment.DEVELOPMENT
    llm: LLMConfig = field(default_factory=LLMConfig)
    hvac: HVACConfig = field(default_factory=HVACConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    finetuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    @classmethod
    def load_from_file(cls, path: str) -> "ProjectConfig":
        """
        Load configuration from JSON file.
        
        Args:
            path: Path to JSON config file
            
        Returns:
            ProjectConfig instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        logger.info(f"Loading config from {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        # Parse each section
        config = cls(
            env=Environment(data.get("env", "development")),
            llm=LLMConfig(**data.get("llm", {})),
            hvac=HVACConfig(**data.get("hvac", {})),
            fewshot=FewShotConfig(**data.get("fewshot", {})),
            prompt=PromptConfig(**data.get("prompt", {})),
            training=TrainingConfig(**data.get("training", {})),
            finetuning=FineTuningConfig(**data.get("finetuning", {})),
            paths=PathConfig(**data.get("paths", {})),
        )
        
        return config
    
    def save_to_file(self, path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: Path to save config
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        # Convert to dict
        data = {
            "env": self.env.value,
            "llm": asdict(self.llm),
            "hvac": asdict(self.hvac),
            "fewshot": asdict(self.fewshot),
            "prompt": asdict(self.prompt),
            "training": asdict(self.training),
            "finetuning": asdict(self.finetuning),
            "paths": asdict(self.paths),
        }
        
        # Save
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved config to {path}")
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate LLM config
        if self.llm.max_new_tokens <= 0:
            errors.append("llm.max_new_tokens must be positive")
        if not (0.0 <= self.llm.temperature <= 2.0):
            errors.append("llm.temperature must be in [0, 2]")
        
        # Validate HVAC config
        if not (10.0 <= self.hvac.target_temp <= 30.0):
            errors.append("hvac.target_temp must be in [10, 30]")
        
        # Validate few-shot config
        if self.fewshot.k <= 0:
            errors.append("fewshot.k must be positive")
        if not (0.0 <= self.fewshot.alpha <= 1.0):
            errors.append("fewshot.alpha must be in [0, 1]")
        
        # Validate training config
        if self.training.total_steps <= 0:
            errors.append("training.total_steps must be positive")
        if not (0.0 < self.training.gamma <= 1.0):
            errors.append("training.gamma must be in (0, 1]")
        
        # Validate fine-tuning config
        if self.finetuning.epochs <= 0:
            errors.append("finetuning.epochs must be positive")
        if self.finetuning.lr <= 0:
            errors.append("finetuning.lr must be positive")
        
        return errors
    
    def __str__(self) -> str:
        """Pretty print configuration"""
        lines = ["Project Configuration:", "=" * 50]
        lines.append(f"Environment: {self.env.value}")
        lines.append("\nLLM:")
        lines.append(f"  Model: {self.llm.model_name}")
        lines.append(f"  Temperature: {self.llm.temperature}")
        lines.append("\nHVAC:")
        lines.append(f"  Building: {self.hvac.building}")
        lines.append(f"  Climate: {self.hvac.climate}")
        lines.append(f"  Location: {self.hvac.location}")
        lines.append("\nPaths:")
        lines.append(f"  Save dir: {self.paths.save_dir}")
        lines.append(f"  Trajectory: {self.paths.trajectory_path}")
        lines.append("=" * 50)
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def get_default_config(env: str = "development") -> ProjectConfig:
    """
    Get default configuration for specified environment.
    
    Args:
        env: Environment name (development/production/testing)
        
    Returns:
        ProjectConfig instance
    """
    config = ProjectConfig(env=Environment(env))
    
    # Apply environment-specific adjustments
    if env == "testing":
        config.training.total_steps = 1000
        config.finetuning.epochs = 1
    
    return config


def load_config(
    config_file: Optional[str] = None,
    env: Optional[str] = None
) -> ProjectConfig:
    """
    Load configuration with fallback chain.
    
    Priority: CLI args > Config file > Env vars > Defaults
    
    Args:
        config_file: Path to config JSON (None to skip)
        env: Environment name (None to use default)
        
    Returns:
        ProjectConfig instance
    """
    # Start with defaults
    if env:
        config = get_default_config(env)
    else:
        config = ProjectConfig()
    
    # Override with file if provided
    if config_file and os.path.exists(config_file):
        config = ProjectConfig.load_from_file(config_file)
    
    # Environment variables are already loaded in __post_init__
    
    # Validate
    errors = config.validate()
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        raise ValueError("Invalid configuration")
    
    return config


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create default config
    config = get_default_config("development")
    
    print(config)
    
    # Save to file
    config.save_to_file("config_example.json")
    
    # Load from file
    loaded = ProjectConfig.load_from_file("config_example.json")
    
    print("\nValidation:")
    errors = loaded.validate()
    if errors:
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ Configuration valid")
