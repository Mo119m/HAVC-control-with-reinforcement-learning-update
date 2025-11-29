"""
Fine-tuned 7B LoRA Model Rollout - Optimized Version

Evaluates the fine-tuned Qwen 7B + LoRA model in the HVAC environment.

Improvements:
1. Type hints and dataclasses
2. Model loading with error handling
3. Configuration management
4. Better logging and validation
5. Checkpoint saving
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available, LoRA loading disabled")

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
class FinetunedRolloutConfig:
    """Configuration for fine-tuned model rollout"""
    # Model
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    finetune_adapter: str = "./ft_out_ppo_7b_lora"
    
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
    
    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.3
    do_sample: bool = True
    
    # Few-shot
    fewshot_json: Optional[str] = None
    k_fewshot: int = 3
    fewshot_alpha: float = 0.6
    
    # Output
    save_path: str = "./outputs/finetuned_rollout.json"
    save_interval: int = 50
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        """Load from environment variables"""
        self.base_model = os.getenv("BASE_MODEL", self.base_model)
        self.finetune_adapter = os.getenv("FINETUNE_ADAPTER", self.finetune_adapter)
        self.building = os.getenv("BUILDING", self.building)
        self.climate = os.getenv("CLIMATE", self.climate)
        self.location = os.getenv("LOCATION", self.location)
        self.save_path = os.getenv("SAVE_PATH", self.save_path)
        
        if os.getenv("MAX_STEPS"):
            self.max_steps = int(os.getenv("MAX_STEPS"))
        if os.getenv("FEWSHOT_JSON"):
            self.fewshot_json = os.getenv("FEWSHOT_JSON")


# ============================================================================
# Model Loading
# ============================================================================

class FinetunedLLMAgent:
    """LLM agent with fine-tuned LoRA adapter"""
    
    def __init__(self, config: FinetunedRolloutConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"Loading model: {config.base_model}")
        logger.info(f"Loading adapter: {config.finetune_adapter}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        except Exception as e:
            logger.warning(f"Failed to load with bfloat16: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Load LoRA adapter
        if PEFT_AVAILABLE and os.path.exists(config.finetune_adapter):
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    config.finetune_adapter
                )
                logger.info("Loaded LoRA adapter successfully")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter: {e}")
                logger.warning("Using base model without adapter")
        else:
            logger.warning(f"LoRA adapter not found: {config.finetune_adapter}")
            logger.warning("Using base model without fine-tuning")
        
        self.model.eval()
        logger.info(f"Model loaded on device: {self.device}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        except Exception as e:
            logger.error(f"Failed to apply chat template: {e}")
            # Fallback to simple encoding
            inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0, inputs.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text


# ============================================================================
# Utility Functions
# ============================================================================

def extract_outside_temp(obs: List[float]) -> float:
    """Extract outside temperature from observation"""
    try:
        n = (len(obs) - 2) // 3
        return float(obs[n])
    except Exception as e:
        logger.warning(f"Failed to extract outside temp: {e}")
        return 0.0


def save_checkpoint(logs: List[Dict], save_path: str, step: int) -> None:
    """Save checkpoint"""
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

def run_finetuned_rollout(config: FinetunedRolloutConfig) -> List[Dict]:
    """
    Run rollout with fine-tuned model.
    
    Args:
        config: Rollout configuration
        
    Returns:
        List of log entries
    """
    # Import here to avoid circular dependency
    from prompt_builder_control import build_prompt, zone_count_from_obs
    from llm_agent_colab import parse_actions_with_validation
    from few_shot_auto import (
        load_examples, select_examples,
        format_few_shot_block, inject_few_shot
    )
    
    # Create output directory
    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
    
    # Load model
    logger.info("Loading fine-tuned model...")
    agent = FinetunedLLMAgent(config)
    
    # Load few-shot examples
    ex_dataset = None
    if config.fewshot_json and os.path.exists(config.fewshot_json):
        try:
            ex_dataset = load_examples(config.fewshot_json)
            logger.info(f"Loaded {len(ex_dataset)} few-shot examples")
        except Exception as e:
            logger.warning(f"Failed to load few-shot: {e}")
    
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
    
    # Initialize
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
        
        # Inject few-shot
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
        
        # Generate with fine-tuned model
        try:
            raw_text = agent.generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raw_text = ""
        
        # Parse actions
        action_unit, meta = parse_actions_with_validation(raw_text, n_zones)
        
        if action_unit is None:
            logger.warning("Parse failed, using zero action")
            action_unit = [0.0] * n_zones
            meta = {"parsed_from": "fallback_zero"}
        
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
            "reward": float(reward),
            "done": bool(done),
            "obs": obs,
            "next_obs": obs_next,
            "env_temp": extract_outside_temp(obs),
            "model": "finetuned_7b_lora"
        }
        
        logs.append(log_entry)
        
        # Save checkpoint
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(logs, config.save_path, step + 1)
        
        # Update obs
        obs = obs_next
        
        # Check done
        if done:
            logger.info(f"Episode ended at step {step+1}")
            break
    
    # Save final
    try:
        with open(config.save_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved final rollout: {config.save_path}")
    except Exception as e:
        logger.error(f"Failed to save: {e}")
    
    # Summary
    if logs:
        rewards = [log["reward"] for log in logs]
        logger.info(f"\nFine-tuned Model Rollout Summary:")
        logger.info(f"  Total steps: {len(logs)}")
        logger.info(f"  Mean reward: {np.mean(rewards):.2f}")
        logger.info(f"  Std reward: {np.std(rewards):.2f}")
        logger.info(f"  Min reward: {np.min(rewards):.2f}")
        logger.info(f"  Max reward: {np.max(rewards):.2f}")
        logger.info(f"  Total reward: {np.sum(rewards):.2f}")
    
    return logs


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fine-tuned model rollout")
    parser.add_argument("--config", type=str, help="Config JSON path")
    parser.add_argument("--adapter", type=str, help="LoRA adapter path")
    parser.add_argument("--max_steps", type=int, help="Maximum steps")
    parser.add_argument("--output", type=str, help="Output path")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = FinetunedRolloutConfig(**config_dict)
    else:
        config = FinetunedRolloutConfig()
    
    # Override
    if args.adapter:
        config.finetune_adapter = args.adapter
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.output:
        config.save_path = args.output
    
    logger.info("Starting fine-tuned model rollout...")
    logger.info(f"Config: {config}")
    
    try:
        logs = run_finetuned_rollout(config)
        logger.info("Rollout completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
