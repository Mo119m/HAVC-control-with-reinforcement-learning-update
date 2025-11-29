# HVAC Control with LLM - Optimized Project

## Overview

This project uses Large Language Models (Qwen 7B) for HVAC control optimization through:
1. **PPO Training**: Collect high-quality trajectory data
2. **Few-Shot Selection**: Select diverse, high-reward examples
3. **LLM Inference**: Use LLM as controller with prompt engineering
4. **LoRA Fine-tuning**: Improve LLM performance with PPO-style optimization

## Project Structure

```
.
├── llm_agent_colab.py          # LLM inference and action parsing
├── prompt_builder_control.py   # Prompt construction for HVAC control
├── few_shot_auto.py            # Automatic few-shot example selection
├── recorder_v2.py              # Trajectory recording callback
├── ppo_collect.py              # PPO training and data collection
├── select_representative.py    # Sample selection with clustering
├── draw_reward.py              # Reward visualization
├── config_manager.py           # Configuration management system (new)
├── test_suite.py               # Test suite (new)
├── main_pipeline.py            # Main pipeline program (new)
├── 7b_finetune_fixed.py        # LoRA fine-tuning (FIXED VERSION)
└── README.md                   # This file
```

## Key Improvements

### 1. Code Quality
- Type hints and dataclasses
- Comprehensive error handling
- Logging instead of print statements
- Docstrings for all functions
- Validation functions

### 2. Reliability
- Robust parsing with multiple fallback strategies
- Safe array clipping and bounds checking
- Configuration validation
- Checkpoint and resume support

### 3. Maintainability
- Clear separation of concerns
- Configurable parameters
- Reusable components
- Comprehensive testing utilities

## Installation

```bash
# Install dependencies
pip install torch transformers stable-baselines3 scikit-learn pandas matplotlib

# For BEAR environment (if available)
pip install -e /path/to/BEAR

# For LoRA fine-tuning
pip install peft
```

## Quick Start

### Using the Pipeline (Recommended)

```bash
# Run all stages automatically
python main_pipeline.py --stage all

# Or run individual stages
python main_pipeline.py --stage ppo
python main_pipeline.py --stage select
python main_pipeline.py --stage finetune
```

### Manual Usage

#### 1. Collect PPO Trajectory

```bash
# Train PPO agent and collect trajectory data
export BUILDING="OfficeSmall"
export WEATHER="Hot_Dry"
export LOCATION="Tucson"
export TOTAL_STEPS="500000"
python ppo_collect.py
```

**Outputs:**
- `runs_officesmall_hotdry/ppo_trajectory.json`: Full trajectory
- `runs_officesmall_hotdry/ppo_final.zip`: Trained model
- `runs_officesmall_hotdry/checkpoints/`: Training checkpoints

#### 2. Select Representative Samples

```bash
python select_representative.py \
    --traj runs_officesmall_hotdry/ppo_trajectory.json \
    --out_dir fs_out \
    --preselect 2000 \
    --clusters 12 \
    --n_per_cluster 20
```

**Outputs:**
- `fs_out/few_shot_examples_structured.json`: Selected examples

#### 3. Run LLM as Controller

```python
from llm_agent_colab import call_llm, parse_actions_with_validation
from prompt_builder_control import build_prompt
from few_shot_auto import load_examples, select_examples

# Load few-shot examples
examples = load_examples("fs_out/few_shot_examples_structured.json")

# For each control step:
prompt = build_prompt(
    obs=current_obs,
    building="OfficeSmall",
    location="Tucson",
    climate="Hot_Dry",
    target=22.0
)

# Generate actions
raw_text = call_llm(prompt, n_actions=n_zones)
actions, meta = parse_actions_with_validation(raw_text, n_zones)
```

#### 4. Visualize Results

```bash
python draw_reward.py \
    ppo_trajectory.json \
    llm_rollout.json \
    --output comparison.png \
    --smooth 5
```

#### 5. Fine-tune with LoRA

**IMPORTANT: Use the FIXED version!**

```bash
# Use the corrected fine-tuning script
python 7b_finetune_fixed.py

# Configuration via environment variables
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export ROLLOUT_GLOBS="llm_rollout/trajectory.json"
export SAVE_DIR="./ft_out_ppo_7b_lora"
```

## Critical Fix: Fine-tuning Code

**IMPORTANT:** The original `7b_finetune.py` has a serious bug in the PPO importance sampling ratio calculation!

### The Problem

```python
# WRONG: Original code
for epoch in range(EPOCHS):
    recompute_old_dists()  # Computes old_lp with current model
    
    for batch in train_loader:
        new_lp = ...  # Computes with current model
        ratio = exp(new_lp - old_lp)  # Both from same model!
        # Result: ratio ≈ 1, PPO clip is useless
```

### The Fix

```python
# CORRECT: Fixed version
for epoch in range(EPOCHS):
    # Compute old policy ONCE at epoch start and FREEZE
    with torch.no_grad():
        old_lp_vec = compute_old_policy()
    
    for batch in train_loader:
        new_lp = ...  # From training model
        old_lp_b = old_lp_vec[batch_idx]  # From FROZEN old policy
        ratio = exp(new_lp - old_lp_b)  # Correct importance sampling
```

**Always use `7b_finetune_fixed.py` or run through the pipeline!**

## Configuration Guide

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | LLM model identifier |
| `HF_TOKEN` | None | HuggingFace API token |
| `BUILDING` | `OfficeSmall` | Building type |
| `WEATHER` | `Hot_Dry` | Climate type |
| `LOCATION` | `Tucson` | Location name |
| `SAVE_DIR` | `./runs_officesmall_hotdry` | Output directory |

### Key Parameters

**LLM Inference:**
- `temperature`: 0.3-0.7 for reasoning-enabled generation
- `top_p`: 0.7 for nucleus sampling
- `top_k`: 50 for top-k sampling
- `max_new_tokens`: 256 for reasoning + action output

**Few-Shot Selection:**
- `k`: 3 examples per prompt
- `alpha`: 0.6 (60% similarity, 40% reward)
- `weights`: [1.0]*n_rooms + [0.5, 0.2, 0.2, 0.1] for features

**Sample Selection:**
- `preselect`: 2000 top samples by reward
- `clusters`: 12 for diversity
- `n_per_cluster`: 20 samples per cluster

## Data Format

### Trajectory JSON
```json
[
  {
    "obs": [22.5, 23.0, ...],
    "action": [0.1, -0.2, ...],
    "reward": -5.2,
    "next_obs": [22.3, 22.8, ...],
    "done": false
  }
]
```

### Few-Shot Examples JSON
```json
[
  {
    "obs": [22.0, 22.5, ...],
    "actions": [0.1, -0.2, ...],
    "reward": -5.2,
    "building": "OfficeSmall",
    "climate": "Hot_Dry"
  }
]
```

## Troubleshooting

### Issue: Model outputs invalid format
**Solution:** 
- Check `parse_actions` fallback strategies
- Enable reasoning with proper prompt instructions
- Increase `max_new_tokens` if output is truncated

### Issue: Out of memory during fine-tuning
**Solution:**
- Reduce `BATCH_SIZE` to 1
- Increase `GRAD_ACCUM` for effective larger batch
- Enable gradient checkpointing
- Use `bfloat16` instead of `float16`

### Issue: Poor action quality
**Solution:**
- Increase number of few-shot examples (`k`)
- Adjust `alpha` to balance similarity/reward
- Add more history context (`history_lines`)
- Fine-tune model with LoRA (use FIXED version)

## Testing

```bash
# Run test suite
python test_suite.py

# Verbose mode
python test_suite.py --verbose
```

## Documentation

- **SETUP_TUTORIAL.md** - Complete setup and usage guide
- **CODE_TRUST_REPORT.md** - Code reliability and trust report
- **FINAL_CHECKLIST.md** - Final checklist before use
- **QUICK_REFERENCE.md** - Quick reference for common commands
- **IMPROVEMENTS_SUMMARY.md** - Detailed improvement summary

## Citation

If you use this code, please cite:
```
@misc{hvac-llm-control,
  title={HVAC Control with Large Language Models},
  year={2024}
}
```

## License

MIT License

## References

- [Qwen2.5 Model](https://github.com/QwenLM/Qwen2.5)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
