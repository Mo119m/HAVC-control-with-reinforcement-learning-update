"""
7B LoRA Fine-tuning with PPO - Fixed Version

FIXES:
1. Fixed importance sampling ratio computation
2. old_lp computed at epoch start and frozen during epoch
3. Proper PPO clipping mechanism
4. Better error handling and logging

Key Changes:
- old_lp_vec computed ONCE per epoch and kept constant
- ratio = exp(new_lp - old_lp) where old_lp is from epoch start
- Added validation and checkpointing
"""

import os
import glob
import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available, LoRA disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Configuration

@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning"""
    # Model
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Training
    epochs: int = 4
    lr: float = 1e-5
    batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 1500
    
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # PPO
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    kl_coef: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Data
    reward_q_low: float = 0.05
    reward_q_high: float = 0.99
    
    # Paths
    rollout_globs: str = "runs/ppo_trajectory.json"
    save_dir: str = "./ft_out_ppo_7b_lora"
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        # Default LoRA targets for Qwen
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        # Load from environment
        self.base_model = os.getenv("BASE_MODEL", self.base_model)
        self.rollout_globs = os.getenv("ROLLOUT_GLOBS", self.rollout_globs)
        self.save_dir = os.getenv("SAVE_DIR", self.save_dir)
        
        if os.getenv("EPOCHS"):
            self.epochs = int(os.getenv("EPOCHS"))
        if os.getenv("LR"):
            self.lr = float(os.getenv("LR"))


# PPO Model

class PPOModel(nn.Module):
    """
    PPO model with policy network (LLM) and value head.
    """
    
    def __init__(self, lm: nn.Module, hidden_size: int):
        super().__init__()
        self.lm = lm
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Tuple of (logits, values)
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        
        # Get last token hidden state
        last_hidden = hidden_states[:, -1, :]
        values = self.value_head(last_hidden).squeeze(-1)
        
        return logits, values


# Data Loading

def is_clean_entry(entry: Dict) -> bool:
    """Filter for clean training samples"""
    if entry.get("used_fallback", False):
        return False
    
    parsed_from = entry.get("parsed_from", "")
    if parsed_from not in ["json", "actions_line", "actions_line_no_brackets", "any_brackets"]:
        return False
    
    action_unit = entry.get("action_unit", [])
    if not isinstance(action_unit, list) or len(action_unit) == 0:
        return False
    
    # Check action range [-1, 1]
    try:
        for a in action_unit:
            if abs(float(a)) > 1.05:
                return False
    except (ValueError, TypeError):
        return False
    
    return True


def load_clean_rollouts(paths: List[str]) -> List[Dict]:
    """Load and filter clean samples from rollout files"""
    all_samples = []
    
    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
        
        logger.info(f"Loading: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Mark last entry as done
        if data:
            data[-1]["done"] = True
        
        for entry in data:
            if not is_clean_entry(entry):
                continue
            
            prompt = entry.get("prompt", "")
            action_unit = entry.get("action_unit", [])
            reward = entry.get("reward", 0.0)
            done = entry.get("done", False)
            
            if not prompt or not action_unit:
                continue
            
            # Format answer
            answer = json.dumps(action_unit, ensure_ascii=False)
            
            all_samples.append({
                "prompt": prompt,
                "answer": answer,
                "reward": float(reward),
                "done": bool(done)
            })
    
    logger.info(f"Loaded {len(all_samples)} clean samples")
    return all_samples


class RolloutDataset(Dataset):
    """Dataset for rollout samples"""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        return {**self.samples[idx], "idx": idx}


def encode_one_sample(
    sample: Dict,
    tokenizer: AutoTokenizer
) -> Tuple[List[int], List[int], List[int]]:
    """
    Encode sample with chat template.
    
    Returns:
        Tuple of (ids_full, ids_prompt_only, labels)
    """
    prompt = sample["prompt"]
    answer = sample["answer"]
    
    # Full conversation
    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]
    
    ids_full = tokenizer.apply_chat_template(
        messages_full,
        add_generation_prompt=False,
        tokenize=True
    )
    
    # Prompt only (for labels masking)
    messages_prompt = [{"role": "user", "content": prompt}]
    ids_prompt_only = tokenizer.apply_chat_template(
        messages_prompt,
        add_generation_prompt=True,
        tokenize=True
    )
    
    # Create labels: mask prompt, keep answer
    labels = ids_full.copy()
    for i in range(len(ids_prompt_only)):
        labels[i] = -100
    
    return ids_full, ids_prompt_only, labels


def collate_chat(
    batch: List[Dict],
    tokenizer: AutoTokenizer,
    max_len: int = 1500
) -> Tuple[torch.Tensor, ...]:
    """Collate batch with padding"""
    
    input_ids_list = []
    labels_list = []
    prompt_ids_list = []
    rewards = []
    dones = []
    idxs = []
    
    for sample in batch:
        ids_full, ids_prompt, labels = encode_one_sample(sample, tokenizer)
        
        # Truncate if needed
        if len(ids_full) > max_len:
            ids_full = ids_full[:max_len]
            labels = labels[:max_len]
        if len(ids_prompt) > max_len:
            ids_prompt = ids_prompt[:max_len]
        
        input_ids_list.append(ids_full)
        labels_list.append(labels)
        prompt_ids_list.append(ids_prompt)
        rewards.append(sample["reward"])
        dones.append(sample["done"])
        idxs.append(sample["idx"])
    
    # Pad
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    max_input_len = max(len(x) for x in input_ids_list)
    max_prompt_len = max(len(x) for x in prompt_ids_list)
    
    # Pad input_ids and labels
    input_ids_padded = []
    attention_mask = []
    labels_padded = []
    
    for ids, labs in zip(input_ids_list, labels_list):
        pad_len = max_input_len - len(ids)
        
        input_ids_padded.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels_padded.append(labs + [-100] * pad_len)
    
    # Pad prompt_ids
    prompt_ids_padded = []
    for ids in prompt_ids_list:
        pad_len = max_prompt_len - len(ids)
        prompt_ids_padded.append(ids + [pad_id] * pad_len)
    
    return (
        torch.tensor(input_ids_padded, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels_padded, dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(dones, dtype=torch.bool),
        torch.tensor(prompt_ids_padded, dtype=torch.long),
        torch.tensor(idxs, dtype=torch.long)
    )


# PPO Utilities

def masked_token_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: AutoTokenizer
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute log probs and entropy for valid tokens only.
    
    Returns:
        Tuple of (log_probs_per_sample, entropy_per_sample, n_tokens_per_sample)
    """
    # Shift for autoregressive
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    # Valid token mask
    valid_mask = (shift_labels != -100) & (shift_mask != 0)
    
    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    lp_tok = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1).clamp(min=0)
    ).squeeze(-1)
    
    # Mask invalid
    lp_tok = lp_tok * valid_mask.float()
    
    # Entropy
    probs = torch.exp(log_probs)
    ent_tok = -(probs * log_probs).sum(dim=-1) * valid_mask.float()
    
    # Per-sample averages
    n_tokens = valid_mask.float().sum(dim=1).clamp(min=1)
    lp_per_sample = lp_tok.sum(dim=1) / n_tokens
    ent_per_sample = ent_tok.sum(dim=1) / n_tokens
    
    return lp_per_sample, ent_per_sample, n_tokens


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: (T,)
        values: (T,)
        dones: (T,)
        
    Returns:
        advantages: (T,)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1].item()
        
        delta = rewards[t].item() + gamma * (1 - dones[t].float().item()) * next_value - values[t].item()
        gae = delta + gamma * lam * (1 - dones[t].float().item()) * gae
        advantages[t] = gae
    
    return advantages



# Main Training

def main():
    """Main training loop"""
    
    # Load config
    config = FineTuneConfig()
    logger.info(f"Config: {config}")
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(config.save_dir, "finetune_config.json")
    with open(config_path, "w") as f:
        import dataclasses
        json.dump(dataclasses.asdict(config), f, indent=2)
    
    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    # Load data
    paths = glob.glob(config.rollout_globs)
    if not paths:
        raise ValueError(f"No files found: {config.rollout_globs}")
    
    samples = load_clean_rollouts(paths)
    
    if not samples:
        raise ValueError("No valid samples found")
    
    # Reward clipping
    rewards_all = torch.tensor([s["reward"] for s in samples])
    q_low = torch.quantile(rewards_all, config.reward_q_low)
    q_high = torch.quantile(rewards_all, config.reward_q_high)
    
    samples_filtered = [
        s for s in samples
        if q_low <= s["reward"] <= q_high
    ]
    
    logger.info(f"Filtered to {len(samples_filtered)} samples (reward in [{q_low:.2f}, {q_high:.2f}])")
    
    dataset = RolloutDataset(samples_filtered)
    
    # Load model
    logger.info(f"Loading model: {config.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True
    )
    
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    
    # Add LoRA
    if PEFT_AVAILABLE:
        logger.info("Adding LoRA adapters")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules
        )
        base = get_peft_model(base, lora_config)
        try:
            base.print_trainable_parameters()
        except:
            pass
    
    # Get hidden size
    hidden_size = getattr(base.config, "hidden_size", None) or getattr(base.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Cannot determine hidden_size from config")
    
    # Create PPO model
    policy_model = PPOModel(base, hidden_size).to(device)
    
    # Optimizer
    optimizer = AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=config.lr
    )
    

    # CRITICAL: Pre-compute old policy ONCE at start
    logger.info("Pre-computing initial old policy...")
    
    with torch.no_grad():
        all_values, all_old_lp, all_rewards, all_dones = [], [], [], []
        
        value_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_chat(b, tokenizer, config.max_seq_len)
        )
        
        policy_model.eval()
        
        for input_ids, attn_mask, labels, rewards, dones, _, _ in value_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            
            logits, values = policy_model(input_ids, attn_mask)
            old_lp, _, _ = masked_token_stats(logits, labels, attn_mask, tokenizer)
            
            all_values.append(values.float().detach().cpu())
            all_old_lp.append(old_lp.float().detach().cpu())
            all_rewards.append(rewards)
            all_dones.append(dones)
        
        values_vec = torch.cat(all_values, dim=0)
        old_lp_vec = torch.cat(all_old_lp, dim=0)
        rewards_vec = torch.cat(all_rewards, dim=0)
        dones_vec = torch.cat(all_dones, dim=0)
    
    logger.info(f"Pre-computed old policy: {len(old_lp_vec)} samples")
    
    # Build index mapping
    index_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_chat(b, tokenizer, config.max_seq_len)
    )
    
    ordered_indices = []
    for *_, idxs in index_loader:
        ordered_indices.extend(idxs.tolist())
    
    idx_to_pos = {idx: pos for pos, idx in enumerate(ordered_indices)}
    
    # Training Loop
    
    for epoch in range(config.epochs):
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{config.epochs}")
        logger.info('='*60)
        
        # CRITICAL: Re-compute old policy at epoch start and FREEZE it
        logger.info("Re-computing old policy for this epoch...")
        
        with torch.no_grad():
            all_values, all_old_lp = [], []
            policy_model.eval()
            
            for input_ids, attn_mask, labels, *_ in DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_chat(b, tokenizer, config.max_seq_len)
            ):
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels = labels.to(device)
                
                logits, values = policy_model(input_ids, attn_mask)
                old_lp, _, _ = masked_token_stats(logits, labels, attn_mask, tokenizer)
                
                all_values.append(values.float().detach().cpu())
                all_old_lp.append(old_lp.float().detach().cpu())
            
            values_vec = torch.cat(all_values, dim=0)
            old_lp_vec = torch.cat(all_old_lp, dim=0)
        
        # Compute advantages
        advantages = compute_gae(rewards_vec, values_vec, dones_vec, config.gamma, config.gae_lambda)
        
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-6)
        advantages = ((advantages - adv_mean) / adv_std).clamp(-5.0, 5.0)
        
        value_targets = advantages + values_vec
        
        # Move to device
        advantages = advantages.to(device)
        value_targets = value_targets.to(device)
        old_lp_vec_device = old_lp_vec.to(device)
        

        # Training with FROZEN old_lp_vec
        
        policy_model.train()
        
        total_loss = 0.0
        total_pl = 0.0
        total_vl = 0.0
        total_el = 0.0
        micro_steps = 0
        
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_chat(b, tokenizer, config.max_seq_len)
        )
        
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (input_ids, attn_mask, labels, _, _, _, idxs) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            idxs = idxs.to(device)
            
            # Get corresponding old policy values
            pos = torch.tensor(
                [idx_to_pos[i.item()] for i in idxs],
                device=device,
                dtype=torch.long
            )
            
            adv_b = advantages[pos]
            vt_b = value_targets[pos]
            old_lp_b = old_lp_vec_device[pos]  # From FROZEN old policy
            
            # Forward pass with current policy
            logits, values_pred = policy_model(input_ids, attn_mask)
            new_lp, new_ent, _ = masked_token_stats(logits, labels, attn_mask, tokenizer)
            

            # CRITICAL: Compute ratio with FROZEN old policy
            ratio = torch.exp(new_lp - old_lp_b)  # old_lp_b is FROZEN
            
            # PPO clipped objective
            obj1 = ratio * adv_b
            obj2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_b
            policy_loss = -torch.min(obj1, obj2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values_pred.float(), vt_b)
            
            # Entropy loss
            entropy_loss = -config.entropy_coef * new_ent.mean()
            
            # Total loss
            loss = (policy_loss + config.value_coef * value_loss + entropy_loss) / config.grad_accum
            
            loss.backward()
            
            micro_steps += 1
            total_loss += loss.item() * config.grad_accum
            total_pl += policy_loss.item()
            total_vl += value_loss.item()
            total_el += new_ent.mean().item()
            
            # Gradient step
            if micro_steps % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_model.parameters() if p.requires_grad],
                    1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        # Epoch stats
        n_batches = len(train_loader)
        elapsed = time.time() - t0
        
        logger.info(f"Epoch {epoch+1} completed in {elapsed:.1f}s")
        logger.info(f"  Avg Loss: {total_loss/n_batches:.4f}")
        logger.info(f"  Policy Loss: {total_pl/n_batches:.4f}")
        logger.info(f"  Value Loss: {total_vl/n_batches:.4f}")
        logger.info(f"  Entropy: {total_el/n_batches:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0 or epoch == config.epochs - 1:
            checkpoint_dir = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            policy_model.lm.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            torch.save({
                "epoch": epoch,
                "policy_model": policy_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, "trainer_state.pt"))
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Final save
    logger.info("Training complete! Saving final model...")
    
    policy_model.lm.save_pretrained(config.save_dir)
    tokenizer.save_pretrained(config.save_dir)
    
    torch.save(
        policy_model.state_dict(),
        os.path.join(config.save_dir, "policy_model.pt")
    )
    
    logger.info(f"Saved final model to {config.save_dir}")
    
    # Validation
    logger.info("\nRunning validation...")
    policy_model.eval()
    
    val_sample = dataset[0]
    prompt = val_sample["prompt"]
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = policy_model.lm.generate(
            inputs,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True
        )
    
    generated = tokenizer.decode(outputs[0, inputs.shape[1]:], skip_special_tokens=True)
    
    logger.info(f"Sample prompt: {prompt[:200]}...")
    logger.info(f"Generated: {generated}")


if __name__ == "__main__":
    main()
