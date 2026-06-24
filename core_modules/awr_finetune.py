"""
Advantage-Weighted Regression (AWR) Fine-tuning
===============================================

Replacement for the original unsound offline-"PPO" fine-tuner (since removed).

What was wrong before (see METHODOLOGY_REVIEW.md, root cause #2)
---------------------------------------------------------------
- GAE was run over a reward-filtered, then shuffled dataset, so temporal
  adjacency was destroyed and advantages were noise.
- The value head was randomly initialized and trained on ~100-200 samples, so
  V(s) (and therefore GAE) was noise on top of noise.
- ``old_lp`` was recomputed from the *updated* policy each epoch, so the PPO
  ratio degenerated and clipping did nothing.
- Advantages were normalized to zero mean *after* filtering to "good" samples,
  so ~half of the curated good samples were pushed down — fighting the
  self-distillation intent.

What this does instead
----------------------
Advantage-Weighted Regression: a single, stable, supervised objective.

    loss = sum_i w_i * NLL(action_i | state_i) / sum_i w_i
    w_i  = clip(exp(A_norm_i / beta), 0, max_weight)

where A is the **critic-based advantage** produced by ``compute_advantage.py``
(decoupled from environment difficulty) and A_norm is its z-score. High-advantage
transitions are imitated strongly; low-advantage ones are down-weighted. This is
"self-distillation done right": the LLM imitates its own *better-than-baseline*
behaviour, with no value head, no GAE, no importance ratio to go wrong.

Input
-----
A rollout JSON that already has an ``advantage`` field on each entry (run
``compute_advantage.py`` first). If ``advantage`` is absent, the script can fall
back to raw reward with ``--allow_reward_fallback`` (not recommended).

Env vars (for pipeline integration) mirror the old finetuner: BASE_MODEL,
ROLLOUT_GLOBS, SAVE_DIR, EPOCHS, LR. Plus AWR_BETA, ADV_KEEP_PERCENTILE.

Usage
-----
    python core_modules/awr_finetune.py \
        --rollout pipeline_output/03_llm_rollout/llm_rollout_adv.json \
        --save_dir pipeline_output/04_finetuning \
        --beta 1.0 --adv_keep_percentile 0.5
"""

import os
import sys
import glob
import json
import time
import math
import argparse
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-numpy AWR weight math (unit-testable without torch)
# ---------------------------------------------------------------------------
def awr_weights(
    advantages: np.ndarray,
    beta: float = 1.0,
    max_weight: float = 20.0,
    normalize_adv: bool = True,
) -> np.ndarray:
    """Compute AWR sample weights w = clip(exp(A_norm / beta), 0, max_weight).

    Args:
        advantages: 1-D array of per-sample advantages.
        beta: temperature. Larger beta -> flatter weights (closer to uniform SFT);
              smaller beta -> sharper preference for high-advantage samples.
        max_weight: clip ceiling to prevent a few samples from dominating.
        normalize_adv: z-score the advantages first (scale invariance).
    """
    adv = np.asarray(advantages, dtype=np.float64)
    if normalize_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    w = np.exp(adv / max(beta, 1e-6))
    return np.clip(w, 0.0, max_weight)


def advantage_keep_mask(advantages: np.ndarray, keep_percentile: float) -> np.ndarray:
    """Boolean mask keeping samples with advantage >= the given percentile.

    keep_percentile=0.5 keeps the better half; 0.0 keeps everything.
    """
    adv = np.asarray(advantages, dtype=np.float64)
    if keep_percentile <= 0.0:
        return np.ones(len(adv), dtype=bool)
    threshold = np.quantile(adv, keep_percentile)
    return adv >= threshold


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AWRConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    rollout_globs: str = "pipeline_output/03_llm_rollout/llm_rollout_adv.json"
    save_dir: str = "./pipeline_output/04_finetuning"

    epochs: int = 3
    lr: float = 1e-5
    batch_size: int = 1
    grad_accum: int = 8
    max_seq_len: int = 1500

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])

    # AWR
    beta: float = 1.0
    max_weight: float = 20.0
    adv_keep_percentile: float = 0.5      # drop the worse half by advantage
    allow_reward_fallback: bool = False   # use raw reward if advantage missing

    def __post_init__(self):
        self.base_model = os.getenv("BASE_MODEL", self.base_model)
        self.rollout_globs = os.getenv("ROLLOUT_GLOBS", self.rollout_globs)
        self.save_dir = os.getenv("SAVE_DIR", self.save_dir)
        if os.getenv("EPOCHS"):
            self.epochs = int(os.getenv("EPOCHS"))
        if os.getenv("LR"):
            self.lr = float(os.getenv("LR"))
        if os.getenv("AWR_BETA"):
            self.beta = float(os.getenv("AWR_BETA"))
        if os.getenv("ADV_KEEP_PERCENTILE"):
            self.adv_keep_percentile = float(os.getenv("ADV_KEEP_PERCENTILE"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def is_clean_entry(entry: Dict) -> bool:
    """Reject parse-failures, fallbacks and out-of-range actions."""
    if entry.get("used_fallback", False):
        return False
    if entry.get("parsed_from", "") not in (
        "json", "actions_line", "actions_line_no_brackets", "any_brackets", "last_json",
        "best_of_n",  # entries chosen by the environment-as-verifier collector
    ):
        return False
    actions = entry.get("action_unit", [])
    if not isinstance(actions, list) or len(actions) == 0:
        return False
    try:
        return all(abs(float(a)) <= 1.05 for a in actions)
    except (ValueError, TypeError):
        return False


def load_awr_samples(paths: List[str], allow_reward_fallback: bool) -> List[Dict]:
    """Load clean samples carrying an advantage (or reward fallback) signal."""
    samples: List[Dict] = []
    used_fallback = False
    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for e in data:
            if not is_clean_entry(e):
                continue
            prompt, actions = e.get("prompt", ""), e.get("action_unit", [])
            if not prompt or not actions:
                continue
            if "advantage" in e and e["advantage"] is not None:
                signal = float(e["advantage"])
            elif allow_reward_fallback:
                signal = float(e.get("reward", 0.0))
                used_fallback = True
            else:
                continue
            samples.append({
                "prompt": prompt,
                "answer": json.dumps(actions, ensure_ascii=False),
                "advantage": signal,
            })
    if used_fallback:
        logger.warning("⚠️  No 'advantage' field found — fell back to RAW REWARD. "
                       "Run compute_advantage.py first for the proper signal.")
    logger.info(f"Loaded {len(samples)} clean AWR samples")
    return samples


# ---------------------------------------------------------------------------
# Tokenization (self-contained; mirrors the chat-template masking used before)
# ---------------------------------------------------------------------------
def encode_sample(sample: Dict, tokenizer) -> Tuple[List[int], List[int]]:
    """Return (input_ids, labels) with the prompt masked (-100) and answer kept."""
    messages_full = [
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["answer"]},
    ]
    ids_full = tokenizer.apply_chat_template(messages_full, add_generation_prompt=False, tokenize=True)
    ids_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample["prompt"]}], add_generation_prompt=True, tokenize=True
    )
    labels = list(ids_full)
    for i in range(min(len(ids_prompt), len(labels))):
        labels[i] = -100
    return ids_full, labels


def collate(batch: List[Dict], tokenizer, max_len: int):
    import torch
    ids_list, labels_list, weights = [], [], []
    for s in batch:
        ids, labels = encode_sample(s, tokenizer)
        if len(ids) > max_len:
            ids, labels = ids[:max_len], labels[:max_len]
        ids_list.append(ids)
        labels_list.append(labels)
        weights.append(s["weight"])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    maxL = max(len(x) for x in ids_list)
    input_ids, attn, labels_pad = [], [], []
    for ids, labs in zip(ids_list, labels_list):
        pad = maxL - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        labels_pad.append(labs + [-100] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        torch.tensor(labels_pad, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


def weighted_nll(logits, labels, weights) -> "torch.Tensor":
    """Per-sample mean NLL over answer tokens, weighted and averaged."""
    import torch
    import torch.nn.functional as F
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = (shift_labels != -100)
    logp = F.log_softmax(shift_logits.float(), dim=-1)
    tok_logp = torch.gather(logp, -1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * valid.float()
    n_tok = valid.float().sum(dim=1).clamp(min=1)
    nll_per_sample = -(tok_logp.sum(dim=1) / n_tok)            # (B,)
    w = weights.to(nll_per_sample.device)
    return (w * nll_per_sample).sum() / w.sum().clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def plot_history(history: Dict[str, List], save_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.get("loss", []), color="crimson", linewidth=1.5)
        ax.set_xlabel("Gradient step")
        ax.set_ylabel("Weighted NLL loss")
        ax.set_title("AWR Fine-tuning Loss")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = Path(save_dir) / "awr_training_curve.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        with open(Path(save_dir) / "awr_history.json", "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
        logger.info(f"📊 Saved AWR training curve: {p}")
    except Exception as e:
        logger.warning(f"Failed to plot AWR history: {e}")


def train(config: AWRConfig):
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, "awr_config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Data + weights -----------------------------------------------------
    paths = glob.glob(config.rollout_globs)
    if not paths:
        raise ValueError(f"No files matched: {config.rollout_globs}")
    samples = load_awr_samples(paths, config.allow_reward_fallback)
    if not samples:
        raise ValueError("No usable samples (need an 'advantage' field; run compute_advantage.py)")

    adv = np.array([s["advantage"] for s in samples])
    keep = advantage_keep_mask(adv, config.adv_keep_percentile)
    samples = [s for s, k in zip(samples, keep) if k]
    adv = adv[keep]
    weights = awr_weights(adv, beta=config.beta, max_weight=config.max_weight)
    for s, w in zip(samples, weights):
        s["weight"] = float(w)
    logger.info(f"After advantage filtering (keep>={config.adv_keep_percentile:.2f}): "
                f"{len(samples)} samples | weight mean={weights.mean():.3f} "
                f"max={weights.max():.3f} min={weights.min():.3f}")

    # ---- Model + LoRA -------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    try:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", inference_mode=False,
            r=config.lora_r, lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout, target_modules=config.lora_target_modules,
        ))
        model.print_trainable_parameters()
    except ImportError:
        logger.warning("peft not available; training full model (not recommended)")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr)

    loader_factory = lambda shuffle: DataLoader(
        samples, batch_size=config.batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate(b, tokenizer, config.max_seq_len),
    )

    history = {"loss": []}
    logger.info("=" * 60)
    logger.info(f"AWR training: {config.epochs} epochs, {len(samples)} samples, "
                f"beta={config.beta}")
    logger.info("=" * 60)

    model.train()
    for epoch in range(config.epochs):
        t0 = time.time()
        loader = loader_factory(shuffle=True)
        optimizer.zero_grad(set_to_none=True)
        running, micro = 0.0, 0
        n_grad = (len(loader) + config.grad_accum - 1) // config.grad_accum

        for bi, (input_ids, attn, labels, w) in enumerate(loader):
            input_ids, attn, labels = input_ids.to(device), attn.to(device), labels.to(device)
            out = model(input_ids=input_ids, attention_mask=attn)
            loss = weighted_nll(out.logits, labels, w) / config.grad_accum
            loss.backward()
            running += loss.item() * config.grad_accum
            micro += 1
            if micro % config.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step = micro // config.grad_accum
                avg = running / micro
                history["loss"].append(avg)
                logger.info(f"  epoch {epoch+1}/{config.epochs} step {step}/{n_grad} | loss {avg:.4f}")
                sys.stdout.flush()

        logger.info(f"Epoch {epoch+1} done in {time.time()-t0:.1f}s | avg loss {running/max(1,micro):.4f}")

    # ---- Save ---------------------------------------------------------------
    logger.info("Saving final model...")
    model.save_pretrained(config.save_dir)
    tokenizer.save_pretrained(config.save_dir)
    # Mirror the path the old pipeline / evaluator expect
    final_dir = os.path.join(config.save_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    plot_history(history, config.save_dir)
    logger.info(f"✅ AWR fine-tuning complete. Adapter saved to: {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Advantage-Weighted Regression fine-tuning")
    parser.add_argument("--rollout", default=None, help="Advantage-augmented rollout JSON (glob ok)")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--adv_keep_percentile", type=float, default=None)
    parser.add_argument("--allow_reward_fallback", action="store_true")
    args = parser.parse_args()

    cfg = AWRConfig()
    if args.rollout: cfg.rollout_globs = args.rollout
    if args.save_dir: cfg.save_dir = args.save_dir
    if args.base_model: cfg.base_model = args.base_model
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.lr is not None: cfg.lr = args.lr
    if args.beta is not None: cfg.beta = args.beta
    if args.adv_keep_percentile is not None: cfg.adv_keep_percentile = args.adv_keep_percentile
    if args.allow_reward_fallback: cfg.allow_reward_fallback = True

    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
