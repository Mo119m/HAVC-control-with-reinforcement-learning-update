"""
Critic-based Advantage Computation
==================================

Why this exists
---------------
Throughout the old pipeline, "how good was this step?" was answered with the
**raw per-step reward**. But BEAR's reward is dominated by how *hard the current
conditions are* (outside temperature, irradiance, occupancy), not by how good the
action was. So selecting / weighting by raw reward mostly selects easy moments,
not good control. See METHODOLOGY_REVIEW.md (root cause #1).

This module computes a **decoupled** signal: the one-step temporal-difference
advantage using the value function (critic) of the trained PPO policy:

    A(s, a) = r + gamma * (1 - done) * V(s') - V(s)

V(s) is a state-difficulty baseline (the expected return of a competent policy
from state s). Subtracting it removes the environment-difficulty confound, so A
measures whether *this transition* did better or worse than expected from s.

Caveat (documented on purpose): PPO's V was trained under PPO's own policy, so A
is the advantage of "take the LLM's action, then continue as PPO would". That is
a perfectly good baseline subtraction for ranking/weighting LLM transitions; it
is not an unbiased advantage of the LLM's own policy. Good enough — and far
better than raw reward — for self-distillation filtering, AWR weighting and
few-shot ranking.

Outputs
-------
Writes an augmented copy of the rollout JSON with two extra fields per entry:
``value`` (V(s)) and ``advantage`` (A). Also prints diagnostics, including the
correlation between reward and advantage and the overlap between "top-k by
reward" and "top-k by advantage" (illustrating how confounded the raw signal is).

Usage
-----
    python core_modules/compute_advantage.py \
        --rollout pipeline_output/03_llm_rollout/llm_rollout.json \
        --ppo_model pipeline_output/01_ppo_training/ppo_final.zip \
        --output pipeline_output/03_llm_rollout/llm_rollout_adv.json \
        --gamma 0.99
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Optional

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-numpy advantage math (unit-testable without torch / sb3)
# ---------------------------------------------------------------------------
def td_advantage(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """One-step TD advantage: r + gamma*(1-done)*V(s') - V(s).

    Order-independent and per-transition, so it is robust even when entries are
    filtered or shuffled (unlike GAE, which the old fine-tuner ran over a broken,
    reward-filtered sequence). All inputs are 1-D arrays of equal length.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    next_values = np.asarray(next_values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    return rewards + gamma * (1.0 - dones) * next_values - values


def normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Zero-mean/unit-std normalization (for AWR temperature stability)."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + eps)


# ---------------------------------------------------------------------------
# PPO critic
# ---------------------------------------------------------------------------
class PPOCritic:
    """Thin wrapper exposing V(s) from a trained Stable-Baselines3 PPO model."""

    def __init__(self, model_path: str):
        from stable_baselines3 import PPO
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPO model not found: {model_path}")
        logger.info(f"Loading PPO critic from: {model_path}")
        self.model = PPO.load(model_path, device="auto")

    def values(self, obs_batch: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """Return V(s) for a batch of observations, shape (N,)."""
        import torch

        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(obs_batch), chunk):
                part = obs_batch[i:i + chunk]
                obs_tensor, _ = self.model.policy.obs_to_tensor(part)
                v = self.model.policy.predict_values(obs_tensor)
                out.append(v.detach().cpu().numpy().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------------------
# Rollout processing
# ---------------------------------------------------------------------------
def _extract_arrays(entries: List[Dict]):
    """Pull obs / next_obs / reward / done into parallel arrays.

    Returns (obs, next_obs, rewards, dones, valid_idx) where valid_idx maps rows
    back to entries that had usable obs/next_obs (others are skipped).
    """
    obs_list, next_list, rewards, dones, valid_idx = [], [], [], [], []
    obs_dim = None
    for i, e in enumerate(entries):
        obs = e.get("obs")
        nxt = e.get("next_obs")
        if obs is None or nxt is None:
            continue
        obs = list(np.asarray(obs).flatten())
        nxt = list(np.asarray(nxt).flatten())
        if obs_dim is None:
            obs_dim = len(obs)
        if len(obs) != obs_dim or len(nxt) != obs_dim:
            continue
        obs_list.append(obs)
        next_list.append(nxt)
        rewards.append(float(e.get("reward", 0.0)))
        dones.append(bool(e.get("done", False)))
        valid_idx.append(i)
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(next_list, dtype=np.float32),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(dones, dtype=np.float64),
        valid_idx,
    )


def compute_advantages_for_rollout(
    rollout_path: str,
    ppo_model_path: str,
    output_path: Optional[str] = None,
    gamma: float = 0.99,
) -> List[Dict]:
    """Augment a rollout JSON with critic value V(s) and TD advantage A.

    Returns the list of entries (with ``value`` and ``advantage`` added to those
    that had usable obs/next_obs).
    """
    with open(rollout_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    logger.info(f"Loaded {len(entries)} rollout entries from {rollout_path}")

    obs, next_obs, rewards, dones, valid_idx = _extract_arrays(entries)
    if len(valid_idx) == 0:
        raise ValueError("No entries with usable obs/next_obs found")
    logger.info(f"{len(valid_idx)}/{len(entries)} entries have usable obs/next_obs")

    critic = PPOCritic(ppo_model_path)
    values = critic.values(obs)
    next_values = critic.values(next_obs)
    advantages = td_advantage(rewards, values, next_values, dones, gamma=gamma)

    # Write back onto the entries
    for row, ent_i in enumerate(valid_idx):
        entries[ent_i]["value"] = float(values[row])
        entries[ent_i]["advantage"] = float(advantages[row])

    _print_diagnostics(rewards, advantages)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved augmented rollout: {output_path}")

    return entries


def _print_diagnostics(rewards: np.ndarray, advantages: np.ndarray, k_frac: float = 0.25):
    """Show how decoupled the advantage signal is from raw reward."""
    corr = float(np.corrcoef(rewards, advantages)[0, 1]) if len(rewards) > 1 else float("nan")

    n = len(rewards)
    k = max(1, int(n * k_frac))
    top_reward = set(np.argsort(rewards)[-k:].tolist())
    top_adv = set(np.argsort(advantages)[-k:].tolist())
    overlap = len(top_reward & top_adv) / k

    logger.info("=" * 60)
    logger.info("Advantage diagnostics")
    logger.info("=" * 60)
    logger.info(f"  N transitions:        {n}")
    logger.info(f"  reward   mean/std:    {rewards.mean():.3f} / {rewards.std():.3f}")
    logger.info(f"  advantage mean/std:   {advantages.mean():.3f} / {advantages.std():.3f}")
    logger.info(f"  corr(reward, adv):    {corr:.3f}")
    logger.info(f"  top-{int(k_frac*100)}% overlap (reward vs adv): {overlap:.2f}")
    logger.info("  -> low overlap == raw-reward selection was picking different "
                "(easier) steps than advantage does")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compute critic-based advantages for a rollout")
    parser.add_argument("--rollout", required=True, help="Path to rollout JSON (needs obs/next_obs)")
    parser.add_argument("--ppo_model", required=True, help="Path to trained PPO model (.zip)")
    parser.add_argument("--output", default=None, help="Output path for augmented rollout JSON")
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    out = args.output
    if out is None:
        base, ext = os.path.splitext(args.rollout)
        out = f"{base}_adv{ext}"

    compute_advantages_for_rollout(args.rollout, args.ppo_model, out, gamma=args.gamma)
    return 0


if __name__ == "__main__":
    sys.exit(main())
