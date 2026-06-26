"""
Best-of-N Self-Distillation with the Environment as Verifier
============================================================

This is the new core of the self-distillation approach. It replaces the
PPO-critic advantage path (``compute_advantage.py``) with something both simpler
and stronger: at each state the LLM proposes N candidate actions, and we pick the
best one by **actually evaluating each candidate in the BEAR environment** and
comparing their true rewards.

Why this is the right signal
----------------------------
The original problem was "BEAR's reward is confounded by environment difficulty,
so you can't tell a good action from an easy moment." Comparing N actions **at the
same state** dissolves that confound completely: the weather/occupancy is
identical across the candidates, so reward differences come purely from the
action. No critic, no PPO ceiling — the teacher is the real environment, so the
distilled policy can in principle approach the task optimum (not just PPO).

BEAR is a deterministic white-box simulator whose state is a handful of
attributes, so we can snapshot it, try a candidate action (counterfactually),
read the true reward, and restore — cheap and exact.

Output
------
A training dataset (same schema the AWR/SFT fine-tuner consumes): each entry has
the prompt, the best action (``action_unit``), and a clean same-state
``advantage`` = best_reward − mean_candidate_reward. Feed it to
``awr_finetune.py`` (advantage-weighted) or train plain SFT on the best actions.

Expert iteration: run this, fine-tune, then run it again with the improved model.

Usage
-----
    python core_modules/best_of_n_collect.py \
        --fewshot_json pipeline_output/02_few_shot_samples/few_shot_examples_structured.json \
        --output pipeline_output/03_llm_rollout/best_of_n_data.json \
        --n_candidates 6 --horizon 1 --max_steps 200 --episodes 1
"""

import os
import sys
import json
import argparse
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_PROJECT_ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment-as-verifier: snapshot / restore / counterfactual evaluation
# ---------------------------------------------------------------------------
def snapshot_env(env) -> Dict:
    """Capture every attribute BEAR's step() mutates, so it can be restored."""
    return {
        "epochs": env.epochs,
        "state": np.array(env.state).copy(),
        "X_new": np.array(env.X_new).copy(),
        "rewardsum": getattr(env, "rewardsum", 0.0),
        "Occupower": getattr(env, "Occupower", 0.0),
        "reward_breakdown": dict(getattr(env, "_reward_breakdown", {})),
        "n_state": len(getattr(env, "statelist", [])),
        "n_action": len(getattr(env, "actionlist", [])),
    }


def restore_env(env, snap: Dict) -> None:
    """Restore the environment to a previous snapshot."""
    env.epochs = snap["epochs"]
    env.state = np.array(snap["state"]).copy()
    env.X_new = np.array(snap["X_new"]).copy()
    if hasattr(env, "rewardsum"):
        env.rewardsum = snap["rewardsum"]
    if hasattr(env, "Occupower"):
        env.Occupower = snap["Occupower"]
    if hasattr(env, "_reward_breakdown"):
        env._reward_breakdown.clear()
        env._reward_breakdown.update(snap["reward_breakdown"])
    if hasattr(env, "statelist"):
        del env.statelist[snap["n_state"]:]
    if hasattr(env, "actionlist"):
        del env.actionlist[snap["n_action"]:]


def evaluate_action(env, action, horizon: int = 1) -> float:
    """Counterfactual: true cumulative reward of committing to ``action`` for
    ``horizon`` steps from the current state, then restore the environment.

    horizon=1 is the immediate true reward — already enough to rank candidates
    at the same state. horizon>1 is a "commit to this action for k steps"
    look-ahead (cheap proxy; no extra LLM calls).
    """
    snap = snapshot_env(env)
    total = 0.0
    a = np.asarray(action, dtype=np.float32)
    for _ in range(max(1, horizon)):
        step_ret = env.step(a)
        reward = step_ret[1]
        total += float(reward)
        done = bool(step_ret[2])
        if done:
            break
    restore_env(env, snap)
    return total


# ---------------------------------------------------------------------------
# Best-of-N selection (pure)
# ---------------------------------------------------------------------------
def select_best(candidates: List[List[float]], rewards: List[float]) -> Tuple[int, float, float]:
    """Pick the best candidate by true reward.

    Returns (best_index, best_reward, same_state_advantage) where the advantage
    is best_reward − mean(candidate rewards): a confound-free measure of how much
    better the chosen action was than the average proposal at this state.
    """
    r = np.asarray(rewards, dtype=np.float64)
    best_idx = int(np.argmax(r))
    advantage = float(r[best_idx] - r.mean())
    return best_idx, float(r[best_idx]), advantage


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BestOfNConfig:
    building: str = "OfficeSmall"
    climate: str = "Hot_Dry"
    location: str = "Tucson"
    target: float = 22.0
    data_root: str = ""

    max_steps: int = 200
    episodes: int = 1
    episode_offset_stride: int = 0

    n_candidates: int = 6       # actions sampled per state
    horizon: int = 4            # counterfactual look-ahead length (mitigates myopia)
    sample_temperature: float = 0.8  # >0 for candidate diversity

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 256
    fewshot_json: Optional[str] = None
    k_fewshot: int = 3
    fewshot_alpha: float = 0.6
    hist_keep: int = 6

    output: str = "./pipeline_output/03_llm_rollout/best_of_n_data.json"

    def __post_init__(self):
        if not self.data_root:
            self.data_root = os.path.abspath(os.path.join(_PROJECT_ROOT, "BEAR", "Data"))


# ---------------------------------------------------------------------------
# LLM candidate sampling (batched)
# ---------------------------------------------------------------------------
def sample_candidate_actions(prompt: str, n: int, n_actions: int, cfg: BestOfNConfig) -> List[List[float]]:
    """Sample N candidate action vectors from the cached LLM in one batched call."""
    import torch
    from llm_agent_colab import load_llm, parse_actions_with_validation

    tokenizer, model = load_llm(cfg.model_name, os.getenv("HF_TOKEN"))
    device = next(model.parameters()).device

    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        add_generation_prompt=True, return_tensors="pt",
    )
    # Newer transformers return a BatchEncoding (a UserDict, NOT a dict subclass)
    # instead of a bare tensor. Normalize to the input_ids tensor.
    input_ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
    input_ids = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # Decode candidates in sub-batches. Default is large enough to do all N at
    # once on a big GPU (A100); lower BON_MAX_PARALLEL on a small GPU to bound
    # activation memory.
    max_parallel = int(os.getenv("BON_MAX_PARALLEL", "8"))
    out_seqs = []
    remaining = n
    with torch.no_grad():
        while remaining > 0:
            k = min(max_parallel, remaining)
            gen_kwargs = dict(
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True, temperature=max(cfg.sample_temperature, 1e-3),
                top_p=0.9, top_k=50, num_return_sequences=k,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            out = model.generate(input_ids, **gen_kwargs)
            out_seqs.extend(list(out))
            remaining -= k

    actions = []
    for seq in out_seqs:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        a, _ = parse_actions_with_validation(text, n_actions)
        if a is not None:
            actions.append([float(np.clip(x, -1.0, 1.0)) for x in a])
    return actions


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------
def _n_zones(obs) -> int:
    return max(1, (len(obs) - 2) // 3)


def collect(cfg: BestOfNConfig) -> List[Dict]:
    from BEAR.Utils.utils_building import ParameterGenerator
    from BEAR.Env.env_building import BuildingEnvReal
    from prompt_builder_control import build_prompt
    from few_shot_auto import (load_examples, select_examples, format_few_shot_block,
                               inject_few_shot, SelectionConfig)

    ex_dataset = None
    if cfg.fewshot_json and os.path.exists(cfg.fewshot_json):
        ex_dataset = load_examples(cfg.fewshot_json)
        logger.info(f"Loaded {len(ex_dataset)} few-shot examples")

    param = ParameterGenerator(cfg.building, cfg.climate, cfg.location,
                               root=cfg.data_root, target=cfg.target)

    all_entries: List[Dict] = []
    for ep in range(cfg.episodes):
        env = BuildingEnvReal(param)
        reset_ret = env.reset()
        obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
        # Optional weather-window diversity (deterministic, like the rollout)
        offset = (ep * cfg.episode_offset_stride) % (env.length_of_weather - 1) \
            if cfg.episode_offset_stride > 0 else 0
        if offset > 0:
            env.epochs = offset
        obs = np.asarray(env.state).flatten().tolist()

        history = deque(maxlen=cfg.hist_keep)
        ep_improvements = []

        for step in range(cfg.max_steps):
            n = _n_zones(obs)
            prompt = build_prompt(obs=obs, building=cfg.building, location=cfg.location,
                                  climate=cfg.climate, target=cfg.target,
                                  round_idx=step + 1, history=list(history))
            if ex_dataset:
                try:
                    examples = select_examples(ex_dataset, current_obs=obs,
                                               config=SelectionConfig(k=cfg.k_fewshot, alpha=cfg.fewshot_alpha),
                                               building=cfg.building, climate=cfg.climate, location=cfg.location)
                    prompt = inject_few_shot(prompt, format_few_shot_block(examples, target=cfg.target, n=n))
                except Exception as e:
                    logger.warning(f"few-shot injection failed: {e}")

            # 1) Propose N candidates
            candidates = sample_candidate_actions(prompt, cfg.n_candidates, n, cfg)
            if not candidates:
                candidates = [[0.0] * n]  # degenerate fallback

            # 2) Score each candidate by its TRUE reward at THIS state
            rewards = [evaluate_action(env, a, cfg.horizon) for a in candidates]

            # 3) Pick the best; advantage is confound-free (same state)
            best_idx, best_reward, advantage = select_best(candidates, rewards)
            best_action = candidates[best_idx]
            ep_improvements.append(best_reward - float(np.mean(rewards)))

            # 4) Advance the REAL environment with the best action (expert-iteration trajectory)
            snap_obs = obs
            step_ret = env.step(np.asarray(best_action, dtype=np.float32))
            obs_next = np.asarray(step_ret[0]).flatten().tolist()
            real_reward = float(step_ret[1])
            done = bool(step_ret[2])

            history.append({"step": step + 1, "action": best_action, "reward": real_reward,
                            "env_temp": float(snap_obs[n]) if len(snap_obs) > n else 0.0,
                            "obs_before": snap_obs, "obs_after": obs_next})

            all_entries.append({
                "episode": ep, "step": step,
                "prompt": prompt,
                "action_unit": best_action,
                "advantage": advantage,            # same-state, confound-free
                "reward": real_reward,
                "n_candidates": len(candidates),
                "candidate_rewards": rewards,
                "parsed_from": "best_of_n",
                "used_fallback": False,
                "obs": snap_obs, "next_obs": obs_next,
                "done": done,
            })

            obs = obs_next
            if (step + 1) % 20 == 0:
                logger.info(f"[ep {ep+1}] step {step+1}/{cfg.max_steps} | "
                            f"best_reward={best_reward:.3f} adv={advantage:.3f} "
                            f"(N={len(candidates)})")
            if done:
                break

        logger.info(f"[ep {ep+1}] done: {len([e for e in all_entries if e['episode']==ep])} states | "
                    f"mean best-vs-avg improvement={np.mean(ep_improvements):.3f}")

    os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
    with open(cfg.output, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Saved {len(all_entries)} best-of-N training entries to {cfg.output}")
    return all_entries


def main():
    p = argparse.ArgumentParser(description="Best-of-N self-distillation (environment as verifier)")
    p.add_argument("--building", default="OfficeSmall")
    p.add_argument("--climate", default="Hot_Dry")
    p.add_argument("--location", default="Tucson")
    p.add_argument("--target", type=float, default=22.0)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--episode_offset_stride", type=int, default=0)
    p.add_argument("--n_candidates", type=int, default=6)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--sample_temperature", type=float, default=0.8)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--fewshot_json", default=None)
    p.add_argument("--output", default="./pipeline_output/03_llm_rollout/best_of_n_data.json")
    args = p.parse_args()

    cfg = BestOfNConfig(
        building=args.building, climate=args.climate, location=args.location, target=args.target,
        max_steps=args.max_steps, episodes=args.episodes,
        episode_offset_stride=args.episode_offset_stride,
        n_candidates=args.n_candidates, horizon=args.horizon,
        sample_temperature=args.sample_temperature, model_name=args.model_name,
        fewshot_json=args.fewshot_json, output=args.output,
    )
    collect(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
