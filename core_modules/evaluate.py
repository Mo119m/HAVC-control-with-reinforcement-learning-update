"""
Controlled Evaluation Harness for HVAC Controllers
==================================================

Why this exists
---------------
The previous comparison (``draw_reward.py``) overlaid reward curves collected on
*different* episodes / weather windows / seeds, so it could not support any claim
like "fine-tuned LLM > base LLM". This module fixes that by running every
controller on the **exact same** deterministic episode(s) and reporting
comparable metrics.

The BEAR environment is fully deterministic: ``reset()`` always starts at
``epochs=0`` with initial temperature = target, and weather is read from the EPW
file in order. So once policy stochasticity is fixed (PPO ``deterministic=True``,
LLM ``temperature=0``), all controllers see a byte-identical environment.

Controllers compared
--------------------
- ``zero``      : HVAC off (do-nothing) sanity baseline.
- ``rule``      : proportional thermostat baseline (action ∝ target − room temp).
- ``ppo``       : trained Stable-Baselines3 PPO policy.
- ``llm``       : base instruction LLM (optionally with few-shot).
- ``llm_ft``    : fine-tuned LLM (base + LoRA adapter).

Metrics (per controller, averaged over seeds for stochastic controllers)
------------------------------------------------------------------------
- total_return          : sum of per-step reward (primary, higher is better)
- mean_step_reward
- comfort_violation_rate : fraction of (step, zone) with room temp outside [18, 22]
- mean_abs_dev_target    : mean |room temp − target|
- energy_proxy           : mean L1 norm of action (actuation magnitude)
- parse_fail_rate        : LLM only

Usage
-----
    python core_modules/evaluate.py \
        --controllers rule ppo llm \
        --ppo_model pipeline_output/01_ppo_training/ppo_final.zip \
        --fewshot_json pipeline_output/02_few_shot_samples/few_shot_examples_structured.json \
        --max_steps 200 --llm_seeds 3 \
        --out_dir pipeline_output/06_eval

Heavy imports (torch / sb3 / BEAR) are loaded lazily so this file can be imported
and ``--help`` works without a GPU or the full dependency stack installed.
"""

import os
import sys
import json
import argparse
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any

import numpy as np

# Make project root importable (so BEAR.* and core_modules.* resolve)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Also add core_modules itself for the flat-style imports used by rollout code
_CORE = os.path.dirname(__file__)
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Comfort band used by BEAR's default reward function (reward_functions.py)
COMFORT_LOW = 18.0
COMFORT_HIGH = 22.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    building: str = "OfficeSmall"
    climate: str = "Hot_Dry"
    location: str = "Tucson"
    target: float = 22.0
    data_root: str = ""

    max_steps: int = 200
    # Episode "offsets": start the weather index at different points to evaluate on
    # several distinct (but still deterministic & shared-across-controllers) windows.
    episode_offsets: List[int] = field(default_factory=lambda: [0])

    # LLM
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 256
    llm_temperature: float = 0.0      # 0 => greedy/deterministic
    llm_seeds: int = 1                # >1 only meaningful if temperature > 0
    fewshot_json: Optional[str] = None
    k_fewshot: int = 3
    fewshot_alpha: float = 0.6
    hist_keep: int = 6
    hist_lines: int = 3

    def __post_init__(self):
        if not self.data_root:
            self.data_root = os.path.abspath(os.path.join(_PROJECT_ROOT, "BEAR", "Data"))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def make_env(cfg: EvalConfig):
    """Create a single (non-vectorized) deterministic BEAR environment."""
    from BEAR.Utils.utils_building import ParameterGenerator
    from BEAR.Env.env_building import BuildingEnvReal

    param = ParameterGenerator(
        cfg.building, cfg.climate, cfg.location,
        root=cfg.data_root, target=cfg.target,
    )
    return BuildingEnvReal(param)


def _reset_env(env, offset: int = 0):
    """Reset env deterministically and optionally advance the weather index.

    ``offset`` lets us evaluate on different weather windows while remaining
    identical across controllers. We advance by stepping the *weather pointer*
    only (env.epochs) and rebuilding the observation, so the starting room
    temperatures stay at target for every offset.
    """
    reset_ret = env.reset()
    obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

    if offset > 0:
        # Advance the deterministic weather pointer and rebuild the state vector
        # to match. This mirrors how reset() constructs ``self.state``.
        env.epochs = int(offset) % (env.length_of_weather - 1)
        T_initial = env.X_new
        ghi_repeated = np.full(T_initial.shape, env.ghi[env.epochs])
        occ_repeated = np.full(T_initial.shape, env.Occupower / 1000)
        env.state = np.concatenate((
            T_initial,
            env.OutTemp[env.epochs].reshape(-1),
            ghi_repeated,
            env.GroundTemp[env.epochs].reshape(-1),
            occ_repeated,
        ), axis=0)
        obs = env.state

    return np.asarray(obs).flatten().tolist()


def _env_step(env, action):
    step_ret = env.step(np.asarray(action, dtype=np.float32))
    if len(step_ret) == 5:
        obs_next, reward, terminated, truncated, info = step_ret
        done = bool(terminated or truncated)
    else:
        obs_next, reward, done, info = step_ret
    return np.asarray(obs_next).flatten().tolist(), float(reward), done, (info or {})


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
class Controller:
    """Base controller interface: map observation -> action list in [-1, 1]."""

    name = "base"
    is_stochastic = False

    def reset(self):
        """Called at the start of each episode."""

    def act(self, obs: List[float], step: int) -> List[float]:
        raise NotImplementedError

    @property
    def extra_stats(self) -> Dict[str, float]:
        return {}


def _n_zones(obs: List[float]) -> int:
    return max(1, (len(obs) - 2) // 3)


class ZeroController(Controller):
    name = "zero"

    def act(self, obs, step):
        return [0.0] * _n_zones(obs)


class RuleBasedController(Controller):
    """Proportional thermostat: drive each room toward target.

    action_i = clip(gain * (target - room_temp_i), -1, 1)
    A 1/gain °C error saturates the actuator. This is the "is the LLM even
    beating a dumb thermostat?" baseline.
    """

    name = "rule"

    def __init__(self, target: float = 22.0, gain: float = 0.5):
        self.target = target
        self.gain = gain

    def act(self, obs, step):
        n = _n_zones(obs)
        temps = obs[:n]
        return [float(np.clip(self.gain * (self.target - t), -1.0, 1.0)) for t in temps]


class PPOController(Controller):
    name = "ppo"

    def __init__(self, model_path: str):
        from stable_baselines3 import PPO
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPO model not found: {model_path}")
        logger.info(f"Loading PPO model: {model_path}")
        self.model = PPO.load(model_path, device="auto")

    def act(self, obs, step):
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        action = np.asarray(action).flatten()
        return [float(np.clip(a, -1.0, 1.0)) for a in action]


class LLMController(Controller):
    """Base or fine-tuned instruction LLM, optionally with few-shot examples.

    Mirrors the prompt construction used in ``rollout_fewshot_version.py`` so the
    evaluation matches how the model was actually used.
    """

    name = "llm"
    is_stochastic = True

    def __init__(self, cfg: EvalConfig, adapter_path: Optional[str] = None, name: Optional[str] = None):
        self.cfg = cfg
        self.adapter_path = adapter_path
        if name:
            self.name = name
        self.is_stochastic = cfg.llm_temperature > 0.0

        # Lazy module refs
        from collections import deque
        self._deque = deque
        self.history = deque(maxlen=cfg.hist_keep)

        # Prime the shared model cache (handles base + optional LoRA adapter)
        self._prime_model()

        # Few-shot dataset
        self.ex_dataset = None
        if cfg.fewshot_json and os.path.exists(cfg.fewshot_json):
            from few_shot_auto import load_examples
            self.ex_dataset = load_examples(cfg.fewshot_json)
            logger.info(f"[{self.name}] loaded {len(self.ex_dataset)} few-shot examples")

        self._parse_fail = 0
        self._calls = 0

    def _prime_model(self):
        """Load base (and adapter) once and install into llm_agent_colab's cache."""
        import llm_agent_colab as agent
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tok = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, trust_remote_code=True,
            device_map=device_map, torch_dtype=dtype,
        )

        if self.adapter_path:
            from peft import PeftModel
            logger.info(f"[{self.name}] loading LoRA adapter: {self.adapter_path}")
            model = PeftModel.from_pretrained(model, self.adapter_path)

        model.eval()
        # Install into the global cache so call_llm() reuses it
        agent._TOKENIZER = tok
        agent._MODEL = model

    def reset(self):
        self.history = self._deque(maxlen=self.cfg.hist_keep)

    def act(self, obs, step):
        from prompt_builder_control import build_prompt, zone_count_from_obs
        from llm_agent_colab import call_llm, parse_actions_with_validation

        n = zone_count_from_obs(obs)
        prompt = build_prompt(
            obs=obs, building=self.cfg.building, location=self.cfg.location,
            climate=self.cfg.climate, target=self.cfg.target,
            round_idx=step + 1, history=list(self.history),
        )

        if self.ex_dataset:
            try:
                from few_shot_auto import select_examples, format_few_shot_block, inject_few_shot, SelectionConfig
                examples = select_examples(
                    self.ex_dataset, current_obs=obs,
                    config=SelectionConfig(k=self.cfg.k_fewshot, alpha=self.cfg.fewshot_alpha),
                    building=self.cfg.building, climate=self.cfg.climate, location=self.cfg.location,
                )
                block = format_few_shot_block(examples, target=self.cfg.target, n=n)
                prompt = inject_few_shot(prompt, block)
            except Exception as e:
                logger.warning(f"[{self.name}] few-shot injection failed: {e}")

        self._calls += 1
        try:
            raw = call_llm(
                prompt, n_actions=n, model_name=self.cfg.model_name,
                max_new_tokens=self.cfg.max_new_tokens, temperature=self.cfg.llm_temperature,
            )
        except Exception as e:
            logger.error(f"[{self.name}] LLM call failed: {e}")
            raw = ""

        action, meta = parse_actions_with_validation(raw, n)
        if action is None:
            self._parse_fail += 1
            action = [0.0] * n
        return [float(np.clip(a, -1.0, 1.0)) for a in action]

    def record(self, step, action, reward, obs, obs_next):
        n = _n_zones(obs)
        self.history.append({
            "step": step + 1,
            "action": [float(x) for x in action],
            "reward": float(reward),
            "env_temp": float(obs[n]) if len(obs) > n else 0.0,
            "obs_before": obs,
            "obs_after": obs_next,
        })

    @property
    def extra_stats(self):
        if self._calls == 0:
            return {}
        return {"parse_fail_rate": self._parse_fail / self._calls}


# ---------------------------------------------------------------------------
# Evaluation loop & metrics
# ---------------------------------------------------------------------------
def run_episode(controller: Controller, env, cfg: EvalConfig, offset: int) -> Dict[str, Any]:
    """Run one deterministic episode and collect per-step metrics."""
    obs = _reset_env(env, offset)
    controller.reset()

    rewards, abs_devs, energies = [], [], []
    violations = 0
    zone_steps = 0
    step_rewards = []

    for step in range(cfg.max_steps):
        n = _n_zones(obs)
        action = controller.act(obs, step)
        obs_next, reward, done, info = _env_step(env, action)

        # Per-step metrics from the room temperatures (obs[:n])
        temps = np.asarray(obs_next[:n], dtype=np.float32)
        abs_devs.append(float(np.mean(np.abs(temps - cfg.target))))
        violations += int(np.sum((temps < COMFORT_LOW) | (temps > COMFORT_HIGH)))
        zone_steps += n
        energies.append(float(np.sum(np.abs(action))))
        rewards.append(reward)
        step_rewards.append(reward)

        # Let stateful controllers (LLM) update their history
        if hasattr(controller, "record"):
            controller.record(step, action, reward, obs, obs_next)

        obs = obs_next
        if done:
            break

    return {
        "total_return": float(np.sum(rewards)),
        "mean_step_reward": float(np.mean(rewards)) if rewards else 0.0,
        "comfort_violation_rate": float(violations / max(1, zone_steps)),
        "mean_abs_dev_target": float(np.mean(abs_devs)) if abs_devs else 0.0,
        "energy_proxy": float(np.mean(energies)) if energies else 0.0,
        "n_steps": len(rewards),
        "step_rewards": step_rewards,
    }


def evaluate_controller(controller: Controller, cfg: EvalConfig) -> Dict[str, Any]:
    """Evaluate a controller across all episode offsets (and seeds if stochastic)."""
    env = make_env(cfg)
    seeds = cfg.llm_seeds if controller.is_stochastic else 1

    per_run = []
    for offset in cfg.episode_offsets:
        for s in range(seeds):
            if controller.is_stochastic:
                import torch
                torch.manual_seed(1000 + s)
            metrics = run_episode(controller, env, cfg, offset)
            metrics["offset"] = offset
            metrics["seed"] = s
            per_run.append(metrics)
            logger.info(
                f"[{controller.name}] offset={offset} seed={s} "
                f"return={metrics['total_return']:.2f} "
                f"violation={metrics['comfort_violation_rate']:.3f} "
                f"energy={metrics['energy_proxy']:.3f}"
            )

    keys = ["total_return", "mean_step_reward", "comfort_violation_rate",
            "mean_abs_dev_target", "energy_proxy"]
    summary = {k: float(np.mean([r[k] for r in per_run])) for k in keys}
    summary.update({f"{k}_std": float(np.std([r[k] for r in per_run])) for k in keys})
    summary.update(controller.extra_stats)
    summary["n_runs"] = len(per_run)
    # Keep one representative reward curve for plotting (first run)
    summary["_step_rewards"] = per_run[0]["step_rewards"] if per_run else []
    return summary


def build_controllers(names: List[str], cfg: EvalConfig, args) -> List[Controller]:
    controllers: List[Controller] = []
    for name in names:
        if name == "zero":
            controllers.append(ZeroController())
        elif name == "rule":
            controllers.append(RuleBasedController(target=cfg.target, gain=args.rule_gain))
        elif name == "ppo":
            controllers.append(PPOController(args.ppo_model))
        elif name == "llm":
            controllers.append(LLMController(cfg, name="llm_base"))
        elif name == "llm_ft":
            controllers.append(LLMController(cfg, adapter_path=args.adapter, name="llm_ft"))
        else:
            logger.warning(f"Unknown controller: {name}")
    return controllers


def plot_comparison(results: Dict[str, Dict], out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable, skipping plot: {e}")
        return

    names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Controlled Evaluation: Controller Comparison", fontsize=14, fontweight="bold")

    def bar(ax, key, title, lower_better=False):
        vals = [results[n][key] for n in names]
        errs = [results[n].get(f"{key}_std", 0.0) for n in names]
        colors = ["#4C72B0"] * len(names)
        ax.bar(names, vals, yerr=errs, capsize=4, color=colors)
        ax.set_title(title + (" (lower is better)" if lower_better else " (higher is better)"))
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    bar(axes[0, 0], "total_return", "Total Return")
    bar(axes[0, 1], "comfort_violation_rate", "Comfort Violation Rate", lower_better=True)
    bar(axes[1, 0], "energy_proxy", "Energy Proxy (|action|)", lower_better=True)

    ax = axes[1, 1]
    for n in names:
        curve = results[n].get("_step_rewards", [])
        if curve:
            ax.plot(curve, label=n, linewidth=1.2, alpha=0.8)
    ax.set_title("Per-step Reward (first run)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "evaluation_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Saved comparison plot: {path}")


def print_table(results: Dict[str, Dict]):
    cols = ["total_return", "comfort_violation_rate", "mean_abs_dev_target", "energy_proxy"]
    header = f"{'controller':<14}" + "".join(f"{c:>24}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<14}"
        for c in cols:
            row += f"{m[c]:>18.3f} ±{m.get(c + '_std', 0):>4.2f}"
        print(row)
    print("=" * len(header) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Controlled evaluation of HVAC controllers")
    parser.add_argument("--controllers", nargs="+", default=["zero", "rule"],
                        help="Subset of: zero rule ppo llm llm_ft")
    parser.add_argument("--building", default="OfficeSmall")
    parser.add_argument("--climate", default="Hot_Dry")
    parser.add_argument("--location", default="Tucson")
    parser.add_argument("--target", type=float, default=22.0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--episode_offsets", nargs="+", type=int, default=[0])
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_seeds", type=int, default=1)
    parser.add_argument("--fewshot_json", default=None)
    parser.add_argument("--ppo_model", default="pipeline_output/01_ppo_training/ppo_final.zip")
    parser.add_argument("--adapter", default="pipeline_output/04_finetuning/final_model")
    parser.add_argument("--rule_gain", type=float, default=0.5)
    parser.add_argument("--out_dir", default="pipeline_output/06_eval")
    args = parser.parse_args()

    cfg = EvalConfig(
        building=args.building, climate=args.climate, location=args.location,
        target=args.target, max_steps=args.max_steps, episode_offsets=args.episode_offsets,
        model_name=args.model_name, llm_temperature=args.llm_temperature,
        llm_seeds=args.llm_seeds, fewshot_json=args.fewshot_json,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CONTROLLED EVALUATION")
    logger.info(f"Controllers: {args.controllers}")
    logger.info(f"Episodes: offsets={cfg.episode_offsets} x {cfg.max_steps} steps")
    logger.info("=" * 60)

    controllers = build_controllers(args.controllers, cfg, args)

    results: Dict[str, Dict] = {}
    for ctrl in controllers:
        logger.info(f"\n--- Evaluating: {ctrl.name} ---")
        try:
            results[ctrl.name] = evaluate_controller(ctrl, cfg)
        except Exception as e:
            logger.error(f"Controller {ctrl.name} failed: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        logger.error("No controller produced results.")
        return 1

    # Persist (drop the bulky step-reward curve from the JSON summary)
    clean = {n: {k: v for k, v in m.items() if not k.startswith("_")} for n, m in results.items()}
    with open(out_dir / "evaluation_results.json", "w") as f:
        json.dump({"config": asdict(cfg), "results": clean}, f, indent=2)
    logger.info(f"💾 Saved results: {out_dir / 'evaluation_results.json'}")

    print_table(results)
    plot_comparison(results, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
