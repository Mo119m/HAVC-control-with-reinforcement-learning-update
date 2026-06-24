"""
Cross-Scenario (Generalization) Evaluation
==========================================

The point of using an LLM controller is NOT to beat PPO/MPC on a single building
— that is a solved control problem. The defensible advantage is **generalization**:
one LLM controller, evaluated zero-shot across many buildings and climates,
without any per-scenario retraining.

This module evaluates the same set of controllers across a list of scenarios
(building × climate) and reports a controller × scenario matrix, so you can show:

- A single fine-tuned LLM (`llm_ft`) holding up across unseen buildings/climates.
- PPO **structurally failing to transfer**: a PPO policy trained on one building
  has a fixed observation/action dimensionality, so it literally cannot run on a
  building with a different number of zones (reported as "incompatible"). Even on
  the same building, a different climate degrades it.
- The rule thermostat as a scenario-agnostic reference.

It reuses the controlled, deterministic evaluation in ``evaluate.py`` (same
episodes, same metrics) — this is that harness swept over scenarios.

Usage
-----
    python core_modules/generalization_eval.py \
        --controllers rule ppo llm llm_ft \
        --preset buildings \
        --ppo_model pipeline_output/01_ppo_training/ppo_final.zip \
        --adapter pipeline_output/04_finetuning/final_model \
        --fewshot_json pipeline_output/02_few_shot_samples/few_shot_examples_structured.json \
        --train_scenario OfficeSmall/Hot_Dry \
        --max_steps 200 --out_dir pipeline_output/07_generalization
"""

import os
import sys
import json
import argparse
import logging
from types import SimpleNamespace
from pathlib import Path
from typing import List, Dict

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_PROJECT_ROOT, os.path.dirname(__file__)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate as E  # noqa: E402  (the controlled single-scenario harness)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Scenario presets (building, climate, location). Buildings differ in zone count,
# so PPO trained on one cannot run on another — that is the headline contrast.
SCENARIO_PRESETS: Dict[str, List[Dict]] = {
    # Vary the building, fix the climate: tests building transfer.
    "buildings": [
        {"building": b, "climate": "Hot_Dry", "location": "Tucson"}
        for b in ["OfficeSmall", "OfficeMedium", "RestaurantSitDown",
                  "SchoolPrimary", "HotelSmall", "RetailStandalone"]
    ],
    # Fix the building, vary the climate: tests climate transfer (PPO can run,
    # same dims, but it was trained on one climate).
    "climates": [
        {"building": "OfficeSmall", "climate": c, "location": "Tucson"}
        for c in ["Hot_Dry", "Warm_Marine", "Mixed_Humid", "Cool_Humid", "Cold_Dry"]
    ],
}


def scenario_label(s: Dict) -> str:
    return f"{s['building']}/{s['climate']}"


def run_one(controller_name: str, scenario: Dict, args) -> Dict:
    """Evaluate a single controller on a single scenario (reusing evaluate.py)."""
    cfg = E.EvalConfig(
        building=scenario["building"], climate=scenario["climate"],
        location=scenario.get("location", "Tucson"),
        target=args.target, max_steps=args.max_steps,
        episode_offsets=list(args.episode_offsets),
        model_name=args.model_name, llm_temperature=args.llm_temperature,
        llm_seeds=args.llm_seeds, fewshot_json=args.fewshot_json,
    )
    try:
        ctrl = E.build_controllers([controller_name], cfg, args)[0]
        summary = E.evaluate_controller(ctrl, cfg)
        return {k: v for k, v in summary.items() if not k.startswith("_")}
    except Exception as e:
        # PPO on a different building => obs/action dim mismatch => lands here.
        msg = str(e)
        reason = "incompatible (needs a per-scenario model)" if (
            "shape" in msg.lower() or "dimension" in msg.lower() or "size" in msg.lower()
        ) else f"error: {msg[:120]}"
        logger.warning(f"[{controller_name} @ {scenario_label(scenario)}] {reason}")
        return {"error": reason}


def print_matrix(results: Dict[str, Dict[str, Dict]], scenarios: List[Dict], metric: str, train_label: str = None):
    labels = [scenario_label(s) for s in scenarios]
    width = max(12, max(len(l) for l in labels) + 2)
    header = f"{'controller':<12}" + "".join(f"{l:>{width}}" for l in labels) + f"{'mean*':>10}"
    print("\n" + "=" * len(header))
    print(f"Metric: {metric}" + (f"   (train scenario: {train_label})" if train_label else ""))
    print(header)
    print("-" * len(header))
    for name, per_scn in results.items():
        row = f"{name:<12}"
        transfer_vals = []
        for s, l in zip(scenarios, labels):
            cell = per_scn.get(l, {})
            if "error" in cell:
                row += f"{'N/A':>{width}}"
            else:
                v = cell.get(metric, float('nan'))
                row += f"{v:>{width}.2f}"
                if l != train_label:
                    transfer_vals.append(v)
        mean_transfer = np.mean(transfer_vals) if transfer_vals else float('nan')
        row += f"{mean_transfer:>10.2f}"
        print(row)
    print("=" * len(header))
    print("* mean over held-out (non-train) scenarios where the controller could run\n")


def plot_matrix(results, scenarios, out_dir: Path, metric: str = "total_return"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"matplotlib unavailable, skipping plot: {e}")
        return
    labels = [scenario_label(s) for s in scenarios]
    names = list(results.keys())
    x = np.arange(len(labels))
    w = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.8), 6))
    for i, name in enumerate(names):
        vals = []
        for l in labels:
            cell = results[name].get(l, {})
            vals.append(np.nan if "error" in cell else cell.get(metric, np.nan))
        ax.bar(x + i * w, vals, w, label=name)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Cross-scenario {metric} (missing bar = controller could not run)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = out_dir / "generalization_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Saved generalization plot: {p}")


def main():
    p = argparse.ArgumentParser(description="Cross-scenario generalization evaluation")
    p.add_argument("--controllers", nargs="+", default=["rule", "ppo", "llm", "llm_ft"])
    p.add_argument("--preset", choices=list(SCENARIO_PRESETS), default="buildings")
    p.add_argument("--scenarios", default=None,
                   help="Optional JSON file with a list of {building,climate,location} (overrides --preset)")
    p.add_argument("--train_scenario", default=None,
                   help="Label 'Building/Climate' the PPO/adapter was trained on (for the table)")
    p.add_argument("--target", type=float, default=22.0)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--episode_offsets", nargs="+", type=int, default=[0])
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--llm_temperature", type=float, default=0.0)
    p.add_argument("--llm_seeds", type=int, default=1)
    p.add_argument("--fewshot_json", default=None)
    p.add_argument("--ppo_model", default="pipeline_output/01_ppo_training/ppo_final.zip")
    p.add_argument("--adapter", default="pipeline_output/04_finetuning/final_model")
    p.add_argument("--rule_gain", type=float, default=0.5)
    p.add_argument("--out_dir", default="pipeline_output/07_generalization")
    args = p.parse_args()

    if args.scenarios and os.path.exists(args.scenarios):
        with open(args.scenarios) as f:
            scenarios = json.load(f)
    else:
        scenarios = SCENARIO_PRESETS[args.preset]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CROSS-SCENARIO GENERALIZATION EVALUATION")
    logger.info(f"Controllers: {args.controllers}")
    logger.info(f"Scenarios ({len(scenarios)}): {[scenario_label(s) for s in scenarios]}")
    logger.info("=" * 60)

    # Controller-outer / scenario-inner so each LLM model loads only once.
    results: Dict[str, Dict[str, Dict]] = {}
    for name in args.controllers:
        results[name] = {}
        for scenario in scenarios:
            logger.info(f"\n--- {name} @ {scenario_label(scenario)} ---")
            results[name][scenario_label(scenario)] = run_one(name, scenario, args)

    with open(out_dir / "generalization_results.json", "w") as f:
        json.dump({"scenarios": scenarios, "train_scenario": args.train_scenario,
                   "results": results}, f, indent=2)
    logger.info(f"💾 Saved results: {out_dir / 'generalization_results.json'}")

    print_matrix(results, scenarios, "total_return", args.train_scenario)
    print_matrix(results, scenarios, "comfort_violation_rate", args.train_scenario)
    plot_matrix(results, scenarios, out_dir, "total_return")
    return 0


if __name__ == "__main__":
    sys.exit(main())
