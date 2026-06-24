# HVAC Control with LLM + RL (Self-Distillation)

Optimize HVAC (heating/cooling) control with a large language model (Qwen2.5-7B)
on the [BEAR](articles) building simulator, improved with reinforcement-learning
signals. The LLM learns from its **own** better-than-baseline behaviour
(self-distillation), guided by a critic-based advantage that separates action
quality from environmental difficulty.

> **2026-06 methodology refactor.** The original pipeline produced mediocre
> results for reasons documented in [`METHODOLOGY_REVIEW.md`](METHODOLOGY_REVIEW.md)
> (reward signal confounded by environment difficulty, an unsound offline-PPO
> fine-tuner, and no controlled evaluation). Those root causes are fixed; the
> current pipeline is the 7-stage flow below.

## Pipeline

```
ppo → select → rollout → advantage → distill → finetune (AWR) → eval (controlled)
```

| Stage | Module | What it does |
|-------|--------|--------------|
| `ppo` | `ppo_collect.py` | Train a PPO baseline; its trajectory seeds few-shot examples and its **critic** is reused for advantages. |
| `select` | `select_representative.py` | Pick diverse, high-quality few-shot examples (ranked by advantage when available, else reward). |
| `rollout` | `rollout_fewshot_version.py` | The base LLM controls the building and generates its own trajectories (optionally across multiple weather windows). |
| `advantage` | `compute_advantage.py` | Use the PPO critic to add `A = r + γ·V(s′) − V(s)` per step, **decoupling action quality from environment difficulty**. |
| `distill` | `prepare_distillation_data.py` | Keep the high-advantage subset of the LLM's own steps (true self-distillation data). |
| `finetune` | `awr_finetune.py` | **Advantage-Weighted Regression**: weighted SFT with `w = clip(exp(A_norm/β))`. Replaces the unsound offline-PPO fine-tuner. |
| `eval` | `evaluate.py` | Compare `zero / rule / ppo / llm / llm_ft` on **identical deterministic episodes** (return, comfort-violation rate, energy). |

Why this works: BEAR's per-step reward is dominated by how hard the current
weather/occupancy is, not by how good the action was. Ranking or weighting by
raw reward therefore selects *easy moments*, not *good control*. Subtracting the
critic baseline `V(s)` removes that confound, so the LLM is trained to imitate
its genuinely better-than-baseline decisions.

## Quick start

```bash
pip install -r requirements.txt

# Run the whole pipeline (needs a GPU for the LLM stages)
python core_modules/main_pipeline.py --stage all

# Or run stages individually
python core_modules/main_pipeline.py --stage ppo
python core_modules/main_pipeline.py --stage advantage
python core_modules/main_pipeline.py --stage finetune
```

Controlled evaluation can be run on its own at any time (cheap baselines need no
GPU; LLM controllers do):

```bash
python core_modules/evaluate.py \
    --controllers zero rule ppo llm llm_ft \
    --ppo_model pipeline_output/01_ppo_training/ppo_final.zip \
    --fewshot_json pipeline_output/02_few_shot_samples/few_shot_examples_structured.json \
    --episode_offsets 0 2000 4000 --max_steps 200 \
    --out_dir pipeline_output/06_eval
```

## Configuration

Runtime configuration is the **flat** `PipelineConfig` dataclass in
`core_modules/main_pipeline.py` (CLI flags and `--config <flat.json>` override
defaults). Key fields:

| Field | Default | Meaning |
|-------|---------|---------|
| `building` / `weather` / `location` | `OfficeSmall` / `Hot_Dry` / `Tucson` | BEAR scenario |
| `model_name` | `Qwen/Qwen2.5-7B-Instruct` | base LLM |
| `gamma` | `0.99` | discount for the TD advantage |
| `awr_beta` | `1.0` | AWR temperature (higher = flatter weights) |
| `adv_keep_percentile` | `0.5` | distillation keeps the better half by advantage |
| `fewshot_source` | `ppo` | `llm` rebuilds few-shot from the LLM's high-advantage steps |
| `llm_rollout_offset_stride` | `0` | `>0` spreads episodes across weather windows |
| `eval_controllers` / `eval_offsets` | see code | controllers and episodes for the eval stage |

> Note: the nested top-level `config.json` is consumed by `config_manager.py`
> and the test suite, **not** by `main_pipeline.py`. Do not pass it to `--config`.

## Repository layout

```
core_modules/        # Pipeline modules (see table above) + helpers
BEAR/                # Vendored BEAR simulator (third party) + scenario data
articles/            # BEAR reference paper
config.json          # Config for config_manager.py / tests (not the pipeline)
requirements.txt
METHODOLOGY_REVIEW.md  # Diagnosis of the old pipeline + improvement roadmap
```

## Status / roadmap

Done: controlled evaluation, critic-based advantage, advantage-based
self-distillation, AWR fine-tuning, multi-window rollout, advantage-ranked
few-shot, end-to-end wiring.

Next: scale the rollout to more episodes/weather windows for a larger
distillation set; run the A/B (legacy offline-PPO vs AWR) through the evaluation
harness to quantify the gain. See [`METHODOLOGY_REVIEW.md`](METHODOLOGY_REVIEW.md).

## References

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) ·
  [LoRA](https://arxiv.org/abs/2106.09685) ·
  [PPO](https://arxiv.org/abs/1707.06347) ·
  [AWR](https://arxiv.org/abs/1910.00177)
- BEAR: Physics-Principled Building Environment for Control and RL (see `articles/`)
</content>
