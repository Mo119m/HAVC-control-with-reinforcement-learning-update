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

Recommended path (best-of-N, environment as verifier):

```
ppo → select → bestofn → finetune (AWR) → eval (controlled)
```

| Stage | Module | What it does |
|-------|--------|--------------|
| `ppo` | `ppo_collect.py` | Train a PPO baseline (for comparison in `eval`; its trajectory also seeds few-shot examples). |
| `select` | `select_representative.py` | Pick diverse, high-quality few-shot examples (ranked by advantage when available, else reward). |
| `bestofn` | `best_of_n_collect.py` | The LLM proposes **N candidate actions per state**; each is scored by its **true reward in BEAR** (same-state comparison ⇒ no environment-difficulty confound, no PPO ceiling). The best action per state becomes training data with a clean same-state advantage. |
| `finetune` | `awr_finetune.py` | **Advantage-Weighted Regression**: weighted SFT with `w = clip(exp(A_norm/β))` on the best-of-N data. Run `bestofn → finetune` repeatedly for expert iteration. |
| `eval` | `evaluate.py` | Compare `zero / rule / ppo / llm / llm_ft` on **identical deterministic episodes** (return, comfort-violation rate, energy). |

Why best-of-N: BEAR's per-step reward is dominated by how hard the current
weather/occupancy is, not by how good the action was. Comparing N actions **at
the same state** dissolves that confound completely (identical conditions, so
reward differences come purely from the action) — no critic needed, and because
the teacher is the real environment, the distilled policy can approach the task
optimum rather than being capped at PPO.

Legacy alternative (PPO-critic advantage) is still available as individual
stages: `rollout → advantage → distill → finetune`. `advantage`
(`compute_advantage.py`) adds `A = r + γ·V(s′) − V(s)` from the PPO critic, and
`distill` (`prepare_distillation_data.py`) keeps the high-advantage subset. This
path is simpler to run but is bounded by the PPO critic.

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
| `n_candidates` | `6` | best-of-N: actions sampled per state |
| `bon_horizon` | `1` | best-of-N: counterfactual look-ahead length |
| `awr_beta` | `1.0` | AWR temperature (higher = flatter weights) |
| `gamma` / `adv_keep_percentile` | `0.99` / `0.5` | legacy PPO-critic advantage path |
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

Done: controlled evaluation, best-of-N self-distillation (environment as
verifier), AWR fine-tuning, multi-window rollout, advantage-ranked few-shot,
end-to-end wiring. (A legacy PPO-critic advantage path is also available.)

Next, for a result worth having (see [`METHODOLOGY_REVIEW.md`](METHODOLOGY_REVIEW.md)):
the LLM will not beat PPO/MPC at single-building optimal control. The defensible
wins are where PPO structurally cannot compete:
1. **Generalization** — self-distill on several buildings/climates, evaluate
   zero-shot on held-out ones vs per-building-retrained PPO.
2. **Language-conditioned control** — natural-language constraints in the prompt
   ("prioritize comfort in room 2", "minimize energy after 6pm") with no retraining.
3. **Self-improvement** — iterate `bestofn → finetune` and show `llm_ft ≫ llm`.

## References

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) ·
  [LoRA](https://arxiv.org/abs/2106.09685) ·
  [PPO](https://arxiv.org/abs/1707.06347) ·
  [AWR](https://arxiv.org/abs/1910.00177)
- BEAR: Physics-Principled Building Environment for Control and RL (see `articles/`)
</content>
