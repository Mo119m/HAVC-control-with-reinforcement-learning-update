# Methodology Review & Improvement Roadmap

> Purpose: diagnose why the original pipeline produced mediocre results, and lay
> out a concrete, prioritized plan to fix it.

---

## TL;DR

The original pipeline was limited by **three root causes**, in order of impact:

1. **The reward signal was confounded by environment difficulty.** Few-shot
   selection, self-distillation filtering, and the fine-tuning advantage all used
   BEAR's raw per-step reward — a value driven mostly by how hard the current
   conditions are (outside temperature, irradiance, occupancy), not by how good
   the action was. This effectively trained the model to "be in easy conditions",
   which it cannot control.
2. **The fine-tuning "offline PPO" was theoretically unsound.** GAE was run over a
   filtered, shuffled, single-step dataset; the value head was randomly
   initialized and trained on ~100-200 samples; the "old policy" was recomputed
   from the updated policy each epoch; and advantage normalization fought the
   "imitate the good samples" intent.
3. **There was no controlled evaluation.** `draw_reward.py` overlaid reward
   curves from different episodes / weather / seeds, so it was impossible to tell
   whether any change actually helped (past tuning was effectively blind).

> The BEAR environment is fully deterministic (`reset()` always starts at
> `epochs=0` with initial temperature = target, and weather is read from the EPW
> file in order). Once policy stochasticity is fixed (PPO `deterministic=True`,
> LLM `temperature=0`), every controller sees a byte-identical episode. This is
> the basis for fixing #3 and the prerequisite for all other improvements.

---

## Detailed diagnosis

### 1. Reward signal confounded by environment difficulty (top root cause)

BEAR's per-step reward (`BEAR/Customize/reward_functions.py`) is
`-(action cost + error + temperature violation + CO2)`. At a given step its
magnitude is dominated by **external conditions**, not by whether the action was
good relative to alternatives. As a result:

- `select_representative.py`: selecting "highest reward" picks **mild-weather
  moments**, not exemplary control.
- `prepare_distillation_data.py` / the in-finetuner quantile clip: keeping
  "high-reward steps" likewise biases toward easy moments.
- `7b_finetune_fixed.py` GAE advantage: mostly reflects the difficulty trajectory,
  not action merit.

**Fix:** use the *advantage relative to a baseline at the same state* to decouple
action quality from environment difficulty. The most direct route is to reuse the
trained PPO critic `V(s)` and compute `A = r + γ·V(s′) − V(s)`.

### 2. The offline-PPO fine-tuner was unsound

In `core_modules/7b_finetune_fixed.py`:

- **GAE over filtered, shuffled data:** the dataset is first reward-quantile
  clipped (breaking temporal adjacency) and then GAE is computed, so
  `next_value = values[t+1]` is the value of an unrelated state → advantages are
  noise.
- **Randomly-initialized value head** (a single bf16 Linear) trained on ~100-200
  samples → `V(s)` is noise → GAE is noise on top of noise.
- **`old_lp` recomputed from the updated policy each epoch** → no stable reference
  policy, the PPO ratio degenerates and clipping does nothing.
- **Self-contradiction:** after filtering to "good" samples, advantages are
  normalized to zero mean, so ~half of the curated good samples get negative
  advantage and are pushed down — fighting the self-distillation intent.

**Fix:** for "single-step reward + offline filtered data", use **filtered BC /
advantage-weighted regression (AWR/RWR) / best-of-N rejection-sampling
distillation**, not GAE-PPO over filtered, shuffled single-step data.

### 3. No controlled evaluation

`core_modules/draw_reward.py` only overlaid reward curves from PPO (collected
during 500k training steps), the LLM rollout, and the fine-tuned rollout — all on
different episodes/weather/seeds. It cannot support claims like "fine-tuned LLM >
base LLM".

**Fix:** evaluate every controller on the *same* fixed episodes and report episode
return, comfort-violation rate, and energy. See `core_modules/evaluate.py`.

### Other issues

4. **The advertised self-distillation filtering was never wired in:**
   `main_pipeline.py` fed the raw `llm_rollout.json` straight to fine-tuning;
   `prepare_distillation_data.py` (the README's "Stage 4") was never called.
5. **Too little data:** default `episodes=1 × 200 steps`, a single weather window,
   no train/val split → overfitting and high variance.
6. **Few-shot came from the PPO (MLP) policy**, contradicting the "self-
   distillation avoids PPO→LLM distribution shift" claim.
7. **Doc/code/config inconsistencies:** README claimed 6 stages incl. `distill`,
   the code had 5 and no `distill`; `run_progressive_training.py` was orphaned;
   nested `config.json` did not match the flat `PipelineConfig` defaults.

---

## Improvement roadmap

| Priority | Change | Status |
|---|---|---|
| **P0** | Controlled evaluation harness (fixed episodes; PPO / rule baseline / base LLM / fine-tuned LLM) | ✅ `core_modules/evaluate.py` |
| **P0** | Critic-based advantage for LLM transitions | ✅ `core_modules/compute_advantage.py` |
| **P1** | Replace offline-PPO fine-tuning with AWR (advantage-weighted regression) | ✅ `core_modules/awr_finetune.py` |
| **P1** | Make `distill` advantage-based and wire it into the pipeline | ✅ `prepare_distillation_data.py` + `main_pipeline.py` |
| **P1** | Multi-window rollout for diverse data | ✅ `rollout_fewshot_version.py` (`episode_offset_stride`) |
| **P2** | Few-shot from the LLM's own high-advantage steps (true self-distillation) | ✅ `select_representative.py` + `fewshot_source='llm'` |
| **P2** | Wire the full pipeline; align README/config | ✅ 7 stages wired; docs aligned |

### Improved pipeline (7 stages)

```
ppo → select → rollout → advantage → distill → finetune (AWR) → eval (controlled)
```

- **advantage** — `compute_advantage.py`: TD advantage from the PPO critic
  (decoupled from environment difficulty).
- **distill** — `prepare_distillation_data.py`: ranks by **advantage** when
  present (falls back to reward) and keeps the high-quality subset.
- **finetune** — `awr_finetune.py` (AWR) by default; set `use_awr=False` to fall
  back to the legacy `7b_finetune_fixed.py` for A/B comparison.
- **eval** — `evaluate.py`: controlled comparison of `zero/rule/ppo/llm/llm_ft`
  on identical episodes.

### Still to do (next)

1. Scale the rollout to more episodes / weather windows to grow the distillation
   set and reduce overfitting.
2. Run the A/B comparison (legacy offline-PPO vs AWR) through the evaluation
   harness to quantify the gain.
3. Optionally iterate the self-distillation loop (rebuild few-shot from the latest
   high-advantage rollout, re-distill, re-finetune).

> Discipline: **every methodology change must be quantified on fixed episodes with
> the evaluation harness before drawing conclusions.**
</content>
