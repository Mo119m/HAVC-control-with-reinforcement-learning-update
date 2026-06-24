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
- the offline-PPO fine-tuner's GAE advantage: mostly reflected the difficulty
  trajectory, not action merit.

**Fix:** use the *advantage relative to a baseline at the same state* to decouple
action quality from environment difficulty. The most direct route is to reuse the
trained PPO critic `V(s)` and compute `A = r + γ·V(s′) − V(s)`.

### 2. The offline-PPO fine-tuner was unsound

In the original offline-PPO fine-tuner (since removed):

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
| **P0** | Critic-based advantage for LLM transitions (legacy alternative) | ✅ `core_modules/compute_advantage.py` |
| **P0** | Best-of-N self-distillation with the environment as verifier (no critic, no PPO ceiling) | ✅ `core_modules/best_of_n_collect.py` |
| **P1** | Replace offline-PPO fine-tuning with AWR (advantage-weighted regression) | ✅ `core_modules/awr_finetune.py` |
| **P1** | Make `distill` advantage-based and wire it into the pipeline | ✅ `prepare_distillation_data.py` + `main_pipeline.py` |
| **P1** | Multi-window rollout for diverse data | ✅ `rollout_fewshot_version.py` (`episode_offset_stride`) |
| **P2** | Few-shot from the LLM's own high-advantage steps (true self-distillation) | ✅ `select_representative.py` + `fewshot_source='llm'` |
| **P2** | Wire the full pipeline; align README/config | ✅ 7 stages wired; docs aligned |

### Improved pipeline (recommended: best-of-N, environment as verifier)

```
ppo → select → bestofn → finetune (AWR) → eval (controlled)
```

- **bestofn** — `best_of_n_collect.py`: the LLM proposes N candidate actions per
  state; each is scored by its **true reward in BEAR** (state snapshot → try
  candidate → read reward → restore). Comparing candidates **at the same state**
  removes the environment-difficulty confound *without a critic*, and because the
  teacher is the real environment there is **no PPO ceiling**. The best action per
  state becomes training data with a clean same-state advantage.
- **finetune** — `awr_finetune.py` (AWR) on the best-of-N data. Iterate
  `bestofn → finetune` for expert iteration. The original unsound offline-PPO
  fine-tuner has been removed.
- **eval** — `evaluate.py`: controlled comparison of `zero/rule/ppo/llm/llm_ft`
  on identical episodes.

A legacy PPO-critic path (`rollout → advantage → distill`, with
`compute_advantage.py`) remains as individual stages, but it is bounded by the
PPO critic and is not the recommended route.

### The strategic point (what makes this produce a result)

An LLM will **not** beat PPO/MPC at single-building optimal control — that is a
solved control problem, and competing there yields a weak/negative result. The
defensible contributions are where PPO structurally cannot compete:

1. **Generalization** — self-distill on several buildings/climates and evaluate
   **zero-shot on held-out** ones against per-building-retrained PPO. One
   controller, no retraining: this is the headline result.
2. **Language-conditioned control** — natural-language constraints/preferences in
   the prompt, with no reward redesign or retraining.
3. **Self-improvement** — iterate best-of-N expert iteration and show
   `llm_ft ≫ llm`.

### Still to do (next)

1. Run the best-of-N expert-iteration loop on one building; confirm `llm_ft ≫ llm`
   and that it approaches the optimum. Ablate: inference-time best-of-N (no
   fine-tuning) vs fine-tuning — the cheaper option may suffice.
2. Run the generalization sweep (`generalization_eval.py` / `--stage generalize`):
   tooling is in place — needs a GPU run to produce the headline numbers
   (one LLM transferring across buildings/climates; PPO cannot even run off its
   training building).
3. Add language-conditioned constraints to the prompt and measure adaptation.

### A note on "beating PPO"

With best-of-N (environment as verifier) PPO is no longer a ceiling, so beating it
is *possible* but not the smart bet: PPO is already strong on single-building
control, best-of-N is capped by the LLM's proposal quality, and a short horizon is
myopic vs PPO's long-horizon return (hence `bon_horizon` defaults to 4). The
robust story is **generalization** (PPO structurally cannot transfer) and that
even *matching* PPO is a strong result given the LLM needs ~hundreds of
self-distillation samples vs PPO's per-building 500k env steps, and transfers
zero-shot.

> Discipline: **every methodology change must be quantified on fixed episodes with
> the evaluation harness before drawing conclusions.**
</content>
