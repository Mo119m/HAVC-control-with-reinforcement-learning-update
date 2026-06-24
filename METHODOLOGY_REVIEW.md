# 方法论审查与改进路线图 (Methodology Review)

> 目的：诊断「为什么最终结果一般」，并给出可落地的改进路线。
> 适用分支：`claude/focused-cori-6TVCM`。

---

## TL;DR

当前 pipeline 的最终效果受限于 **3 个根因**，按影响排序：

1. **奖励信号被环境难度污染** —— few-shot 选择、自蒸馏筛选、微调 advantage 全部用 BEAR 的逐步
   reward 绝对值，而这个值主要由"当时天气/负荷有多难"决定，而非"动作有多好"。等于在训练模型
   "去待在容易的环境里"。
2. **微调阶段的 offline-PPO 在理论上不成立** —— GAE 跑在被筛选+被打乱的单步数据上、value head
   随机初始化只在百级样本上训练、old policy 每个 epoch 用更新后的策略重算、advantage 归一化
   与"筛选好样本"的初衷互相打架。
3. **整条链路缺乏受控评估** —— 现有 `draw_reward.py` 把不同 episode / 天气 / seed 的曲线叠在一起，
   不是同条件对比，因此无法判断任何改动是否真的带来提升（过去的调参基本是盲调）。

> BEAR 环境其实是**完全确定性**的（`reset` 永远从 `epochs=0`、初始温度=target 开始，天气按 EPW
> 顺序读取）。只要固定策略随机性（PPO `deterministic=True`、LLM `temperature=0`），就能做到逐字节
> 相同 episode 的受控对比。这是修复 #3 的基础，也是其他一切改进的前提。

---

## 详细诊断

### 1. 奖励信号被环境难度污染（头号根因）

`BEAR/Customize/reward_functions.py` 的逐步 reward = `-(动作能耗 + 误差 + 温度越界 + CO2)`。
在某个时间步，这个值的大小主要取决于**外部条件**（室外温度、太阳辐照、人员负荷），而不是
"这个动作相对其它动作好不好"。因此：

- `select_representative.py`：按 "reward 最高" 选 few-shot → 实际选出**天气温和的轻松时刻**。
- `prepare_distillation_data.py` / 微调内的分位筛选：保留 "高 reward 步" → 同样偏向轻松时刻。
- `7b_finetune_fixed.py` 的 GAE advantage → 主要反映环境难度起伏，而非动作优劣。

**正确做法**：用 *同一状态下相对 baseline 的优势 (advantage)* 作为信号，把"动作优劣"与"环境难度"
解耦。最直接的方式是用 PPO 训练得到的 critic `V(s)` 计算 `A = r + γV(s') − V(s)`。

### 2. 微调阶段 offline-PPO 不成立

`core_modules/7b_finetune_fixed.py`：

- **GAE 跑在筛选+乱序数据上**：先按 reward 分位裁剪（破坏时间相邻性），再 `compute_gae`，
  `next_value = values[t+1]` 取到的是无关状态的值 → advantage ≈ 噪声。
- **value head 随机初始化**（单层 bf16 Linear），只在约 100–200 个样本上训几个 epoch → `V(s)` ≈ 噪声
  → GAE 噪声叠噪声。
- **`old_lp` 每个 epoch 用更新后的策略重算** → 没有稳定参考策略，PPO ratio 退化、clip 失效。
- **逻辑自相矛盾**：先筛出"高 reward 好样本"，又把 advantage 归一化到零均值 → 约一半好样本拿到
  负 advantage 被往下压；"模仿成功经验"与 PPO 目标互相打架。

**正确做法**：对"单步 reward + 离线筛选数据"，应使用 **filtered BC / 加权回归 (AWR/RWR) /
Best-of-N 拒绝采样蒸馏**，而非在筛选乱序单步数据上跑 GAE-PPO。

### 3. 缺乏受控评估

`core_modules/draw_reward.py` 仅把 PPO 轨迹（500k 训练中采的）、LLM rollout、微调后 rollout 三条
**不同 episode/天气/seed** 的 reward 曲线叠加。无法得出 "PPO ≈ 微调LLM > 基础LLM" 的结论。

**正确做法**：固定一组 eval episodes，所有控制器在**相同状态序列**上跑，报告 episode 总回报、
舒适度越界率、能耗。详见 `core_modules/evaluate.py`。

### 其他问题

4. **宣传的自蒸馏筛选未接入流程**：`main_pipeline.py` 把原始 `llm_rollout.json` 直接喂给微调，
   `prepare_distillation_data.py`（README 的 Stage 4）从未被调用。
5. **数据量太小**：默认 `episodes=1 × 200 steps`，单一天气窗口，无 train/val 划分 → 过拟合+高方差。
6. **few-shot 来自 PPO（MLP）策略**，却宣称"自蒸馏避免 PPO→LLM 分布偏移" → 自相矛盾。
7. **代码/文档/配置不一致**：README 称 6 阶段含 `distill`，代码仅 5 阶段且无 distill；
   `run_progressive_training.py` 未接入；`config.json` 与 `PipelineConfig` 默认值/路径不一致。

---

## 改进路线图

| 优先级 | 改动 | 状态 |
|---|---|---|
| **P0** | 受控评估 harness（固定 episode 对比 PPO / 规则基线 / 基础LLM / 微调LLM）| ✅ `core_modules/evaluate.py` |
| **P0** | critic-based advantage：用 PPO critic 给 LLM transition 算 advantage | ✅ `core_modules/compute_advantage.py` |
| **P1** | 用 AWR（advantage 加权回归）替换 offline-PPO 微调 | ✅ `core_modules/awr_finetune.py` |
| **P1** | distill 阶段改成 advantage-based 筛选并接入主流程 | ✅ `prepare_distillation_data.py` + `main_pipeline.py` |
| **P1** | 扩大并多样化数据（多 episode / 多天气窗口）+ train/val 划分 | ⏳ 待做（评估已支持多 offset；rollout 多 episode 仍需扩量）|
| **P2** | few-shot 改用 LLM 自身高 advantage 范例（真自蒸馏），按 advantage 排序 | ⏳ 待做 |
| **P2** | 接通 pipeline、对齐 README/config | ✅ 7 阶段已接通；config/README 已对齐 |

### 新版 pipeline（7 阶段）

```
ppo → select → rollout → advantage → distill → finetune(AWR) → eval(受控)
```

- **advantage**：`compute_advantage.py`，用 PPO critic 给 rollout 算 TD advantage（解耦环境难度）。
- **distill**：`prepare_distillation_data.py`，**优先按 advantage**（缺失才回退 reward）筛高质量子集。
- **finetune**：默认 `awr_finetune.py`（AWR）；`use_awr=False` 可切回旧 `7b_finetune_fixed.py` 做对比。
- **eval**：`evaluate.py`，固定 episode 受控对比 `zero/rule/ppo/llm/llm_ft`。

> 仍待做（下一步）：(1) rollout 扩到多 episode/多天气窗口以增大蒸馏数据量并降过拟合；
> (2) few-shot 从 LLM 自身高 advantage 步骤里选，做到端到端的真自蒸馏；
> (3) 用评估 harness 跑 A/B：旧 offline-PPO vs 新 AWR，量化提升。

### 度量指标（评估 harness 输出）

- **episode 总回报**（越高越好，主指标）
- **舒适度越界率**：room temp 落在 [18, 22]°C 之外的 (step, zone) 比例
- **能耗代理**：平均 `||action||`（动作幅度）
- **目标温差**：room temp 偏离 target 的平均绝对值
- **解析失败率**（仅 LLM）

> 关键纪律：**任何方法改动都必须先用评估 harness 在固定 episode 上量化，再下结论。**
</content>
</invoke>
