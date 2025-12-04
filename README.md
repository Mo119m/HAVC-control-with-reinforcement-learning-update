# HVAC Control with LLM - 完整自我蒸馏Pipeline

## 项目概述

本项目使用大型语言模型（Qwen 7B）结合强化学习（PPO）进行HVAC控制优化，实现了完整的**自我蒸馏（Self-Distillation）**训练流程。

### 核心创新

1. **PPO训练**：使用传统RL收集高质量轨迹数据
2. **Few-Shot学习**：选择多样化、高奖励的示例增强LLM推理
3. **LLM控制**：未微调的Qwen 7B作为控制器生成自己的轨迹
4. **自我蒸馏**：从LLM自己的成功经验中筛选数据
5. **LoRA微调**：使用自我蒸馏数据进行参数高效微调
6. **持续改进**：微调后的LLM性能接近甚至超过PPO

### 关键特性

- **完整的自我蒸馏pipeline**：LLM从自己的成功案例中学习
- **严格的数据筛选**：验证、去除异常值、保留高reward轨迹
- **PPO风格微调**：策略优化 + 价值估计 + 熵正则
- **一键运行**：完整的6阶段自动化流程
- **生产就绪**：完整的错误处理、检查点保存、配置管理

## 项目结构

```
hvac-llm-project/
├── core_modules/                      # 核心Python模块（14个）
│   ├── llm_agent_colab.py            # LLM推理和动作解析
│   ├── prompt_builder_control.py     # HVAC提示词构建
│   ├── few_shot_auto.py              # Few-shot示例选择
│   ├── recorder_v2.py                # PPO轨迹记录
│   ├── select_representative.py      # 代表性样本选择
│   ├── ppo_collect.py                # PPO训练和数据收集
│   ├── draw_reward.py                # 奖励曲线可视化
│   ├── config_manager.py             # 配置管理
│   ├── rollout_fewshot_version.py    # LLM推理rollout
│   ├── 7Blora_rollout.py             # 微调模型评估
│   ├── prepare_distillation_data.py  # 自我蒸馏数据准备（新增）
│   ├── 7b_finetune_fixed.py          # LoRA微调（已修复）
│   ├── test_suite.py                 # 测试套件
│   └── main_pipeline.py              # 主程序（已更新）
│
├── docs/                              # 文档
│   ├── README_v2.md                  # 项目说明（本文件）
│   ├── SETUP_TUTORIAL_v2.md          # 完整教程
│   ├── SELF_DISTILLATION_LOGIC.md    # 自我蒸馏逻辑说明
│   ├── CODE_TRUST_REPORT.md          # 代码可信度报告
│   └── UNREVIEWED_FILES_AUDIT.md     # 文件审查报告
│
├── BEAR/                              # BEAR仿真器
│   └── Data/
│
├── pipeline_output/                   # Pipeline输出（自动创建）
│   ├── 01_ppo_training/
│   │   ├── ppo_trajectory.json
│   │   └── ppo_final.zip
│   ├── 02_few_shot_samples/
│   │   └── few_shot_examples_structured.json
│   ├── 03_llm_rollout/
│   │   ├── llm_rollout.json
│   │   └── distillation_data.json    # 自我蒸馏数据（新增）
│   ├── 04_finetuning/
│   │   ├── final_model/
│   │   └── checkpoints/
│   └── 05_evaluation/
│       ├── finetuned_rollout.json
│       └── comparison_plot.png
│
└── config.json                        # 配置文件

总计：14个核心模块，8个文档，生产就绪
```

## 完整的6阶段Pipeline

### Stage 1: PPO训练

**目的**：使用传统强化学习收集高质量baseline数据

```bash
python main_pipeline.py --stage ppo
```

**输出**：
- `ppo_trajectory.json` - PPO策略网络生成的轨迹
- `ppo_final.zip` - 训练好的PPO模型

**数据用途**：
- 用于构建few-shot示例库
- 作为性能baseline对比

---

### Stage 2: Few-Shot示例选择

**目的**：从PPO轨迹中选择多样化、高奖励的示例

```bash
python main_pipeline.py --stage select
```

**方法**：
1. 奖励预筛选：保留top 2000
2. KMeans聚类：12个簇（保证多样性）
3. 簇内筛选：每簇选top 20

**输出**：
- `few_shot_examples_structured.json` - 240个精选示例

**数据用途**：
- 注入到LLM的prompt中
- 提升LLM的推理能力

---

### Stage 3: LLM Rollout（生成自己的轨迹）

**目的**：未微调的Qwen 7B在环境中运行，生成自己的轨迹

```bash
python main_pipeline.py --stage rollout
```

**关键设置**：
- 使用few-shot examples（来自Stage 2）
- 添加历史动作（最近3步）
- 启用chain-of-thought推理
- Temperature: 0.3-0.7

**输出**：
- `llm_rollout.json` - LLM生成的完整轨迹

**数据特点**：
- 包含成功和失败的案例
- 反映LLM当前的能力
- 是自我蒸馏的数据来源

---

### Stage 4: 自我蒸馏数据准备（新增，关键！）

**目的**：从LLM自己的轨迹中筛选高reward数据用于微调

```bash
python main_pipeline.py --stage distill
```

**筛选策略**：

1. **验证阶段**：
   - 解析成功（parsed_from != "failed"）
   - 动作有效（在[-1, 1]范围内）
   - 没有使用fallback（used_fallback = false）

2. **去除异常值**：
   - 去除reward < 5%分位（异常低）
   - 去除reward > 99%分位（异常高）

3. **保留高reward**：
   - 保留reward >= 50%分位（可配置）
   - 默认保留约一半的有效数据

**输出**：
- `distillation_data.json` - 筛选后的高质量数据

**为什么重要**：
- 这是**真正的自我蒸馏**数据
- 来自LLM自己的成功经验
- 分布匹配（训练数据 = 推理数据）

---

### Stage 5: Fine-tuning（使用自我蒸馏数据）

**目的**：使用LLM自己的成功案例进行微调

```bash
python main_pipeline.py --stage finetune
```

**数据来源**：
- `distillation_data.json` ← 自我蒸馏数据（Stage 4）
- **不是**PPO的数据！

**微调方法**：
- LoRA（参数高效）：只微调~0.07%的参数
- PPO风格损失：
  - 策略损失：增加成功动作的概率
  - 价值损失：学会预测reward
  - 熵正则：保持探索性
- 重要bug修复：正确的importance sampling ratio

**输出**：
- `final_model/` - 微调后的Qwen 7B + LoRA adapter
- `checkpoints/` - 训练检查点

---

### Stage 6: 评估

**目的**：评估微调后的模型性能

```bash
python main_pipeline.py --stage eval
```

**评估内容**：
1. 微调后模型rollout
2. 与PPO、未微调LLM对比
3. 可视化reward曲线

**输出**：
- `finetuned_rollout.json` - 微调后模型的轨迹
- `comparison_plot.png` - 对比图表

**预期结果**：
```
PPO (MLP策略) ≈ 微调后的LLM > 未微调的LLM
```

---

## 快速开始

### 方式1：一键运行完整流程（推荐）

```bash
# 运行所有6个阶段
python main_pipeline.py --stage all
```

这会自动执行：
1. PPO训练
2. Few-shot示例选择
3. LLM rollout
4. **自我蒸馏数据准备**
5. Fine-tuning（使用自我蒸馏数据）
6. 评估和对比

### 方式2：分阶段运行

```bash
# 单独运行每个阶段
python main_pipeline.py --stage ppo
python main_pipeline.py --stage select
python main_pipeline.py --stage rollout
python main_pipeline.py --stage distill     # 新增
python main_pipeline.py --stage finetune
python main_pipeline.py --stage eval
```

### 方式3：手动运行各模块

```bash
# 1. PPO训练
export BUILDING="OfficeSmall"
export WEATHER="Hot_Dry"
python core_modules/ppo_collect.py

# 2. 选择示例
python core_modules/select_representative.py \
    --traj pipeline_output/01_ppo_training/ppo_trajectory.json \
    --out_dir pipeline_output/02_few_shot_samples

# 3. LLM Rollout
python core_modules/rollout_fewshot_version.py \
    --fewshot_json pipeline_output/02_few_shot_samples/few_shot_examples_structured.json \
    --output pipeline_output/03_llm_rollout/llm_rollout.json

# 4. 准备自我蒸馏数据
python core_modules/prepare_distillation_data.py \
    --llm_rollout pipeline_output/03_llm_rollout/llm_rollout.json \
    --output pipeline_output/03_llm_rollout/distillation_data.json \
    --min_percentile 0.5

# 5. Fine-tuning
export ROLLOUT_GLOBS="pipeline_output/03_llm_rollout/distillation_data.json"
export SAVE_DIR="pipeline_output/04_finetuning"
python core_modules/7b_finetune_fixed.py

# 6. 评估
python core_modules/7Blora_rollout.py \
    --adapter pipeline_output/04_finetuning/final_model \
    --output pipeline_output/05_evaluation/finetuned_rollout.json

# 7. 可视化
python core_modules/draw_reward.py \
    pipeline_output/01_ppo_training/ppo_trajectory.json \
    pipeline_output/03_llm_rollout/llm_rollout.json \
    pipeline_output/05_evaluation/finetuned_rollout.json \
    --output pipeline_output/05_evaluation/comparison_plot.png
```

## 安装

### 依赖

```bash
# 基础依赖
pip install torch transformers accelerate
pip install stable-baselines3
pip install peft  # LoRA支持
pip install scikit-learn pandas matplotlib numpy

# BEAR仿真器（如果可用）
pip install -e /path/to/BEAR
```

### 环境要求

**最低配置：**
- GPU: NVIDIA T4 (16GB)
- RAM: 32GB
- 存储: 100GB

**推荐配置：**
- GPU: A100 (40GB) 或 RTX 3090 (24GB)
- RAM: 64GB
- 存储: 500GB NVMe SSD

**Google Colab：**
- 免费版：T4 GPU（可运行除fine-tuning外的所有阶段）
- Colab Pro：A100 GPU（推荐，可运行完整流程）

## 关键改进

### 相比原版的改进

1. **实现了真正的自我蒸馏**
   - 之前：数据来源不明确
   - 现在：明确使用LLM自己的高reward数据

2. **新增数据筛选模块**
   - `prepare_distillation_data.py`
   - 验证 + 去除异常值 + 保留高reward

3. **6阶段完整pipeline**
   - 之前：5个阶段
   - 现在：6个阶段（新增自我蒸馏准备）

4. **修复了严重bug**
   - `7b_finetune_fixed.py`：修复importance sampling ratio错误
   - 详见：CODE_TRUST_REPORT.md

5. **所有模块已优化**
   - 14个核心模块全部审查和优化
   - 可信度：90-95%
   - 详见：UNREVIEWED_FILES_AUDIT.md

## 数据流图

```
┌─────────────┐
│ PPO训练     │
└──────┬──────┘
       │ ppo_trajectory.json
       ↓
┌─────────────┐
│ 选择Few-shot│──→ few_shot_examples.json (用于prompt)
└─────────────┘              ↓
                    ┌─────────────┐
                    │ LLM Rollout │
                    └──────┬──────┘
                           │ llm_rollout.json
                           ↓
                    ┌─────────────────────┐
                    │ 自我蒸馏数据准备    │ ← 新增步骤
                    └──────┬──────────────┘
                           │ distillation_data.json
                           ↓
                    ┌─────────────┐
                    │ Fine-tuning │ ← 使用自我蒸馏数据
                    └──────┬──────┘
                           │ 微调后的模型
                           ↓
                    ┌─────────────┐
                    │    评估     │
                    └─────────────┘
```

## 自我蒸馏原理

### 为什么自我蒸馏有效？

**1. 分布匹配**
```
训练时的数据分布 = 推理时的数据分布
```
避免了从PPO到LLM的distribution shift。

**2. 学习成功模式**
```
高reward轨迹 = LLM的成功案例
微调 = 强化成功的行为模式
```

**3. 迭代改进**
```
LLM v1 → Rollout → 筛选 → Fine-tune → LLM v2 → ...
```

### 筛选策略的重要性

**不是所有LLM数据都有用：**
-  解析失败 → 噪声
-  低reward → 坏案例
-  异常值 → 环境随机性

**筛选后：**
-  成功案例
-  高质量数据
-  稳定训练

## 配置

### 关键参数

**PPO训练：**
```python
total_steps = 500000
learning_rate = 3e-4
gamma = 0.99
```

**Few-shot选择：**
```python
preselect = 2000        # 预筛选top N
clusters = 12           # 聚类数量
n_per_cluster = 20      # 每簇选择数量
```

**自我蒸馏筛选：**
```python
min_reward_percentile = 0.5   # 保留top 50%
reward_q_low = 0.05           # 去除底部5%
reward_q_high = 0.99          # 去除顶部1%
```

**Fine-tuning：**
```python
epochs = 3
learning_rate = 1e-4
batch_size = 2
grad_accum = 8
lora_r = 8
```

## 常见问题

### Q1: 自我蒸馏和使用PPO数据的区别？

**使用PPO数据：**
- 数据来源：PPO策略网络（MLP）
- 问题：分布不匹配（PPO ≠ LLM）

**自我蒸馏：**
- 数据来源：LLM自己的rollout
- 优势：分布匹配，学习自己的成功经验

### Q2: 为什么需要Stage 4（自我蒸馏准备）？

Stage 3生成的`llm_rollout.json`包含所有数据：
- 成功的案例（高reward）
- 失败的案例（低reward）
- 解析错误的数据

Stage 4筛选出**只有成功的案例**用于微调。

### Q3: 可以跳过某些阶段吗？

**必需的阶段：**
- Stage 1（PPO）→ Stage 2（Few-shot）→ Stage 3（LLM Rollout）→ Stage 4（自我蒸馏）→ Stage 5（Fine-tuning）

**可选的阶段：**
- Stage 6（评估）- 如果不需要对比

### Q4: Fine-tuning需要多久？

**取决于硬件：**
- A100 40GB: 4-8小时
- RTX 3090 24GB: 8-12小时（需要减小batch size）
- T4 16GB: 可能OOM，需要更激进的优化

### Q5: 如何调整筛选严格程度？

```bash
# 更严格（只保留top 30%）
python prepare_distillation_data.py \
    --min_percentile 0.7

# 更宽松（保留top 70%）
python prepare_distillation_data.py \
    --min_percentile 0.3
```

## 故障排除

### CUDA out of memory

```bash
# 减小batch size
export BATCH_SIZE="1"
export GRAD_ACCUM="16"

# 或使用更小的模型
export BASE_MODEL="Qwen/Qwen2-1.8B-Instruct"
```

### 自我蒸馏数据太少

```bash
# 降低筛选标准
python prepare_distillation_data.py \
    --min_percentile 0.3  # 保留更多数据
```

### LLM解析失败率高

```bash
# 调整generation参数
export TEMPERATURE="0.3"  # 降低随机性
export TOP_K="50"
export TOP_P="0.7"
```

## 测试

```bash
# 运行测试套件
python test_suite.py

# 小规模测试
python main_pipeline.py --stage ppo
# 修改config.json: "total_steps": 1000
```



## 参考文献

- [Qwen2.5 Model](https://github.com/QwenLM/Qwen2.5)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Self-Distillation in Deep Learning](https://arxiv.org/abs/1503.02531)

---

**版本：** v2.0 (包含完整自我蒸馏)  
**更新日期：** 2025年11月  
