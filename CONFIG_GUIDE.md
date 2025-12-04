# 配置文件说明 (config.json)

## 概述

`config.json`是项目的中心配置文件，包含所有模块的参数设置。支持：
- 默认配置
- 环境变量覆盖
- 命令行参数覆盖

## 配置层级

```
优先级（高到低）：
1. 命令行参数
2. 环境变量
3. config.json文件
4. 代码中的默认值
```

## 配置部分

### 1. env - 环境设置

```json
{
  "env": "development"
}
```

**选项：**
- `"development"` - 开发环境（默认）
- `"production"` - 生产环境
- `"testing"` - 测试环境（使用较小的训练步数）

**用途：** 自动调整其他参数（如测试环境会减少训练步数）

---

### 2. llm - LLM推理配置

```json
{
  "llm": {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "hf_token": null
  }
}
```

**参数说明：**

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `model_name` | HuggingFace模型名称 | Qwen/Qwen2.5-7B-Instruct | - |
| `max_new_tokens` | 最大生成token数 | 256 | 128-512 |
| `temperature` | 随机性控制 | 0.7 | 0.3-0.9 |
| `top_p` | 核采样阈值 | 0.7 | 0.5-0.95 |
| `top_k` | 候选词数量 | 50 | 20-100 |
| `repetition_penalty` | 重复惩罚 | 1.0 | 1.0-1.2 |
| `hf_token` | HuggingFace访问令牌 | null | - |

**环境变量覆盖：**
```bash
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="your_token_here"
```

**使用建议：**
- 降低`temperature`（0.3-0.5）可减少解析失败
- 增加`top_k`可提高多样性
- Chain-of-thought需要较高的`max_new_tokens`（256+）

---

### 3. hvac - HVAC环境配置

```json
{
  "hvac": {
    "building": "OfficeSmall",
    "climate": "Hot_Dry",
    "location": "Tucson",
    "target_temp": 22.0,
    "data_root": "./BEAR/Data/"
  }
}
```

**参数说明：**

| 参数 | 说明 | 示例 |
|------|------|------|
| `building` | 建筑类型 | OfficeSmall, OfficeMedium, OfficeLarge |
| `climate` | 气候类型 | Hot_Dry, Hot_Humid, Cold, Mixed |
| `location` | 地理位置 | Tucson, Miami, Chicago, etc. |
| `target_temp` | 目标温度（°C） | 22.0 |
| `data_root` | BEAR数据根目录 | ./BEAR/Data/ |

**环境变量覆盖：**
```bash
export BUILDING="OfficeSmall"
export WEATHER="Hot_Dry"  # 注意：这里用WEATHER而不是CLIMATE
export LOCATION="Tucson"
```

**添加新数据：**
1. 将新的建筑/气候数据放到`BEAR/Data/`
2. 更新`config.json`中的对应字段
3. 重新运行pipeline

---

### 4. fewshot - Few-shot示例配置

```json
{
  "fewshot": {
    "json_path": "fs_out/few_shot_examples_structured.json",
    "k": 3,
    "alpha": 0.6,
    "preselect": 2000,
    "n_clusters": 12,
    "n_per_cluster": 20
  }
}
```

**参数说明：**

| 参数 | 说明 | 默认值 | 用途 |
|------|------|--------|------|
| `json_path` | Few-shot数据路径 | fs_out/few_shot_examples_structured.json | Stage 2输出 |
| `k` | 每次选择的示例数 | 3 | Prompt中的示例数量 |
| `alpha` | 相似度权重 | 0.6 | 0.6*相似度 + 0.4*reward |
| `preselect` | 预筛选数量 | 2000 | 从PPO轨迹中预选top N |
| `n_clusters` | 聚类数量 | 12 | KMeans聚类簇数 |
| `n_per_cluster` | 每簇选择数 | 20 | 每簇取top N |

**Stage 2输出总数：** `n_clusters × n_per_cluster = 12 × 20 = 240`个示例

**调整建议：**
- 增加`k`（如5）可提供更多上下文，但会增加prompt长度
- 增加`alpha`（如0.8）更注重相似度，减少则更注重reward
- 增加`n_clusters`可提高多样性

---

### 5. prompt - 提示词配置

```json
{
  "prompt": {
    "history_keep": 6,
    "history_lines": 3,
    "enable_cot": true,
    "system_message": "You are an HVAC control expert..."
  }
}
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `history_keep` | 保留的历史步数 | 6 |
| `history_lines` | Prompt中显示的历史行数 | 3 |
| `enable_cot` | 启用Chain-of-Thought | true |
| `system_message` | 系统提示词 | "You are..." |

**Chain-of-Thought的重要性：**
- 启用后，LLM会先推理再输出动作
- 防止小模型重复输出相同动作
- 需要更高的`max_new_tokens`（256+）

---

### 6. training - PPO训练配置

```json
{
  "training": {
    "total_steps": 500000,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "policy_name": "MlpPolicy",
    "n_envs": 1
  }
}
```

**参数说明：**

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `total_steps` | 总训练步数 | 500000 | 100000-1000000 |
| `learning_rate` | 学习率 | 3e-4 | 1e-5 - 1e-3 |
| `gamma` | 折扣因子 | 0.99 | 0.95-0.99 |
| `gae_lambda` | GAE lambda | 0.95 | 0.90-0.99 |
| `policy_name` | 策略类型 | MlpPolicy | MlpPolicy |
| `n_envs` | 并行环境数 | 1 | 1-8 |

**时间估算：**
- 500k steps ≈ 2-4小时（取决于硬件）
- 测试时可设置为1000快速验证

---

### 7. finetuning - Fine-tuning配置

```json
{
  "finetuning": {
    "epochs": 3,
    "lr": 1e-4,
    "batch_size": 2,
    "grad_accum": 8,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "max_seq_len": 1500,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "ppo_clip_eps": 0.2
  }
}
```

**参数说明：**

| 参数 | 说明 | 默认值 | 显存影响 |
|------|------|--------|----------|
| `epochs` | 训练轮数 | 3 | 低 |
| `lr` | 学习率 | 1e-4 | 无 |
| `batch_size` | 批大小 | 2 | **高** |
| `grad_accum` | 梯度累积步数 | 8 | 低 |
| `lora_r` | LoRA秩 | 8 | 中 |
| `lora_alpha` | LoRA alpha | 16 | 无 |
| `lora_dropout` | LoRA dropout | 0.05 | 无 |
| `max_seq_len` | 最大序列长度 | 1500 | **高** |
| `value_coef` | 价值损失系数 | 0.5 | 无 |
| `entropy_coef` | 熵正则系数 | 0.01 | 无 |
| `ppo_clip_eps` | PPO裁剪epsilon | 0.2 | 无 |

**有效批大小：** `batch_size × grad_accum = 2 × 8 = 16`

**显存不足时调整：**
```json
{
  "batch_size": 1,      // 减半
  "grad_accum": 16,     // 加倍
  "max_seq_len": 1024   // 减少
}
```

---

### 8. distillation - 自我蒸馏配置（新增）

```json
{
  "distillation": {
    "min_reward_percentile": 0.5,
    "reward_q_low": 0.05,
    "reward_q_high": 0.99,
    "check_action_validity": true,
    "check_parse_success": true
  }
}
```

**参数说明：**

| 参数 | 说明 | 默认值 | 效果 |
|------|------|--------|------|
| `min_reward_percentile` | 最小reward百分位 | 0.5 | 保留top 50% |
| `reward_q_low` | 去除底部异常值分位 | 0.05 | 去除底部5% |
| `reward_q_high` | 去除顶部异常值分位 | 0.99 | 去除顶部1% |
| `check_action_validity` | 检查动作有效性 | true | 验证动作范围 |
| `check_parse_success` | 检查解析成功 | true | 验证解析状态 |

**筛选严格程度调整：**
```json
// 更严格（只保留top 30%）
{"min_reward_percentile": 0.7}

// 更宽松（保留top 70%）
{"min_reward_percentile": 0.3}

// 非常严格（只保留top 10%）
{"min_reward_percentile": 0.9}
```

---

### 9. paths - 路径配置

```json
{
  "paths": {
    "base_dir": "./pipeline_output",
    "ppo_dir": "01_ppo_training",
    "fewshot_dir": "02_few_shot_samples",
    "llm_rollout_dir": "03_llm_rollout",
    "finetune_dir": "04_finetuning",
    "eval_dir": "05_evaluation",
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs"
  }
}
```

**目录结构：**
```
pipeline_output/
├── 01_ppo_training/
│   ├── ppo_trajectory.json
│   └── ppo_final.zip
├── 02_few_shot_samples/
│   └── few_shot_examples_structured.json
├── 03_llm_rollout/
│   ├── llm_rollout.json
│   └── distillation_data.json  # 自我蒸馏数据
├── 04_finetuning/
│   ├── final_model/
│   └── checkpoints/
└── 05_evaluation/
    ├── finetuned_rollout.json
    └── comparison_plot.png
```

---

### 10. rollout - Rollout配置

```json
{
  "rollout": {
    "max_steps": 200,
    "save_checkpoints": true,
    "checkpoint_interval": 50
  }
}
```

**参数说明：**
- `max_steps`: LLM rollout的最大步数（一个episode的长度）
- `save_checkpoints`: 是否保存检查点
- `checkpoint_interval`: 每N步保存一次检查点

---

## 使用方法

### 方法1：直接使用默认配置

```bash
# 使用项目中的config.json
python main_pipeline.py --stage all
```

### 方法2：使用自定义配置文件

```bash
# 创建自定义配置
cp config.json my_config.json
# 编辑my_config.json...

# 使用自定义配置
python main_pipeline.py --config my_config.json --stage all
```

### 方法3：环境变量覆盖

```bash
# 覆盖特定参数
export BUILDING="OfficeMedium"
export WEATHER="Hot_Humid"
export TOTAL_STEPS="100000"

python main_pipeline.py --stage all
```

### 方法4：命令行参数覆盖

```bash
# 命令行优先级最高
python main_pipeline.py \
    --building OfficeSmall \
    --weather Hot_Dry \
    --stage all
```

---

## 快速配置模板

### 测试配置（快速验证）

```json
{
  "env": "testing",
  "training": {
    "total_steps": 1000
  },
  "finetuning": {
    "epochs": 1
  },
  "rollout": {
    "max_steps": 10
  }
}
```

### 生产配置（完整训练）

```json
{
  "env": "production",
  "training": {
    "total_steps": 1000000
  },
  "finetuning": {
    "epochs": 5,
    "lr": 5e-5
  }
}
```

### 低显存配置（GPU < 24GB）

```json
{
  "finetuning": {
    "batch_size": 1,
    "grad_accum": 16,
    "max_seq_len": 1024
  }
}
```

---

## 生成配置文件

### 使用config_manager.py生成

```python
import sys
sys.path.insert(0, 'core_modules')
from config_manager import get_default_config

# 生成开发环境配置
config = get_default_config('development')
config.save_to_file('config.json')

# 生成生产环境配置
config = get_default_config('production')
config.save_to_file('config_production.json')

# 生成测试环境配置
config = get_default_config('testing')
config.save_to_file('config_testing.json')
```

### 命令行生成

```bash
# 生成默认配置
python -c "
import sys
sys.path.insert(0, 'core_modules')
from config_manager import get_default_config
config = get_default_config('development')
config.save_to_file('config.json')
print('Config created: config.json')
"
```

---

## 验证配置

```python
from config_manager import ProjectConfig

# 加载配置
config = ProjectConfig.load_from_file('config.json')

# 验证
errors = config.validate()
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")
```

---

## 常见配置场景

### 场景1：更换建筑类型

```json
{
  "hvac": {
    "building": "OfficeMedium",  // 从OfficeSmall改为OfficeMedium
    "climate": "Hot_Dry",
    "location": "Tucson"
  }
}
```

### 场景2：调整筛选严格度

```json
{
  "distillation": {
    "min_reward_percentile": 0.7  // 只保留top 30%
  }
}
```

### 场景3：减少训练时间

```json
{
  "training": {
    "total_steps": 100000  // 从500k减少到100k
  },
  "finetuning": {
    "epochs": 2  // 从3减少到2
  }
}
```

### 场景4：提高LLM质量

```json
{
  "llm": {
    "temperature": 0.3,  // 降低随机性
    "max_new_tokens": 512  // 增加输出长度
  },
  "fewshot": {
    "k": 5  // 增加示例数量
  }
}
```

---

## 总结

- **config.json**是项目的中心配置文件
- 支持3层覆盖：文件 < 环境变量 < 命令行
- 所有参数都有合理的默认值
- 可以根据需求灵活调整

**推荐工作流程：**
1. 使用默认config.json开始
2. 根据需要调整特定参数
3. 使用环境变量进行快速实验
4. 保存成功的配置为新文件


