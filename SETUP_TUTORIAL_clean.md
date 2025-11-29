# 完整项目设置和使用教程

## 目录

1. [环境准备](#环境准备)
2. [项目结构](#项目结构)
3. [初次设置](#初次设置)
4. [使用Pipeline](#使用pipeline)
5. [添加新数据](#添加新数据)
6. [运行环境建议](#运行环境建议)
7. [常见问题](#常见问题)

---

## 环境准备

### 推荐运行环境

#### **Google Colab (推荐)**

**优势：**
- 免费GPU (Tesla T4/V100)
- A100可用 (Colab Pro)
- 无需本地配置
- 适合快速实验

**限制：**
- 运行时限制（12-24小时）
- 需要定期保存
- 网络可能断开

**适用场景：**
- PPO训练（中小规模）
- LLM推理
- LoRA微调
- 快速原型

#### **本地GPU服务器（高性能）**

**要求：**
- NVIDIA GPU (≥16GB VRAM)
- RTX 3090 / A100 / V100
- CUDA 11.8+

**优势：**
- 无时间限制
- 完全控制
- 数据安全

**适用场景：**
- 大规模训练
- 长时间实验
- 生产环境

#### **本地CPU（不推荐）**

**限制：**
- 训练极慢
- 无法运行7B模型
- 仅适合测试代码逻辑

### 安装依赖

```bash
# 1. 基础依赖
pip install torch transformers accelerate
pip install stable-baselines3
pip install peft  # LoRA支持
pip install scikit-learn pandas matplotlib

# 2. BEAR仿真器（如果可用）
# 假设BEAR在 /path/to/BEAR
pip install -e /path/to/BEAR

# 3. 验证安装
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 项目结构

### 完整目录树

```
hvac-llm-project/
│
├── BEAR/                          # BEAR仿真器目录
│   └── Data/                      # 建筑和气候数据
│       ├── OfficeSmall/           # 办公楼数据
│       ├── Hot_Dry/               # 气候数据
│       └── USA_AZ_Tucson.../      # 天气文件 (.epw)
│
├── core_modules/                  # 核心Python模块
│   ├── llm_agent_colab.py
│   ├── prompt_builder_control.py
│   ├── few_shot_auto.py
│   ├── recorder_v2.py
│   ├── select_representative.py
│   ├── ppo_collect.py
│   ├── draw_reward.py
│   ├── config_manager.py
│   └── 7b_finetune_fixed.py
│
├── pipeline_output/               # Pipeline输出（自动创建）
│   ├── 01_ppo_training/
│   │   ├── ppo_trajectory.json    # PPO轨迹数据
│   │   ├── ppo_final.zip          # 训练好的模型
│   │   └── checkpoints/           # 训练检查点
│   │
│   ├── 02_few_shot_samples/
│   │   └── few_shot_examples_structured.json
│   │
│   ├── 03_llm_rollout/
│   │   └── llm_rollout.json       # LLM控制轨迹
│   │
│   ├── 04_finetuning/
│   │   ├── final_model/           # 微调后的模型
│   │   └── checkpoints/
│   │
│   └── 05_evaluation/
│       ├── results.json
│       └── comparison_plot.png
│
├── main_pipeline.py               # 主程序（一键运行）
├── config.json                    # 配置文件
├── test_suite.py                  # 测试工具
│
└── docs/                          # 文档
    ├── README.md
    ├── SETUP_TUTORIAL.md          # 本文件
    ├── QUICK_REFERENCE.md
    └── IMPROVEMENTS_SUMMARY.md
```

---

## 初次设置

### Step 1: 下载所有文件

1. **下载核心模块** (从之前的FILE_INDEX.md)：
   - llm_agent_colab.py
   - prompt_builder_control.py
   - few_shot_auto.py
   - recorder_v2.py
   - select_representative.py
   - ppo_collect.py
   - draw_reward.py
   - config_manager.py
   - 7b_finetune_fixed.py (使用修复版本)
   - main_pipeline.py
   - test_suite.py

2. **创建项目结构**：
```bash
# 创建主目录
mkdir hvac-llm-project
cd hvac-llm-project

# 创建子目录
mkdir core_modules
mkdir docs

# 移动文件到对应位置
mv *.py core_modules/
mv main_pipeline.py ./
mv test_suite.py ./
```

### Step 2: 准备BEAR数据

```bash
# 假设BEAR已安装
# 确认数据文件存在
ls BEAR/Data/

# 应该看到：
# OfficeSmall/
# Hot_Dry/
# 天气文件（.epw）
```

**如果没有BEAR：**
- 可以使用模拟数据进行测试
- 或者联系BEAR开发者获取

### Step 3: 创建配置文件

```bash
cd hvac-llm-project
python -c "
from core_modules.config_manager import get_default_config
config = get_default_config('development')
config.save_to_file('config.json')
"
```

这会创建一个 `config.json`：
```json
{
  "env": "development",
  "llm": {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0.7,
    ...
  },
  "hvac": {
    "building": "OfficeSmall",
    "climate": "Hot_Dry",
    "location": "Tucson"
  },
  ...
}
```

### Step 4: 验证环境

```bash
# 运行测试
python test_suite.py

# 应该看到所有测试通过
```

### Step 5: 设置HuggingFace Token（可选）

如果需要访问Qwen模型：

```bash
# 方法1：环境变量
export HF_TOKEN="your_huggingface_token"

# 方法2：在Colab中
import os
os.environ["HF_TOKEN"] = "your_token"

# 获取token: https://huggingface.co/settings/tokens
```

---

## 使用Pipeline

### 方式1：一键运行完整流程

```bash
# 运行所有阶段
python main_pipeline.py --stage all

# 使用自定义配置
python main_pipeline.py --config my_config.json --stage all
```

### 方式2：分阶段运行

```bash
# 阶段1：PPO训练（收集数据）
python main_pipeline.py --stage ppo

# 阶段2：选择代表性样本
python main_pipeline.py --stage select

# 阶段3：LLM rollout
python main_pipeline.py --stage rollout

# 阶段4：微调模型
python main_pipeline.py --stage finetune

# 阶段5：评估对比
python main_pipeline.py --stage eval
```

### 方式3：直接调用模块

```bash
# 单独运行PPO训练
cd core_modules
export BUILDING="OfficeSmall"
export WEATHER="Hot_Dry"
export TOTAL_STEPS="500000"
python ppo_collect.py

# 单独运行样本选择
python select_representative.py \
    --traj ../pipeline_output/01_ppo_training/ppo_trajectory.json \
    --out_dir ../pipeline_output/02_few_shot_samples \
    --clusters 12

# 单独微调
export BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
export ROLLOUT_GLOBS="../pipeline_output/03_llm_rollout/llm_rollout.json"
export SAVE_DIR="../pipeline_output/04_finetuning"
python 7b_finetune_fixed.py
```

### 在Google Colab中运行

```python
# 1. 安装依赖
!pip install -q transformers accelerate peft stable-baselines3 scikit-learn

# 2. 上传文件
from google.colab import files
# 上传所有.py文件和config.json

# 3. 挂载Google Drive（保存结果）
from google.colab import drive
drive.mount('/content/drive')

# 4. 运行pipeline
!python main_pipeline.py --stage all

# 5. 保存结果到Drive
!cp -r pipeline_output /content/drive/MyDrive/hvac_results
```

---

## 添加新数据

### 场景：获得新的建筑/天气数据

假设你有新的 `.htm` 文件（BEAR天气数据）或新的建筑配置：

#### Step 1: 准备数据文件

```bash
# 将新的天气文件放到BEAR/Data目录
cp new_weather.epw BEAR/Data/USA_XX_NewLocation.epw

# 或者新的建筑配置
cp -r NewBuildingType/ BEAR/Data/
```

#### Step 2: 更新配置

```json
{
  "hvac": {
    "building": "NewBuildingType",  // 更新
    "climate": "New_Climate",       // 更新
    "location": "NewLocation"       // 更新
  }
}
```

#### Step 3: 重新运行Pipeline

```bash
# 使用新配置运行
python main_pipeline.py --config new_config.json --stage all

# 或者直接指定参数
python main_pipeline.py \
    --building NewBuildingType \
    --weather New_Climate \
    --stage all
```

#### Step 4: 结果会保存到新的目录

```
pipeline_output/
├── 01_ppo_training/
│   └── ppo_trajectory.json  # 新数据的轨迹
├── 02_few_shot_samples/
│   └── few_shot_examples.json  # 新数据的示例
...
```

### 处理多个数据集

```bash
# 方法1：使用不同的base_dir
python main_pipeline.py --stage all  # 使用默认 ./pipeline_output

# 修改config.json中的base_dir
{
  "base_dir": "./experiments/experiment_1"
}

# 方法2：为每个实验创建目录
mkdir experiments
cd experiments

mkdir office_hot
cd office_hot
python ../../main_pipeline.py \
    --building OfficeSmall \
    --weather Hot_Dry \
    --stage all
cd ..

mkdir office_cold
cd office_cold
python ../../main_pipeline.py \
    --building OfficeSmall \
    --weather Cold_Dry \
    --stage all
```

---

## 运行环境建议

### 对于不同阶段的推荐环境

| 阶段 | 推荐环境 | GPU需求 | 估计时间 | 备注 |
|------|----------|---------|----------|------|
| PPO训练 | Colab/本地GPU | T4+ (4GB+) | 2-4小时 | 可中断恢复 |
| 样本选择 | CPU即可 | 无 | <5分钟 | 纯计算 |
| LLM Rollout | Colab/本地GPU | T4+ (8GB+) | 1-2小时 | 需要LLM |
| 微调 | Colab Pro (A100) | A100 (40GB) | 4-8小时 | 最耗资源 |
| 评估 | CPU即可 | 无 | <5分钟 | 绘图 |

### Google Colab详细配置

#### Colab免费版 (T4 GPU)

**适合：**
- PPO训练（小规模）
- LLM推理
- 微调（需要减小batch size）

**限制：**
- 12小时运行限制
- T4 GPU (16GB)
- 可能被中断

**优化技巧：**
```python
# 启用高RAM模式
# Runtime > Change runtime type > High-RAM

# 定期保存
import time
while training:
    # 每30分钟保存一次
    if time.time() % 1800 < 60:
        model.save("checkpoint.zip")
```

#### Colab Pro (A100 GPU)

**适合：**
- 所有阶段
- 大规模训练
- 7B模型微调

**优势：**
- A100 40GB
- 24小时运行
- 更快速度

**推荐：**
```python
# 确认GPU类型
!nvidia-smi

# 如果不是A100，可以重新连接
# Runtime > Disconnect and delete runtime
# Runtime > Run all
```

### 本地环境配置

#### 最低配置

```
CPU: 8核+
RAM: 32GB+
GPU: RTX 3090 (24GB)
存储: 100GB SSD
```

#### 推荐配置

```
CPU: 16核+
RAM: 64GB+
GPU: A100 (40GB) 或 多卡
存储: 500GB NVMe SSD
```

#### Docker环境（推荐）

```bash
# 使用NVIDIA Docker
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -p 8888:8888 \
    nvcr.io/nvidia/pytorch:23.10-py3

# 进入容器后
cd /workspace
pip install -r requirements.txt
```

---

## 常见问题

### Q1: 运行时内存不足

**问题：** CUDA out of memory

**解决：**
```bash
# 方法1：减小batch size
export BATCH_SIZE="1"
export GRAD_ACCUM="16"  # 增加梯度累积

# 方法2：使用更小的模型
export BASE_MODEL="Qwen/Qwen2-1.8B-Instruct"

# 方法3：启用梯度检查点（代码中已启用）
```

### Q2: HuggingFace模型下载失败

**问题：** Connection timeout

**解决：**
```bash
# 方法1：使用镜像
export HF_ENDPOINT="https://hf-mirror.com"

# 方法2：手动下载模型
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

# 方法3：使用本地路径
export MODEL_NAME="/path/to/local/model"
```

### Q3: PPO训练不收敛

**问题：** Reward不下降

**解决：**
```python
# 检查奖励函数
# 查看 trajectory.json 确认reward范围

# 调整PPO超参数
config.training.learning_rate = 5e-5  # 降低学习率
config.training.gamma = 0.95  # 调整折扣因子
```

### Q4: LLM输出格式错误

**问题：** Actions解析失败

**解决：**
```python
# 使用strict模式
actions, meta = parse_actions(text, n=3, strict=True)

# 检查temperature
config.llm.temperature = 0.3  # 降低randomness

# 添加更多few-shot examples
config.fewshot.k = 5  # 增加示例数量
```

### Q5: 微调后性能下降

**问题：** 微调后reward更差

**可能原因：**
1. 使用了旧版的finetune代码
   - 解决：使用 `7b_finetune_fixed.py`

2. 训练数据质量差
   - 解决：提高 `min_reward_percentile`

3. 过拟合
   - 解决：减少 `epochs`，增加 `entropy_coef`

4. Learning rate太大
   - 解决：降低到 `1e-6`

### Q6: Colab断开连接

**问题：** Session terminated

**解决：**
```python
# 方法1：定期保存到Google Drive
import time
while training:
    time.sleep(1800)  # 每30分钟
    !cp -r pipeline_output /content/drive/MyDrive/backup

# 方法2：使用tmux（本地服务器）
tmux new -s training
python main_pipeline.py --stage all
# Ctrl+B, D 离开
# tmux attach -t training  # 重新连接

# 方法3：使用nohup
nohup python main_pipeline.py --stage all > output.log 2>&1 &
```

---

## 检查清单

### 初次运行前

- [ ] 所有Python文件已下载
- [ ] BEAR数据可用
- [ ] GPU可用（`nvidia-smi`）
- [ ] 依赖已安装
- [ ] 测试通过（`python test_suite.py`）
- [ ] HuggingFace token已设置（如需要）
- [ ] config.json已创建

### 运行中

- [ ] 监控GPU使用率
- [ ] 定期检查日志
- [ ] 保存中间结果
- [ ] 记录实验参数

### 运行后

- [ ] 检查所有输出文件
- [ ] 绘制对比图
- [ ] 分析结果
- [ ] 备份重要数据

---

## 下一步

1. **运行测试**
   ```bash
   python test_suite.py
   ```

2. **小规模测试**
   ```bash
   # 减少训练步数测试
   python main_pipeline.py --stage ppo
   # 修改config: total_steps = 1000
   ```

3. **完整运行**
   ```bash
   python main_pipeline.py --stage all
   ```

4. **分析结果**
   ```bash
   # 查看生成的图表
   open pipeline_output/05_evaluation/comparison_plot.png
   ```

---

## 快速开始指南

对于急于开始的用户：

```bash
# 1. 下载所有文件

# 2. 运行测试
python test_suite.py

# 3. 一键运行
python main_pipeline.py --stage all

# 完成！结果在 pipeline_output/
```

---

**祝实验顺利！如有问题，请参考本教程或查看其他文档。**
