# ==============================================================================
# HVAC-RL Complete Pipeline for Google Colab
# ==============================================================================
#
# 这是一个完整的、修复了所有 bug 的 pipeline
# 按顺序运行每个 cell 即可
#
# 流程: PPO训练 → 样本选择 → LLM Rollout → Fine-tuning → 评估
# ==============================================================================

# %% [markdown]
# # 🏠 HVAC-RL 完整 Pipeline
#
# ## 运行说明
# 1. 确保使用 GPU 运行时 (Runtime > Change runtime type > GPU)
# 2. 按顺序运行每个 cell
# 3. 数据会自动保存到 Google Drive

# %% [markdown]
# ## Step 1: 环境设置

# %%
# ===== 1.1 挂载 Google Drive =====
from google.colab import drive
drive.mount('/content/drive')

# 创建工作目录
import os
os.makedirs('/content/drive/MyDrive/rl', exist_ok=True)
print("✓ Google Drive 已挂载")

# %%
# ===== 1.2 克隆项目 =====
import os

if not os.path.exists("/content/HAVC-control-with-reinforcement-learning-update"):
    !git clone https://github.com/Mo119m/HAVC-control-with-reinforcement-learning-update.git
    print("✓ 项目已克隆")
else:
    print("✓ 项目已存在")

# 设置工作目录
PROJECT_ROOT = "/content/HAVC-control-with-reinforcement-learning-update"
os.chdir(PROJECT_ROOT)

# %%
# ===== 1.3 安装依赖 =====
!pip install -q stable-baselines3==2.1.0 gymnasium
!pip install -q transformers accelerate bitsandbytes peft
!pip install -q scikit-learn matplotlib tqdm
!pip install -q pvlib  # BEAR 环境需要

print("✓ 依赖安装完成")

# %%
# ===== 1.4 设置 Python 路径 =====
import sys
PROJECT_ROOT = "/content/HAVC-control-with-reinforcement-learning-update"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 验证 BEAR 可用
try:
    from BEAR.Env.env_building import BuildingEnvReal
    print("✓ BEAR 环境已加载")
except ImportError as e:
    print(f"✗ BEAR 导入失败: {e}")

# %%
# ===== 1.5 设置数据路径 =====
import os
import shutil

# 可能的数据路径（按优先级）
POSSIBLE_DATA_PATHS = [
    "/content/drive/MyDrive/rl/Data",
    "/content/drive/MyDrive/HAVC-control-with-reinforcement-learning-update/Data",
    "/content/drive/MyDrive/Data",
]

BEAR_DATA = "/content/HAVC-control-with-reinforcement-learning-update/BEAR/Data"
os.makedirs(BEAR_DATA, exist_ok=True)

# 查找数据源
data_found = False
for data_path in POSSIBLE_DATA_PATHS:
    if os.path.exists(data_path):
        print(f"✓ 找到数据: {data_path}")
        !cp -r "{data_path}"/* {BEAR_DATA}/
        print(f"✓ 数据已复制到 {BEAR_DATA}")
        data_found = True
        break

if not data_found:
    print("⚠ 未找到数据，请将数据上传到以下任一位置:")
    for p in POSSIBLE_DATA_PATHS:
        print(f"   - {p}")

# 显示数据内容
print(f"\nBEAR/Data 目录内容:")
!ls {BEAR_DATA}/

# %% [markdown]
# ## Step 2: PPO 训练

# %%
# ===== 2.1 PPO 训练配置 =====
PPO_CONFIG = {
    "total_timesteps": 200000,  # 训练步数 (可调整)
    "building": "OfficeSmall",
    "climate": "Hot_Dry",
    "save_dir": "/content/output_full/01_ppo",
}

# 创建输出目录
os.makedirs(PPO_CONFIG["save_dir"], exist_ok=True)
print(f"PPO 输出目录: {PPO_CONFIG['save_dir']}")

# %%
# ===== 2.2 运行 PPO 训练 =====
import os
os.chdir("/content/HAVC-control-with-reinforcement-learning-update/core_modules")

# 设置环境变量
os.environ["PYTHONPATH"] = "/content/HAVC-control-with-reinforcement-learning-update"

!python ppo_collect.py \
    --building {PPO_CONFIG["building"]} \
    --climate {PPO_CONFIG["climate"]} \
    --total_timesteps {PPO_CONFIG["total_timesteps"]} \
    --output_dir {PPO_CONFIG["save_dir"]}

# %%
# ===== 2.3 保存 PPO 结果到 Google Drive =====
import shutil

# 复制重要文件到 Google Drive
drive_ppo_dir = "/content/drive/MyDrive/rl/01_ppo"
os.makedirs(drive_ppo_dir, exist_ok=True)

# 复制轨迹文件
ppo_traj = f"{PPO_CONFIG['save_dir']}/ppo_trajectory.json"
if os.path.exists(ppo_traj):
    shutil.copy(ppo_traj, f"{drive_ppo_dir}/ppo_trajectory.json")
    print(f"✓ PPO 轨迹已保存到 Google Drive")

# 复制训练图
ppo_plot = f"{PPO_CONFIG['save_dir']}/training_results.png"
if os.path.exists(ppo_plot):
    shutil.copy(ppo_plot, f"{drive_ppo_dir}/training_results.png")
    print(f"✓ 训练图已保存到 Google Drive")

print(f"\n文件已保存到: {drive_ppo_dir}")
!ls -la {drive_ppo_dir}

# %% [markdown]
# ## Step 3: 样本选择

# %%
# ===== 3.1 样本选择配置 =====
SELECTION_CONFIG = {
    "traj_path": f"{PPO_CONFIG['save_dir']}/ppo_trajectory.json",
    "output_dir": "/content/output_full/02_few_shot",
    "preselect": 2000,
    "clusters": 12,
    "n_per_cluster": 20,
}

os.makedirs(SELECTION_CONFIG["output_dir"], exist_ok=True)

# %%
# ===== 3.2 运行样本选择 =====
os.chdir("/content/HAVC-control-with-reinforcement-learning-update/core_modules")

!python select_representative.py \
    --traj {SELECTION_CONFIG["traj_path"]} \
    --out_dir {SELECTION_CONFIG["output_dir"]} \
    --preselect {SELECTION_CONFIG["preselect"]} \
    --clusters {SELECTION_CONFIG["clusters"]} \
    --n_per_cluster {SELECTION_CONFIG["n_per_cluster"]} \
    --building "OfficeSmall" \
    --climate "Hot_Dry" \
    --location "Tucson"

# %%
# ===== 3.3 保存样本选择结果到 Google Drive =====
drive_fs_dir = "/content/drive/MyDrive/rl/02_few_shot"
os.makedirs(drive_fs_dir, exist_ok=True)

fs_file = f"{SELECTION_CONFIG['output_dir']}/few_shot_examples_structured.json"
if os.path.exists(fs_file):
    shutil.copy(fs_file, f"{drive_fs_dir}/few_shot_examples_structured.json")
    print(f"✓ Few-shot 样本已保存到 Google Drive")

# %% [markdown]
# ## Step 4: LLM Rollout

# %%
# ===== 4.1 LLM Rollout 配置 =====
ROLLOUT_CONFIG = {
    "fewshot_json": f"{SELECTION_CONFIG['output_dir']}/few_shot_examples_structured.json",
    "output_path": "/content/output_full/03_llm_rollout/llm_trajectory.json",
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "max_steps": 200,
    "building": "OfficeSmall",
    "climate": "Hot_Dry",
}

os.makedirs(os.path.dirname(ROLLOUT_CONFIG["output_path"]), exist_ok=True)

# %%
# ===== 4.2 运行 LLM Rollout =====
import os
import sys

# 确保路径正确
sys.path.insert(0, "/content/HAVC-control-with-reinforcement-learning-update")
os.chdir("/content/HAVC-control-with-reinforcement-learning-update/core_modules")

# 设置环境变量
os.environ["MODEL_NAME"] = ROLLOUT_CONFIG["model_name"]
os.environ["PYTHONPATH"] = "/content/HAVC-control-with-reinforcement-learning-update"

!PYTHONPATH=/content/HAVC-control-with-reinforcement-learning-update python rollout_fewshot_version.py \
    --fewshot_json {ROLLOUT_CONFIG["fewshot_json"]} \
    --output {ROLLOUT_CONFIG["output_path"]} \
    --building {ROLLOUT_CONFIG["building"]} \
    --climate {ROLLOUT_CONFIG["climate"]} \
    --max_steps {ROLLOUT_CONFIG["max_steps"]}

# %%
# ===== 4.3 保存 LLM Rollout 结果到 Google Drive =====
drive_rollout_dir = "/content/drive/MyDrive/rl/03_llm_rollout"
os.makedirs(drive_rollout_dir, exist_ok=True)

if os.path.exists(ROLLOUT_CONFIG["output_path"]):
    shutil.copy(ROLLOUT_CONFIG["output_path"], f"{drive_rollout_dir}/llm_trajectory.json")
    print(f"✓ LLM 轨迹已保存到 Google Drive")

# %% [markdown]
# ## Step 5: Fine-tuning (LoRA)

# %%
# ===== 5.1 Fine-tuning 配置 =====
FINETUNE_CONFIG = {
    "trajectory_path": ROLLOUT_CONFIG["output_path"],
    "output_dir": "/content/output_full/04_finetune",
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
}

os.makedirs(FINETUNE_CONFIG["output_dir"], exist_ok=True)

# %%
# ===== 5.2 运行 Fine-tuning =====
os.chdir("/content/HAVC-control-with-reinforcement-learning-update/core_modules")

!PYTHONPATH=/content/HAVC-control-with-reinforcement-learning-update python 7b_finetune_fixed.py \
    --trajectory_path {FINETUNE_CONFIG["trajectory_path"]} \
    --output_dir {FINETUNE_CONFIG["output_dir"]} \
    --model_name {FINETUNE_CONFIG["model_name"]} \
    --num_epochs {FINETUNE_CONFIG["num_epochs"]} \
    --batch_size {FINETUNE_CONFIG["batch_size"]} \
    --learning_rate {FINETUNE_CONFIG["learning_rate"]}

# %%
# ===== 5.3 保存 Fine-tuned 模型到 Google Drive =====
drive_ft_dir = "/content/drive/MyDrive/rl/04_finetune"
os.makedirs(drive_ft_dir, exist_ok=True)

# 复制 LoRA adapter
lora_dir = f"{FINETUNE_CONFIG['output_dir']}/lora_adapter"
if os.path.exists(lora_dir):
    !cp -r {lora_dir} {drive_ft_dir}/
    print(f"✓ LoRA adapter 已保存到 Google Drive")

# %% [markdown]
# ## Step 6: 评估 Fine-tuned 模型

# %%
# ===== 6.1 使用 Fine-tuned 模型进行 Rollout =====
EVAL_CONFIG = {
    "lora_path": f"{FINETUNE_CONFIG['output_dir']}/lora_adapter",
    "output_path": "/content/output_full/05_eval/finetuned_trajectory.json",
    "max_steps": 200,
}

os.makedirs(os.path.dirname(EVAL_CONFIG["output_path"]), exist_ok=True)

# %%
# ===== 6.2 运行评估 =====
os.chdir("/content/HAVC-control-with-reinforcement-learning-update/core_modules")

!PYTHONPATH=/content/HAVC-control-with-reinforcement-learning-update python 7Blora_rollout.py \
    --lora_path {EVAL_CONFIG["lora_path"]} \
    --output {EVAL_CONFIG["output_path"]} \
    --building "OfficeSmall" \
    --climate "Hot_Dry" \
    --max_steps {EVAL_CONFIG["max_steps"]}

# %% [markdown]
# ## Step 7: 结果对比

# %%
# ===== 7.1 对比 Before vs After =====
import json
import numpy as np

def load_trajectory_rewards(path):
    """加载轨迹并计算平均 reward"""
    if not os.path.exists(path):
        return None, None
    with open(path, 'r') as f:
        traj = json.load(f)
    rewards = [step.get('reward', 0) for step in traj]
    return np.mean(rewards), np.sum(rewards)

# 加载各阶段结果
ppo_mean, ppo_sum = load_trajectory_rewards(f"{PPO_CONFIG['save_dir']}/ppo_trajectory.json")
llm_mean, llm_sum = load_trajectory_rewards(ROLLOUT_CONFIG["output_path"])
ft_mean, ft_sum = load_trajectory_rewards(EVAL_CONFIG["output_path"])

print("=" * 50)
print("结果对比")
print("=" * 50)
print(f"{'模型':<20} {'平均 Reward':<15} {'总 Reward':<15}")
print("-" * 50)
if ppo_mean: print(f"{'PPO Expert':<20} {ppo_mean:<15.2f} {ppo_sum:<15.2f}")
if llm_mean: print(f"{'LLM (Before FT)':<20} {llm_mean:<15.2f} {llm_sum:<15.2f}")
if ft_mean:  print(f"{'LLM (After FT)':<20} {ft_mean:<15.2f} {ft_sum:<15.2f}")
print("=" * 50)

# %%
# ===== 7.2 可视化对比 =====
import matplotlib.pyplot as plt

def plot_comparison(ppo_path, llm_path, ft_path):
    """绘制对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trajectories = {}
    labels = ['PPO Expert', 'LLM Before FT', 'LLM After FT']
    paths = [ppo_path, llm_path, ft_path]
    colors = ['blue', 'orange', 'green']

    for label, path, color in zip(labels, paths, colors):
        if os.path.exists(path):
            with open(path, 'r') as f:
                traj = json.load(f)
            rewards = [step.get('reward', 0) for step in traj[:200]]

            # Cumulative reward
            axes[0].plot(np.cumsum(rewards), label=label, color=color, alpha=0.8)

            # Step reward
            axes[1].plot(rewards, label=label, color=color, alpha=0.5)

    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Cumulative Reward Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Step Reward')
    axes[1].set_title('Step Reward Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/content/output_full/comparison.png', dpi=150)
    plt.show()
    print("✓ 对比图已保存")

plot_comparison(
    f"{PPO_CONFIG['save_dir']}/ppo_trajectory.json",
    ROLLOUT_CONFIG["output_path"],
    EVAL_CONFIG["output_path"]
)

# %% [markdown]
# ## 🎉 完成！
#
# 所有结果已保存到 Google Drive:
# - `/content/drive/MyDrive/rl/01_ppo/` - PPO 训练结果
# - `/content/drive/MyDrive/rl/02_few_shot/` - Few-shot 样本
# - `/content/drive/MyDrive/rl/03_llm_rollout/` - LLM Rollout 结果
# - `/content/drive/MyDrive/rl/04_finetune/` - Fine-tuned 模型

# %% [markdown]
# ---
# ## 📋 快速恢复 (如果 Colab 断开)
#
# 如果 Colab 断开连接，运行以下代码恢复:

# %%
# ===== 快速恢复脚本 =====
"""
# 1. 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 克隆项目
!git clone https://github.com/Mo119m/HAVC-control-with-reinforcement-learning-update.git 2>/dev/null || echo "Already exists"

# 3. 安装依赖
!pip install -q stable-baselines3 gymnasium transformers accelerate bitsandbytes peft scikit-learn matplotlib

# 4. 设置路径
import sys
sys.path.insert(0, "/content/HAVC-control-with-reinforcement-learning-update")

# 5. 复制数据
!mkdir -p /content/HAVC-control-with-reinforcement-learning-update/BEAR/Data
!cp -r /content/drive/MyDrive/rl/Data/* /content/HAVC-control-with-reinforcement-learning-update/BEAR/Data/

# 6. 恢复之前的输出
!mkdir -p /content/output_full
!cp -r /content/drive/MyDrive/rl/01_ppo /content/output_full/ 2>/dev/null
!cp -r /content/drive/MyDrive/rl/02_few_shot /content/output_full/ 2>/dev/null
!cp -r /content/drive/MyDrive/rl/03_llm_rollout /content/output_full/ 2>/dev/null
!cp -r /content/drive/MyDrive/rl/04_finetune /content/output_full/ 2>/dev/null

print("✓ 环境已恢复，从断点继续运行即可")
"""
