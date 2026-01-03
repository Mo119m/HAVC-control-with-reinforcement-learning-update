#!/usr/bin/env python
"""
环境验证脚本
用于检查所有依赖是否正确安装
"""

import sys
import importlib

def check_package(package_name, import_name=None):
    """检查包是否可以导入"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:20s} - version: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name:20s} - NOT FOUND: {e}")
        return False

def main():
    print("="*70)
    print("HVAC-RL 环境验证")
    print("="*70)
    print()

    # 基础包
    print("📦 检查基础依赖:")
    print("-"*70)
    basic_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('tqdm', 'tqdm'),
    ]

    basic_ok = all(check_package(name, import_name) for name, import_name in basic_packages)
    print()

    # 深度学习包
    print("🧠 检查深度学习依赖:")
    print("-"*70)
    dl_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('accelerate', 'accelerate'),
        ('peft', 'peft'),
    ]

    dl_ok = all(check_package(name, import_name) for name, import_name in dl_packages)
    print()

    # 强化学习包
    print("🎮 检查强化学习依赖:")
    print("-"*70)
    rl_packages = [
        ('stable-baselines3', 'stable_baselines3'),
        ('sb3-contrib', 'sb3_contrib'),
        ('gymnasium', 'gymnasium'),
    ]

    rl_ok = all(check_package(name, import_name) for name, import_name in rl_packages)
    print()

    # 其他依赖
    print("🌤️  检查其他依赖:")
    print("-"*70)
    other_packages = [
        ('pvlib', 'pvlib'),
    ]

    other_ok = all(check_package(name, import_name) for name, import_name in other_packages)
    print()

    # BEAR模块
    print("🏢 检查BEAR仿真器:")
    print("-"*70)
    bear_ok = True
    try:
        from BEAR.Env.env_building import BuildingEnvReal
        from BEAR.Utils.utils_building import ParameterGenerator
        print("✅ BEAR.Env.env_building  - OK")
        print("✅ BEAR.Utils.utils_building - OK")
    except ImportError as e:
        print(f"❌ BEAR模块 - NOT FOUND: {e}")
        bear_ok = False
    print()

    # Core modules
    print("🔧 检查核心模块:")
    print("-"*70)
    core_ok = True
    core_modules = [
        'core_modules.config_manager',
        'core_modules.ppo_collect',
        'core_modules.recorder_v2',
        'core_modules.main_pipeline',
    ]

    for module_name in core_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name:35s} - OK")
        except ImportError as e:
            print(f"❌ {module_name:35s} - FAILED: {e}")
            core_ok = False
    print()

    # GPU检查
    print("🎮 检查GPU支持:")
    print("-"*70)
    gpu_ok = False
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            gpu_ok = True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
    except:
        print("❌ Cannot check GPU status")
    print()

    # 总结
    print("="*70)
    print("验证总结:")
    print("="*70)
    all_ok = basic_ok and dl_ok and rl_ok and other_ok and bear_ok and core_ok

    if all_ok:
        print("✅ 所有依赖和模块都已正确安装！")
        print()
        print("你可以运行以下命令开始训练:")
        print("  python core_modules/main_pipeline.py --stage all")
        return 0
    else:
        print("❌ 部分依赖或模块缺失，请检查上面的错误信息")
        print()
        if not (basic_ok and dl_ok and rl_ok and other_ok):
            print("请运行以下命令安装依赖:")
            print("  pip install -r requirements.txt")
        if not bear_ok:
            print("请检查BEAR模块是否正确安装:")
            print("  ls -la BEAR/")
        return 1

if __name__ == "__main__":
    sys.exit(main())
