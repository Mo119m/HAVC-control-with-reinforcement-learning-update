#!/usr/bin/env python
"""
Environment verification script.
Checks that all dependencies are installed correctly.
"""

import sys
import importlib


def check_package(package_name, import_name=None):
    """Check whether a package can be imported."""
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
    print("=" * 70)
    print("HVAC-RL environment verification")
    print("=" * 70)
    print()

    # Basic packages
    print("📦 Checking basic dependencies:")
    print("-" * 70)
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

    # Deep-learning packages
    print("🧠 Checking deep-learning dependencies:")
    print("-" * 70)
    dl_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('accelerate', 'accelerate'),
        ('peft', 'peft'),
    ]

    dl_ok = all(check_package(name, import_name) for name, import_name in dl_packages)
    print()

    # Reinforcement-learning packages
    print("🎮 Checking reinforcement-learning dependencies:")
    print("-" * 70)
    rl_packages = [
        ('stable-baselines3', 'stable_baselines3'),
        ('sb3-contrib', 'sb3_contrib'),
        ('gymnasium', 'gymnasium'),
    ]

    rl_ok = all(check_package(name, import_name) for name, import_name in rl_packages)
    print()

    # Other dependencies
    print("🌤️  Checking other dependencies:")
    print("-" * 70)
    other_packages = [
        ('pvlib', 'pvlib'),
    ]

    other_ok = all(check_package(name, import_name) for name, import_name in other_packages)
    print()

    # BEAR modules
    print("🏢 Checking the BEAR simulator:")
    print("-" * 70)
    bear_ok = True
    try:
        from BEAR.Env.env_building import BuildingEnvReal
        from BEAR.Utils.utils_building import ParameterGenerator
        print("✅ BEAR.Env.env_building  - OK")
        print("✅ BEAR.Utils.utils_building - OK")
    except ImportError as e:
        print(f"❌ BEAR modules - NOT FOUND: {e}")
        bear_ok = False
    print()

    # Core modules
    print("🔧 Checking core modules:")
    print("-" * 70)
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

    # GPU check
    print("🎮 Checking GPU support:")
    print("-" * 70)
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
    except Exception:
        print("❌ Cannot check GPU status")
    print()

    # Summary
    print("=" * 70)
    print("Verification summary:")
    print("=" * 70)
    all_ok = basic_ok and dl_ok and rl_ok and other_ok and bear_ok and core_ok

    if all_ok:
        print("✅ All dependencies and modules are installed correctly!")
        print()
        print("You can start the pipeline with:")
        print("  python core_modules/main_pipeline.py --stage all")
        return 0
    else:
        print("❌ Some dependencies or modules are missing; see the errors above.")
        print()
        if not (basic_ok and dl_ok and rl_ok and other_ok):
            print("Install dependencies with:")
            print("  pip install -r requirements.txt")
        if not bear_ok:
            print("Check that the BEAR modules are present:")
            print("  ls -la BEAR/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
