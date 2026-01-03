#!/usr/bin/env python
"""
快速测试BEAR环境是否能正常工作
"""

import sys
import numpy as np
from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator

def test_bear_environment():
    """测试BEAR环境初始化和基本操作"""
    print("="*70)
    print("BEAR环境测试")
    print("="*70)

    try:
        # 创建参数生成器
        print("\n1. 创建环境参数...")
        param = ParameterGenerator(
            "OfficeLarge",  # 使用实际存在的数据文件
            "Hot_Dry",
            "Tucson",
            root="./BEAR/Data/"
        )
        print("✅ 参数生成器创建成功")

        # 创建环境
        print("\n2. 初始化BEAR环境...")
        env = BuildingEnvReal(param)
        print("✅ 环境初始化成功")

        # 检查环境属性
        print("\n3. 检查环境属性:")
        print(f"   观察空间: {env.observation_space}")
        print(f"   动作空间: {env.action_space}")
        print(f"   最大步数: {param.max_steps}")

        # 重置环境
        print("\n4. 重置环境...")
        obs, info = env.reset()
        print(f"✅ 环境重置成功")
        print(f"   观察维度: {obs.shape}")
        print(f"   观察范围: [{obs.min():.2f}, {obs.max():.2f}]")

        # 执行几步测试
        print("\n5. 执行测试步骤...")
        total_reward = 0
        for step in range(5):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"   步骤 {step+1}: reward={reward:.2f}, done={terminated or truncated}")

            if terminated or truncated:
                print("   环境终止")
                break

        print(f"\n✅ 累计奖励: {total_reward:.2f}")

        # 关闭环境
        env.close()
        print("\n✅ 环境关闭成功")

        print("\n" + "="*70)
        print("BEAR环境测试完成 - 所有测试通过！")
        print("="*70)
        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_bear_environment())
