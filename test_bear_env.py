#!/usr/bin/env python
"""
Quick smoke test that the BEAR environment works.
"""

import sys
import numpy as np
from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator


def test_bear_environment():
    """Test BEAR environment initialization and basic operations."""
    print("=" * 70)
    print("BEAR environment test")
    print("=" * 70)

    try:
        # Create the parameter generator
        print("\n1. Creating environment parameters...")
        param = ParameterGenerator(
            "OfficeLarge",  # use a data file that actually exists
            "Hot_Dry",
            "Tucson",
            root="./BEAR/Data/"
        )
        print("✅ Parameter generator created")

        # Create the environment
        print("\n2. Initializing the BEAR environment...")
        env = BuildingEnvReal(param)
        print("✅ Environment initialized")

        # Inspect environment properties
        print("\n3. Environment properties:")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        if isinstance(param, dict):
            print(f"   Max steps: {param.get('max_steps', 'N/A')}")
        else:
            print(f"   Max steps: {param.max_steps}")

        # Reset the environment
        print("\n4. Resetting the environment...")
        obs, info = env.reset()
        print("✅ Environment reset")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation range: [{obs.min():.2f}, {obs.max():.2f}]")

        # Run a few test steps
        print("\n5. Running test steps...")
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"   Step {step+1}: reward={reward:.2f}, done={terminated or truncated}")

            if terminated or truncated:
                print("   Environment terminated")
                break

        print(f"\n✅ Total reward: {total_reward:.2f}")

        # Close the environment
        env.close()
        print("\n✅ Environment closed")

        print("\n" + "=" * 70)
        print("BEAR environment test complete - all checks passed!")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_bear_environment())
