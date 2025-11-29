"""
Simple Test Suite for HVAC-LLM Project

This script runs basic tests on all major modules to verify they work correctly.
Not a replacement for comprehensive unit tests, but useful for quick validation.

Usage:
    python test_suite.py
    python test_suite.py --verbose
"""

import sys
import logging
from typing import List, Tuple, Callable
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    message: str
    error: str = ""


class TestSuite:
    """Simple test suite runner"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
    
    def run_test(
        self,
        name: str,
        test_func: Callable[[], Tuple[bool, str]]
    ) -> TestResult:
        """
        Run a single test function.
        
        Args:
            name: Test name
            test_func: Function that returns (passed, message)
            
        Returns:
            TestResult
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {name}")
        logger.info('=' * 60)
        
        try:
            passed, message = test_func()
            result = TestResult(name=name, passed=passed, message=message)
            
            if passed:
                logger.info("✅ PASSED")
            else:
                logger.error(f"❌ FAILED: {message}")
            
            if self.verbose:
                logger.info(f"Details: {message}")
            
        except Exception as e:
            result = TestResult(
                name=name,
                passed=False,
                message="Exception raised",
                error=str(e)
            )
            logger.error(f"❌ FAILED with exception: {e}")
            
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        for result in self.results:
            status = "✅" if result.passed else "❌"
            logger.info(f"{status} {result.name}")
            if not result.passed and result.error:
                logger.info(f"   Error: {result.error}")
        
        logger.info("=" * 60)
        logger.info(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)")
        logger.info("=" * 60)
        
        return passed == total


# ============================================================================
# Individual Tests
# ============================================================================

def test_llm_agent():
    """Test llm_agent_colab module"""
    try:
        from llm_agent_colab import parse_actions, parse_actions_with_validation, GenerationConfig
        
        # Test 1: Parse with brackets
        text1 = "Actions: [0.1, -0.2, 0.3]"
        actions1, meta1 = parse_actions(text1, n=3)
        assert actions1 == [0.1, -0.2, 0.3], "Failed to parse bracketed actions"
        assert meta1["parsed_from"] == "actions_line", "Wrong parsing method"
        
        # Test 2: Parse without brackets
        text2 = "Actions: 0.5, -0.3, 0.8"
        actions2, meta2 = parse_actions(text2, n=3)
        assert len(actions2) == 3, "Failed to parse non-bracketed actions"
        
        # Test 3: Validation with clipping
        text3 = "[2.0, -1.5, 0.5]"  # Out of range values
        actions3, meta3 = parse_actions_with_validation(text3, n=3)
        assert all(-1.0 <= a <= 1.0 for a in actions3), "Validation failed to clip"
        assert meta3.get("clipped") == True, "Should mark as clipped"
        
        # Test 4: GenerationConfig validation
        config = GenerationConfig(temperature=0.7, top_p=0.9)
        config.validate()  # Should not raise
        
        return True, "All parsing tests passed"
    
    except AssertionError as e:
        return False, str(e)
    except ImportError as e:
        return False, f"Import error: {e}"


def test_prompt_builder():
    """Test prompt_builder_control module"""
    try:
        from prompt_builder_control import (
            zone_count_from_obs, extract_env_terms, build_prompt,
            validate_observation, PromptConfig
        )
        
        # Test observation (3n+2 structure with n=3)
        obs = [22.5, 23.0, 21.8,  # temps (n=3)
               30.0,               # outside
               500, 450, 480,      # ghi (n=3)
               20.0,               # ground
               0.5, 0.3, 0.4]      # occupancy (n=3)
        
        # Test 1: Zone count
        n = zone_count_from_obs(obs)
        assert n == 3, f"Expected 3 zones, got {n}"
        
        # Test 2: Extract environmental terms
        env = extract_env_terms(obs)
        assert abs(env.outside_temp - 30.0) < 0.01, "Wrong outside temp"
        assert abs(env.ghi_avg - 476.67) < 1.0, "Wrong GHI average"
        
        # Test 3: Validate observation
        valid, msg = validate_observation(obs)
        assert valid, f"Observation should be valid: {msg}"
        
        # Test 4: Build prompt
        config = PromptConfig(history_lines=2, enable_reasoning=True)
        prompt = build_prompt(
            obs=obs,
            building="OfficeSmall",
            location="Tucson",
            climate="Hot_Dry",
            target=22.0,
            config=config
        )
        assert "OfficeSmall" in prompt, "Building not in prompt"
        assert "3 rooms" in prompt, "Room count not in prompt"
        assert len(prompt) > 100, "Prompt too short"
        
        return True, "All prompt building tests passed"
    
    except AssertionError as e:
        return False, str(e)
    except ImportError as e:
        return False, f"Import error: {e}"


def test_few_shot():
    """Test few_shot_auto module"""
    try:
        from few_shot_auto import (
            featurize_observation, weighted_euclidean_distance,
            compute_similarity, select_examples, SelectionConfig
        )
        
        obs1 = [22.0, 23.0, 21.5, 30.0, 500, 480, 490, 20.0, 0.4, 0.3, 0.3]
        obs2 = [22.5, 23.5, 22.0, 31.0, 510, 490, 500, 21.0, 0.5, 0.4, 0.4]
        
        # Test 1: Featurize
        feat1 = featurize_observation(obs1)
        assert len(feat1) == 7, f"Expected 7 features, got {len(feat1)}"  # 3 temps + 4 env
        
        # Test 2: Distance computation
        dist = weighted_euclidean_distance(feat1, featurize_observation(obs2))
        assert dist > 0, "Distance should be positive"
        
        # Test 3: Similarity
        sim = compute_similarity(feat1, featurize_observation(obs2))
        assert 0 < sim <= 1, f"Similarity {sim} not in (0,1]"
        
        # Test 4: Selection
        dataset = [
            {"obs": obs1, "actions": [0.1, -0.2, 0.0], "reward": -5.0},
            {"obs": obs2, "actions": [-0.3, -0.4, -0.2], "reward": -3.0},
        ]
        
        config = SelectionConfig(k=2, alpha=0.6)
        selected = select_examples(dataset, obs1, config=config)
        assert len(selected) <= 2, "Should select at most k examples"
        
        return True, "All few-shot tests passed"
    
    except AssertionError as e:
        return False, str(e)
    except ImportError as e:
        return False, f"Import error: {e}"


def test_recorder():
    """Test recorder_v2 module"""
    try:
        from recorder_v2 import validate_trajectory, summarize_trajectory
        
        # Create sample trajectory
        trajectory = [
            {
                "obs": [22.0, 23.0],
                "action": [0.1, -0.2],
                "action_raw": [0.1, -0.2],
                "reward": -5.2,
                "next_obs": [22.2, 22.8],
                "done": False,
                "terminal_obs": None
            },
            {
                "obs": [22.2, 22.8],
                "action": [0.0, -0.1],
                "action_raw": [0.0, -0.1],
                "reward": -3.1,
                "next_obs": [22.1, 22.7],
                "done": True,
                "terminal_obs": [22.1, 22.7]
            }
        ]
        
        # Test 1: Validation
        valid, msg = validate_trajectory(trajectory)
        assert valid, f"Trajectory should be valid: {msg}"
        
        # Test 2: Summary
        summary = summarize_trajectory(trajectory)
        assert summary["length"] == 2, "Wrong trajectory length"
        assert summary["episodes"] == 1, "Wrong episode count"
        assert abs(summary["mean_reward"] - (-4.15)) < 0.1, "Wrong mean reward"
        
        return True, "All recorder tests passed"
    
    except AssertionError as e:
        return False, str(e)
    except ImportError as e:
        return False, f"Import error: {e}"


def test_config():
    """Test config_manager module"""
    try:
        from config_manager import (
            ProjectConfig, LLMConfig, HVACConfig,
            get_default_config, Environment
        )
        
        # Test 1: Create default config
        config = get_default_config("development")
        assert config.env == Environment.DEVELOPMENT, "Wrong environment"
        
        # Test 2: Validation
        errors = config.validate()
        assert len(errors) == 0, f"Config has errors: {errors}"
        
        # Test 3: Individual configs
        llm_config = LLMConfig(temperature=0.7)
        assert 0.0 <= llm_config.temperature <= 2.0, "Invalid temperature"
        
        hvac_config = HVACConfig(building="OfficeSmall")
        assert hvac_config.building == "OfficeSmall", "Wrong building"
        
        # Test 4: Config modification
        config.llm.temperature = 0.5
        assert config.llm.temperature == 0.5, "Config modification failed"
        
        return True, "All config tests passed"
    
    except AssertionError as e:
        return False, str(e)
    except ImportError as e:
        return False, f"Import error: {e}"


def test_integration():
    """Integration test: prompt + few-shot + parsing"""
    try:
        from prompt_builder_control import build_prompt
        from few_shot_auto import select_examples, format_few_shot_block, inject_few_shot
        from llm_agent_colab import parse_actions
        
        # Sample observation
        obs = [22.5, 23.0, 21.8, 30.0, 500, 450, 480, 20.0, 0.5, 0.3, 0.4]
        
        # Build prompt
        prompt = build_prompt(
            obs=obs,
            building="OfficeSmall",
            location="Tucson",
            climate="Hot_Dry",
            target=22.0
        )
        
        # Create few-shot examples
        dataset = [
            {"obs": obs, "actions": [0.1, -0.2, 0.0], "reward": -5.0}
        ]
        
        selected = select_examples(dataset, obs, k=1)
        few_shot = format_few_shot_block(selected, target=22.0, n=3)
        
        # Inject
        prompt_with_fs = inject_few_shot(prompt, few_shot)
        assert len(prompt_with_fs) > len(prompt), "Few-shot not injected"
        
        # Parse a mock response
        mock_response = "Let me think... Actions: [0.1, -0.2, 0.0]"
        actions, meta = parse_actions(mock_response, n=3)
        assert len(actions) == 3, "Failed to parse integrated response"
        
        return True, "Integration test passed"
    
    except AssertionError as e:
        return False, str(e)
    except ImportError as e:
        return False, f"Import error: {e}"


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run test suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    suite = TestSuite(verbose=args.verbose)
    
    # Run all tests
    suite.run_test("LLM Agent", test_llm_agent)
    suite.run_test("Prompt Builder", test_prompt_builder)
    suite.run_test("Few-Shot Selection", test_few_shot)
    suite.run_test("Trajectory Recorder", test_recorder)
    suite.run_test("Config Manager", test_config)
    suite.run_test("Integration Test", test_integration)
    
    # Print summary
    all_passed = suite.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
