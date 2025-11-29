"""
Trajectory Recorder for PPO Collection - Optimized Version

This module records environment interactions during PPO training.
Handles action clipping, terminal observation extraction, and JSON export.

Key Features:
- Safe action clipping to environment bounds
- Terminal observation extraction
- Robust error handling
- Clean JSON export
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrajectoryRecorder(BaseCallback):
    """
    Callback for recording trajectory data during PPO training.
    
    Attributes:
        save_path: Path to save trajectory JSON
        buffer: List to accumulate trajectory steps
        prev_obs: Previous observation for transition tracking
        verbose: Logging verbosity level
    """
    
    def __init__(
        self,
        save_path: str = "ppo_trajectory.json",
        verbose: int = 1,
        auto_save_interval: Optional[int] = None
    ):
        """
        Initialize trajectory recorder.
        
        Args:
            save_path: Path to save JSON file
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
            auto_save_interval: Auto-save every N steps (None to disable)
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.buffer: List[Dict[str, Any]] = []
        self.prev_obs: Optional[Any] = None
        self.auto_save_interval = auto_save_interval
        self.step_count = 0
        
        # Ensure save directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    @staticmethod
    def _to_list(x: Any) -> List[float]:
        """
        Convert various types to Python list of floats.
        
        Args:
            x: Input (can be tensor, numpy array, list, or scalar)
            
        Returns:
            List of floats
        """
        # Handle PyTorch tensors
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        
        # Handle numpy arrays and lists
        if hasattr(x, "tolist"):
            result = x.tolist()
            # Ensure we return a list
            if not isinstance(result, list):
                return [float(result)]
            return [float(v) for v in result]
        
        # Handle scalars
        try:
            return [float(x)]
        except (ValueError, TypeError):
            # Last resort: convert to list
            try:
                return list(x)
            except TypeError:
                logger.error(f"Cannot convert to list: {type(x)}")
                return []
    
    def _get_current_obs(self) -> List[float]:
        """
        Extract current observation from environment.
        
        Tries multiple methods to get the observation:
        1. Environment state attribute
        2. new_obs from locals
        3. observations from locals
        
        Returns:
            Current observation as list
            
        Raises:
            RuntimeError: If observation cannot be extracted
        """
        # Try environment state
        try:
            state = self.training_env.get_attr("state")[0]
            return self._to_list(state)
        except (AttributeError, IndexError, TypeError) as e:
            logger.debug(f"Could not get env.state: {e}")
        
        # Try new_obs from locals
        new_obs = self.locals.get("new_obs")
        if new_obs is not None:
            try:
                return self._to_list(new_obs[0])
            except (IndexError, TypeError):
                return self._to_list(new_obs)
        
        # Try observations from locals
        observations = self.locals.get("observations")
        if observations is not None:
            try:
                return self._to_list(observations[0])
            except (IndexError, TypeError):
                return self._to_list(observations)
        
        raise RuntimeError(
            "Cannot extract observation. "
            "env.state, new_obs, and observations are all unavailable."
        )
    
    def _clip_action(
        self,
        raw_action: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """
        Clip action to environment action space bounds.
        
        Args:
            raw_action: Raw action from policy
            
        Returns:
            Tuple of (clipped_action, raw_action) as lists
        """
        raw_arr = np.asarray(raw_action, dtype=float)
        
        # Get action space bounds
        try:
            low = np.asarray(self.training_env.action_space.low, dtype=float)
            high = np.asarray(self.training_env.action_space.high, dtype=float)
        except (AttributeError, TypeError):
            logger.warning("Could not get action space bounds, using [-1, 1]")
            low = -np.ones_like(raw_arr, dtype=float)
            high = np.ones_like(raw_arr, dtype=float)
        
        # Clip
        clipped = np.clip(raw_arr, low, high)
        
        return self._to_list(clipped), self._to_list(raw_arr)
    
    def _extract_terminal_obs(
        self,
        infos: Union[List[Dict], Dict]
    ) -> Optional[List[float]]:
        """
        Extract terminal observation from info dict.
        
        Args:
            infos: Info dict or list of info dicts
            
        Returns:
            Terminal observation if available, else None
        """
        # Handle list of infos
        if isinstance(infos, (list, tuple)):
            if len(infos) == 0:
                return None
            info = infos[0]
        elif isinstance(infos, dict):
            info = infos
        else:
            logger.warning(f"Unexpected infos type: {type(infos)}")
            return None
        
        # Try different field names
        for field in ["terminal_observation", "final_observation"]:
            term_obs = info.get(field)
            if term_obs is not None:
                try:
                    return self._to_list(term_obs)
                except Exception as e:
                    logger.warning(f"Failed to convert {field}: {e}")
        
        return None
    
    def _on_rollout_start(self) -> bool:
        """Called before collecting new rollout"""
        try:
            self.prev_obs = self._get_current_obs()
        except RuntimeError as e:
            logger.error(f"Failed to get initial observation: {e}")
            # Don't stop training, just log error
            self.prev_obs = []
        return True
    
    def _on_step(self) -> bool:
        """
        Called after each step.
        Records transition (s, a, r, s', done, terminal_obs).
        
        Returns:
            True to continue training
        """
        try:
            # Extract action (raw and clipped)
            raw_action = self.locals["actions"][0]
            action_clipped, action_raw = self._clip_action(raw_action)
            
            # Extract reward and done
            reward = float(self.locals["rewards"][0])
            done = bool(self.locals["dones"][0])
            
            # Extract next observation
            next_obs = self._get_current_obs()
            
            # Extract terminal observation if done
            terminal_obs = None
            if done:
                infos = self.locals.get("infos", [])
                terminal_obs = self._extract_terminal_obs(infos)
            
            # Record transition
            transition = {
                "obs": self.prev_obs,
                "action": action_clipped,
                "action_raw": action_raw,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "terminal_obs": terminal_obs
            }
            
            self.buffer.append(transition)
            self.step_count += 1
            
            # Auto-save if enabled
            if (self.auto_save_interval is not None and 
                self.step_count % self.auto_save_interval == 0):
                self._save_buffer(incremental=True)
            
            # Update previous observation
            self.prev_obs = next_obs
            
        except Exception as e:
            logger.error(f"Error in _on_step: {e}", exc_info=True)
            # Don't stop training on error
        
        return True
    
    def _save_buffer(self, incremental: bool = False) -> None:
        """
        Save buffer to JSON file.
        
        Args:
            incremental: If True, append to existing file
        """
        try:
            save_path = self.save_path
            
            # For incremental saves, use temporary path
            if incremental:
                save_path = self.save_path.replace(".json", "_partial.json")
            
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.buffer, f, indent=2, ensure_ascii=False)
            
            if self.verbose >= 1:
                logger.info(
                    f"Saved {len(self.buffer)} transitions to {save_path}"
                )
        
        except Exception as e:
            logger.error(f"Failed to save buffer: {e}", exc_info=True)
    
    def _on_training_end(self) -> None:
        """Called when training ends. Save final buffer."""
        self._save_buffer(incremental=False)
        
        if self.verbose >= 1:
            logger.info(f"Training complete. Total steps recorded: {len(self.buffer)}")


# ============================================================================
# Utility Functions
# ============================================================================

def load_trajectory(path: str) -> List[Dict[str, Any]]:
    """
    Load trajectory from JSON file.
    
    Args:
        path: Path to trajectory JSON
        
    Returns:
        List of transition dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Trajectory must be a list of transitions")
        
        logger.info(f"Loaded {len(data)} transitions from {path}")
        return data
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")


def validate_trajectory(trajectory: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Validate trajectory structure.
    
    Args:
        trajectory: List of transitions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not trajectory:
        return False, "Empty trajectory"
    
    required_fields = ["obs", "action", "reward", "next_obs", "done"]
    
    for i, trans in enumerate(trajectory):
        # Check required fields
        for field in required_fields:
            if field not in trans:
                return False, f"Transition {i} missing field: {field}"
        
        # Validate types
        if not isinstance(trans["obs"], list):
            return False, f"Transition {i}: obs must be a list"
        if not isinstance(trans["action"], list):
            return False, f"Transition {i}: action must be a list"
        if not isinstance(trans["next_obs"], list):
            return False, f"Transition {i}: next_obs must be a list"
        if not isinstance(trans["done"], bool):
            return False, f"Transition {i}: done must be a boolean"
        
        try:
            float(trans["reward"])
        except (ValueError, TypeError):
            return False, f"Transition {i}: reward must be numeric"
    
    return True, "Valid"


def summarize_trajectory(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for trajectory.
    
    Args:
        trajectory: List of transitions
        
    Returns:
        Dictionary of statistics
    """
    if not trajectory:
        return {"length": 0}
    
    rewards = [t["reward"] for t in trajectory]
    
    # Count episodes
    episodes = sum(1 for t in trajectory if t["done"])
    
    return {
        "length": len(trajectory),
        "episodes": episodes,
        "total_reward": sum(rewards),
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
    }


if __name__ == "__main__":
    # Test trajectory loading and validation
    print("Trajectory Recorder Test")
    print("=" * 50)
    
    # Create sample trajectory
    sample_traj = [
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
    
    # Validate
    valid, msg = validate_trajectory(sample_traj)
    print(f"Validation: {valid} - {msg}")
    
    # Summarize
    if valid:
        summary = summarize_trajectory(sample_traj)
        print("\nTrajectory Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
