"""
Prompt Builder for HVAC Control - Optimized Version

This module constructs natural language prompts for HVAC control tasks.
Handles state descriptions, history formatting, and instruction generation.

Key Features:
- Automatic observation parsing (3n+2 structure)
- Natural language state descriptions
- History formatting with configurable window
- Type-safe implementation
"""

import re
from textwrap import dedent
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnvTerms:
    """Environmental conditions extracted from observation"""
    outside_temp: float
    ghi_avg: float
    ground_temp: float
    occ_sum_kw: float


@dataclass
class PromptConfig:
    """Configuration for prompt generation"""
    history_lines: int = 3
    enable_history: bool = True
    enable_reasoning: bool = True
    max_history_keep: int = 10


def zone_count_from_obs(obs: List[float]) -> int:
    """
    Calculate number of zones from observation vector.
    
    BEAR observation structure: 3n+2
    - temps: n values (room temperatures)
    - outside: 1 value
    - ghi: n values (per-zone irradiance)
    - ground: 1 value
    - occupancy: n values (per-zone occupancy power)
    
    Args:
        obs: Observation vector
        
    Returns:
        Number of zones (rooms)
    """
    if not obs or len(obs) < 3:
        logger.warning(f"Invalid observation length: {len(obs)}")
        return 1
    
    try:
        n = (len(obs) - 2) // 3
        return max(1, n)
    except Exception as e:
        logger.error(f"Error calculating zone count: {e}")
        return 1


def extract_env_terms(
    obs: List[float], 
    n_override: Optional[int] = None
) -> EnvTerms:
    """
    Extract environmental terms from observation vector.
    
    Args:
        obs: Observation vector
        n_override: Force specific number of zones
        
    Returns:
        EnvTerms object with environmental data
    """
    n = n_override if n_override is not None else zone_count_from_obs(obs)
    
    # Safe extraction with bounds checking
    def safe_get(index: int, default: float = 0.0) -> float:
        try:
            return float(obs[index]) if index < len(obs) else default
        except (IndexError, ValueError, TypeError):
            return default
    
    # Extract outside temperature
    outside_temp = safe_get(n)
    
    # Extract and average GHI values
    ghi_vals = []
    for i in range(n + 1, min(2 * n + 1, len(obs))):
        ghi_vals.append(safe_get(i))
    ghi_avg = sum(ghi_vals) / len(ghi_vals) if ghi_vals else 0.0
    
    # Extract ground temperature
    ground_idx = 2 * n + 1
    ground_temp = safe_get(ground_idx)
    
    # Extract and sum occupancy power
    occ_vals = []
    for i in range(2 * n + 2, min(3 * n + 2, len(obs))):
        occ_vals.append(safe_get(i))
    occ_sum_kw = sum(occ_vals) if occ_vals else 0.0
    
    return EnvTerms(
        outside_temp=outside_temp,
        ghi_avg=ghi_avg,
        ground_temp=ground_temp,
        occ_sum_kw=occ_sum_kw
    )


def format_number(x: float, decimals: int = 1) -> str:
    """Format number with specified decimal places"""
    return f"{float(x):.{decimals}f}"


def format_list(values: List[float], decimals: int = 1) -> str:
    """Format list of numbers as bracketed string"""
    formatted = [format_number(v, decimals) for v in values]
    return "[" + ", ".join(formatted) + "]"


def prettify_building_name(name: str) -> str:
    """
    Convert building name to readable format.
    
    Examples:
        "OfficeSmall" -> "Office Small"
        "office_small" -> "Office Small"
    """
    # Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ").strip()
    
    # Split camelCase
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    
    # Capitalize words
    return " ".join(word.capitalize() for word in name.split())


def prettify_climate_name(name: str) -> str:
    """
    Convert climate name to readable format.
    
    Examples:
        "Hot_Dry" -> "Hot and Dry"
        "hot_humid" -> "Hot and Humid"
    """
    name = name.strip().replace("_", " ")
    
    # Handle specific patterns
    replacements = {
        "Hot Dry": "Hot and Dry",
        "Hot Humid": "Hot and Humid",
        "Warm Humid": "Warm and Humid",
        "Cold Dry": "Cold and Dry"
    }
    
    # Apply replacements
    for old, new in replacements.items():
        if old.lower() in name.lower():
            name = re.sub(old, new, name, flags=re.IGNORECASE)
    
    # Capitalize
    return " ".join(word.capitalize() for word in name.split())


def format_history_entry(
    history_item: Dict[str, Any],
    n_zones: int,
    compact: bool = False
) -> str:
    """
    Format a single history entry as human-readable text.
    
    Args:
        history_item: Dictionary with step, action, reward, etc.
        n_zones: Number of zones for consistent formatting
        compact: If True, use more compact format
        
    Returns:
        Formatted history string
    """
    step = history_item.get("step", 0)
    action = history_item.get("action", [])
    reward = history_item.get("reward", 0.0)
    env_temp = history_item.get("env_temp")
    obs_before = history_item.get("obs_before", [])[:n_zones]
    obs_after = history_item.get("obs_after", [])[:n_zones]
    power = history_item.get("power", 0)
    
    # Format action with minimal decimal places
    action_str = format_list(action, decimals=2).replace(".00", "")
    
    if compact:
        return (
            f"Step {step}: Action={action_str}, "
            f"Reward={reward:.3g}, OutdoorT={format_number(env_temp) if env_temp else 'N/A'}"
        )
    
    return (
        f"Step {step}, Action: {action_str}, Reward: {reward:.6g}, "
        f"Env Temp: {format_number(env_temp) if env_temp else 'N/A'}, "
        f"Room Temp Before: {format_list(obs_before)}, "
        f"Room Temp After: {format_list(obs_after)}, Power: {power}"
    )


def format_history_block(
    history: List[Dict[str, Any]], 
    n_zones: int,
    max_lines: int = 3,
    compact: bool = False
) -> str:
    """
    Format history entries into readable block.
    
    Args:
        history: List of history dictionaries
        n_zones: Number of zones
        max_lines: Maximum number of lines to include
        compact: Use compact format
        
    Returns:
        Formatted history block or "None"
    """
    if not history:
        return "None"
    
    # Take most recent entries
    recent = history[-max_lines:] if max_lines > 0 else history
    
    lines = [format_history_entry(h, n_zones, compact) for h in recent]
    return "\n".join(lines)


def build_prompt(
    obs: List[float],
    building: str,
    location: str,
    climate: str,
    target: float,
    round_idx: int = 1,
    history: Optional[List[Dict[str, Any]]] = None,
    config: Optional[PromptConfig] = None,
    n_rooms: Optional[int] = None,
) -> str:
    """
    Build comprehensive HVAC control prompt.
    
    Args:
        obs: Current observation vector
        building: Building type (e.g., "OfficeSmall")
        location: Location name (e.g., "Tucson")
        climate: Climate type (e.g., "Hot_Dry")
        target: Target temperature in Celsius
        round_idx: Current round/step index
        history: Optional list of past actions and feedback
        config: Prompt configuration
        n_rooms: Override number of rooms
        
    Returns:
        Formatted prompt string
    """
    # Initialize config
    if config is None:
        config = PromptConfig()
    
    # Determine number of zones
    n = n_rooms if n_rooms is not None else zone_count_from_obs(obs)
    
    # Extract data
    temps = obs[:n]
    env = extract_env_terms(obs, n_override=n)
    
    # Prettify names
    building_pretty = prettify_building_name(building)
    climate_pretty = prettify_climate_name(climate)
    
    # Format room temperatures
    temp_lines = "\n".join([
        f"   Room {i+1}: {format_number(temps[i])} degrees Celsius"
        for i in range(n)
    ])
    
    # Format history
    history_text = "None"
    if config.enable_history and history:
        history_text = format_history_block(
            history, 
            n, 
            max_lines=config.history_lines,
            compact=True
        )
    
    # Format occupancy power
    occ_kw_text = f"{format_number(abs(env.occ_sum_kw))} KW  (internal heat gain)"
    
    # Build reasoning instruction
    reasoning_instruction = ""
    if config.enable_reasoning:
        reasoning_instruction = dedent("""
            IMPORTANT: Give 1–2 sentences of reasoning (no '[' or ']'). Then END with exactly one final line:
            Actions: [x1, x2, ..., x{n}]
            This must be the last line; 'Actions:' appears only once; after the final ']' output nothing else.
        """).format(n=n).strip()
    else:
        reasoning_instruction = dedent("""
            IMPORTANT: Respond with exactly one single JSON array of floats:
            [x1, x2, ..., x{n}]
            Do not include any other text.
        """).format(n=n).strip()
    
    # Construct full prompt
    prompt = dedent(f"""\
    You are the HVAC administrator responsible for managing a building of type {building_pretty} located in {location}, where the climate is {climate_pretty}.
    The building has {n} rooms in total.
    
    Currently, temperature in each room is as follows:
{temp_lines}
    
    The external climate conditions are as follows:
       Outside Temperature: {format_number(env.outside_temp)} degrees Celsius
       Global Horizontal Irradiance: {format_number(env.ghi_avg)} W/m²
       Ground Temperature: {format_number(env.ground_temp)} degrees Celsius
       Occupant Power: {occ_kw_text}
       Target Temperature: {format_number(target)} degrees Celsius
    
    To optimize HVAC control, follow these guidelines:
    1. Output one list of length {n} with each value ranging from -1 to 1.
       - Absolute value represents HVAC power
       - Positive number means heating (raise temperature)
       - Negative number means cooling (lower temperature)
    
    2. The order must match the room order above.
    
    3. Match the sign to (Target − Room Temperature). Avoid identical actions for all rooms unless all room temperatures are identical.
    
    4. Since all actions are within the range [-1, 1], avoid making large changes compared to the most recent action history on this scale (e.g., a change of 0.5 is already significant), unless there is a notable change in room temperatures.
    
    History Action And Feedback Reference:
    {history_text}
    
    {reasoning_instruction}
    """).strip()
    
    logger.debug(f"Generated prompt: {len(prompt)} chars, {n} zones")
    
    return prompt


def build_simple_prompt(
    obs: List[float],
    target: float = 22.0,
    n_rooms: Optional[int] = None
) -> str:
    """
    Build simplified prompt without history or detailed context.
    Useful for quick testing or baseline comparisons.
    
    Args:
        obs: Current observation vector
        target: Target temperature
        n_rooms: Override number of rooms
        
    Returns:
        Simple prompt string
    """
    n = n_rooms if n_rooms is not None else zone_count_from_obs(obs)
    temps = obs[:n]
    env = extract_env_terms(obs, n_override=n)
    
    temp_list = ", ".join([f"R{i+1}={format_number(t)}" for i, t in enumerate(temps)])
    
    return dedent(f"""\
    HVAC Control Task:
    - {n} rooms with temperatures: {temp_list}
    - Target: {format_number(target)}°C
    - Outside: {format_number(env.outside_temp)}°C
    
    Output action array of length {n}, values in [-1, 1]:
    Actions: [x1, ..., x{n}]
    """).strip()


# ============================================================================
# Validation and Testing
# ============================================================================

def validate_observation(obs: List[float]) -> Tuple[bool, str]:
    """
    Validate observation vector structure.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not obs:
        return False, "Empty observation"
    
    if len(obs) < 3:
        return False, f"Observation too short: {len(obs)} < 3"
    
    # Check if length matches 3n+2 structure
    n = (len(obs) - 2) // 3
    expected_len = 3 * n + 2
    
    if len(obs) != expected_len:
        return False, f"Invalid length {len(obs)}, expected 3n+2 structure"
    
    # Check for NaN or inf values
    for i, val in enumerate(obs):
        try:
            f = float(val)
            if not (-1000 <= f <= 1000):  # Reasonable range check
                return False, f"Value at index {i} out of reasonable range: {f}"
        except (ValueError, TypeError):
            return False, f"Non-numeric value at index {i}: {val}"
    
    return True, "Valid"


if __name__ == "__main__":
    # Test with sample observation
    sample_obs = [22.5, 23.0, 21.8, 30.0, 500, 450, 480, 20.0, 0.5, 0.3, 0.4]
    
    print("Testing prompt builder:")
    print(f"Observation length: {len(sample_obs)}")
    print(f"Zones: {zone_count_from_obs(sample_obs)}")
    
    valid, msg = validate_observation(sample_obs)
    print(f"Validation: {valid} - {msg}")
    
    if valid:
        prompt = build_prompt(
            obs=sample_obs,
            building="OfficeSmall",
            location="Tucson",
            climate="Hot_Dry",
            target=22.0,
            round_idx=1,
            history=[]
        )
        print(f"\nGenerated prompt ({len(prompt)} chars):\n")
        print(prompt[:500] + "...")
