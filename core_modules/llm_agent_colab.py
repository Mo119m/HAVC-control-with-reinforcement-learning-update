"""
LLM Agent for HVAC Control - Optimized Version

This module provides LLM inference and action parsing for HVAC control tasks.
Supports reasoning-enabled generation with robust action parsing.

Key Features:
- Efficient model loading with caching
- Multiple parsing strategies with fallback
- Configurable generation parameters
- Comprehensive error handling
"""

import os
import re
import json
import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForCausalLM] = None


@dataclass
class GenerationConfig:
    """Configuration for LLM generation"""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    def validate(self) -> None:
        """Validate generation parameters"""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be in [0.0, 2.0]")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be in [0.0, 1.0]")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


def _select_device_and_dtype() -> Tuple[Any, torch.dtype]:
    """
    Select optimal device map and dtype based on available hardware.
    
    Returns:
        Tuple of (device_map, torch_dtype)
    """
    if torch.cuda.is_available():
        try:
            # Prefer bfloat16 for better numerical stability
            return "auto", torch.bfloat16
        except Exception as e:
            logger.warning(f"bfloat16 not available, falling back to float16: {e}")
            return "auto", torch.float16
    
    logger.info("CUDA not available, using CPU with float32")
    return {"": "cpu"}, torch.float32


def load_llm(
    model_name: str, 
    hf_token: Optional[str] = None,
    force_reload: bool = False
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load language model and tokenizer with caching.
    
    Args:
        model_name: HuggingFace model identifier
        hf_token: Optional HuggingFace API token
        force_reload: Force reload even if cached
        
    Returns:
        Tuple of (tokenizer, model)
    """
    global _TOKENIZER, _MODEL
    
    # Return cached if available and not forcing reload
    if not force_reload and _MODEL is not None and _TOKENIZER is not None:
        logger.info("Using cached model")
        return _TOKENIZER, _MODEL
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Model has no EOS token to use as pad token")
        
        # Load model with optimal settings
        device_map, torch_dtype = _select_device_and_dtype()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
            device_map=device_map,
            torch_dtype=torch_dtype
        ).eval()
        
        logger.info("Model loaded successfully")
        
        # Cache for future use
        _TOKENIZER = tokenizer
        _MODEL = model
        
        return tokenizer, model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def call_llm(
    prompt: str,
    n_actions: int,
    model_name: Optional[str] = None,
    config: Optional[GenerationConfig] = None,
    **kwargs
) -> str:
    """
    Generate text from LLM with reasoning capability.
    
    Args:
        prompt: Input prompt for the model
        n_actions: Expected number of actions in output
        model_name: Model name (uses default if None)
        config: Generation configuration
        **kwargs: Override config parameters
        
    Returns:
        Generated text string
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If generation fails
    """
    # Validate inputs
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    
    # Initialize config
    if config is None:
        config = GenerationConfig()
    
    # Override with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    config.validate()
    
    # Load model
    model_name = model_name or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    tokenizer, model = load_llm(model_name, os.getenv("HF_TOKEN"))
    
    device = next(model.parameters()).device
    
    # Prepare messages (user only, no system prompt)
    messages = [{"role": "user", "content": prompt.strip()}]
    
    try:
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        )
        
        logger.debug(f"Input tokens: {inputs.shape[1]}")
        
        # Truncate if needed
        model_max = getattr(model.config, "max_position_embeddings", 4096)
        safe_cap = min(model_max, 8192)
        
        if inputs.shape[1] > safe_cap:
            logger.warning(f"Truncating input from {inputs.shape[1]} to {safe_cap} tokens")
            inputs = inputs[:, -safe_cap:]
        
        # Prepare generation kwargs
        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": [tokenizer.eos_token_id],
            "use_cache": True,
            "repetition_penalty": config.repetition_penalty,
        }
        
        # Add sampling parameters
        if config.temperature > 0.0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            })
        else:
            gen_kwargs["do_sample"] = False
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(inputs.to(device), **gen_kwargs)
        
        # Decode only generated portion
        generated = outputs[0, inputs.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        logger.debug(f"Generated text tail (200 chars): {text[-200:]}")
        
        return text
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise RuntimeError(f"LLM generation failed: {e}") from e


# ============================================================================
# Action Parsing with Multiple Strategies
# ============================================================================

# Regex patterns for parsing (in priority order)
_ACT_LINE_RE = re.compile(r'Actions\s*:\s*(\[[^\[\]]*\])', re.IGNORECASE)
_ACT_LINE_NOBRACKETS_RE = re.compile(r'Actions\s*:\s*([\-0-9\.,\s]+)$', re.IGNORECASE | re.MULTILINE)
_ANY_BRACKET_NUMS_RE = re.compile(r'\[\s*([^\[\]]*?)\s*\]')


def _clean_text(text: str) -> str:
    """Remove markdown code blocks and special characters"""
    text = re.sub(r"```[\w\-]*", "", text)
    text = text.replace("```", "")
    text = text.replace("\u200b", "").replace("\u00a0", " ")
    return text.strip()


def _parse_number_list(text: str) -> Optional[List[float]]:
    """Parse comma-separated numbers into float list"""
    try:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return [float(x) for x in parts]
    except (ValueError, TypeError):
        return None


def parse_actions(
    raw_text: str, 
    n: int,
    strict: bool = False
) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    """
    Parse action array from LLM output with multiple fallback strategies.
    
    Parsing priority:
    1. "Actions: [x1, x2, ...]" format (with brackets)
    2. "Actions: x1, x2, ..." format (no brackets)
    3. Last line as JSON array
    4. Any bracketed number list matching length n
    
    Args:
        raw_text: Raw text from LLM
        n: Expected number of actions
        strict: If True, only accept "Actions:" format
        
    Returns:
        Tuple of (action_list, metadata_dict)
        - action_list: List of n floats, or None if parsing failed
        - metadata: Dictionary with parsing information
    """
    if not raw_text or not raw_text.strip():
        return None, {"parsed_from": "empty_input", "error": "Empty input"}
    
    if n <= 0:
        return None, {"parsed_from": "invalid_n", "error": f"Invalid n: {n}"}
    
    # Clean text
    text = _clean_text(raw_text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    meta = {"parsed_from": "failed", "raw_length": len(raw_text)}
    
    # Strategy 1: Actions line with brackets
    for line in reversed(lines):
        match = _ACT_LINE_RE.search(line)
        if match:
            try:
                arr = json.loads(match.group(1))
                if isinstance(arr, list) and len(arr) == n:
                    actions = [float(x) for x in arr]
                    meta["parsed_from"] = "actions_line"
                    return actions, meta
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(f"Failed to parse Actions line: {e}")
    
    # Strategy 2: Actions line without brackets
    for line in reversed(lines):
        match = _ACT_LINE_NOBRACKETS_RE.search(line)
        if match:
            arr = _parse_number_list(match.group(1))
            if arr and len(arr) == n:
                meta["parsed_from"] = "actions_line_no_brackets"
                return arr, meta
    
    # If strict mode, stop here
    if strict:
        return None, {**meta, "error": "No Actions: format found in strict mode"}
    
    # Strategy 3: Last line as JSON
    if lines:
        last_line = lines[-1]
        
        # Try to fix missing closing bracket
        if last_line.count("[") >= 1 and last_line.count("]") == 0:
            last_line = last_line + "]"
        
        try:
            data = json.loads(last_line)
            if isinstance(data, list) and len(data) == n:
                actions = [float(x) for x in data]
                meta["parsed_from"] = "last_json"
                return actions, meta
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"Failed to parse last line as JSON: {e}")
    
    # Strategy 4: Any bracketed numbers
    last_good = None
    for match in _ANY_BRACKET_NUMS_RE.finditer(text):
        arr = _parse_number_list(match.group(1))
        if arr and len(arr) == n:
            last_good = arr
    
    if last_good is not None:
        meta["parsed_from"] = "any_brackets"
        return last_good, meta
    
    # All strategies failed
    return None, {**meta, "error": f"No valid array of length {n} found"}


def parse_actions_with_validation(
    raw_text: str,
    n: int,
    min_val: float = -1.0,
    max_val: float = 1.0,
    strict: bool = False
) -> Tuple[Optional[List[float]], Dict[str, Any]]:
    """
    Parse and validate actions are within expected range.
    
    Args:
        raw_text: Raw text from LLM
        n: Expected number of actions
        min_val: Minimum allowed action value
        max_val: Maximum allowed action value
        strict: If True, only accept "Actions:" format
        
    Returns:
        Tuple of (action_list, metadata)
    """
    actions, meta = parse_actions(raw_text, n, strict=strict)
    
    if actions is None:
        return None, meta
    
    # Validate range
    out_of_range = [i for i, a in enumerate(actions) if not (min_val <= a <= max_val)]
    
    if out_of_range:
        meta["warning"] = f"Actions at indices {out_of_range} out of range [{min_val}, {max_val}]"
        meta["out_of_range_indices"] = out_of_range
        
        # Clip to range
        actions = [max(min_val, min(max_val, a)) for a in actions]
        meta["clipped"] = True
    
    return actions, meta


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test parsing
    test_cases = [
        ("[0.1, 0.2, 0.3]", 3),
        ("Actions: [0.5, -0.3, 0.8]", 3),
        ("The temperature is high. Actions: 0.2, -0.1, 0.4", 3),
        ("Let me think... \n[0.1, 0.2]", 2),
    ]
    
    print("Testing action parsing:")
    for text, n in test_cases:
        actions, meta = parse_actions(text, n)
        print(f"\nInput: {text[:50]}...")
        print(f"Actions: {actions}")
        print(f"Meta: {meta}")
