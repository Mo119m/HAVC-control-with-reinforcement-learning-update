"""
BEAR - Building Environment for AI Research

A physics-principled building simulator for HVAC control research.
"""

__version__ = "1.0.0"
__author__ = "BEAR Team"

from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator, get_user_input
from BEAR.Controller.MPC_Controller import MPCAgent

__all__ = [
    "BuildingEnvReal",
    "ParameterGenerator",
    "get_user_input",
    "MPCAgent",
]
