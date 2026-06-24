"""
BEAR - Building Environment for AI Research

A physics-principled building simulator for HVAC control research.
"""

__version__ = "1.0.0"
__author__ = "BEAR Team"

from BEAR.Env.env_building import BuildingEnvReal
from BEAR.Utils.utils_building import ParameterGenerator, get_user_input

# The MPC controller depends on cvxpy, which is heavy and unused by this project's
# pipeline. Import it lazily so a missing cvxpy never breaks core env usage.
try:
    from BEAR.Controller.MPC_Controller import MPCAgent
except ImportError:
    MPCAgent = None

__all__ = [
    "BuildingEnvReal",
    "ParameterGenerator",
    "get_user_input",
    "MPCAgent",
]
