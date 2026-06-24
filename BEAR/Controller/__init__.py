"""BEAR Controller module"""
try:
    from BEAR.Controller.MPC_Controller import MPCAgent  # requires cvxpy (optional)
except ImportError:
    MPCAgent = None

__all__ = ["MPCAgent"]
