"""
Agent-based model package for prediction market dynamics.

Stages:
  1 — Minimal ABM: informed + noise + herding, static participation + deadline ramp
  2 — Endogenous participation + state-dependent depth
  3 — (Optional) Shock-type separation for reversal asymmetry
"""

from src.abm.model import ABMParams, simulate_market, sim_to_features, simulate_panel  # noqa: F401
from src.abm.calibration import compute_targets, compute_distance, random_search  # noqa: F401
