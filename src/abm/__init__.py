"""
Agent-based model package for prediction market dynamics.

Stages:
  1 — Minimal ABM: informed + noise + herding, static participation + deadline ramp
  2 — Endogenous participation + state-dependent depth
  3 — (Optional) Shock-type separation for reversal asymmetry
"""

from src.abm.model import (  # noqa: F401
    ABMParams,
    panel_template_from_reference,
    sim_to_features,
    simulate_market,
    simulate_panel,
)
from src.abm.calibration import (  # noqa: F401
    compute_distance,
    compute_quantile_regression_targets,
    compute_targets,
    evaluate_params,
    random_search,
)
