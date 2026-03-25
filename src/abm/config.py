"""Configuration and dataclasses for the RQ4 ABM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from src.config import DATA_PROCESSED, PROJECT_ROOT

ABM_OUTPUT_DIR = DATA_PROCESSED / "abm"
ABM_PLOT_PREFIX = "rq4_abm"
ABM_EMPIRICAL_DIR = ABM_OUTPUT_DIR / "empirical"
PLOTS_DIR = PROJECT_ROOT / "plots"

EARLY_DEADLINE_DAYS = 30.0
ACTIVITY_LABELS = ("Low", "Mid", "High")
QUANTILES = (0.50, 0.75, 0.90, 0.95)
QUANTILES_CALIBRATION = (0.50, 0.90, 0.95)
REVERSAL_HORIZONS = (1, 6, 12, 24)
ACF_LAGS = 24

REPRESENTATIVE_PLOT_MARKETS = (
    "will-donald-trump-win-the-2024-us-presidential-election",
    "will-kamala-harris-win-the-popular-vote-in-the-2024-presidential-election",
    "will-joe-biden-win-the-2024-democratic-presidential-nomination",
    "will-a-republican-win-texas-in-the-2024-us-presidential-election",
)


@dataclass(frozen=True)
class MarketSkeleton:
    """Empirical market scaffold used to simulate one synthetic market."""

    market_slug: str
    market_type: str
    timestamps: tuple
    time_to_resolution: tuple[float, ...]
    initial_price: float
    initial_logit: float

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)

    @property
    def max_time_to_resolution(self) -> float:
        return float(max(self.time_to_resolution))


@dataclass(frozen=True)
class SimulationParams:
    """Compact stage-wise ABM parameter vector."""

    stage: str = "stage1"

    # Agent cohorts
    n_informed: int = 12
    n_noise: int = 24
    n_herding: int = 18
    initial_cash: float = 18.0
    max_position: float = 7.0
    max_order: float = 1.2
    trade_weight_informed: int = 10
    trade_weight_noise: int = 14
    trade_weight_herding: int = 14
    volume_scale: float = 12_000.0

    # Latent anchor and trader behaviour
    baseline_participation: float = 0.13
    deadline_ramp: float = 1.0
    anchor_phi: float = 0.88
    anchor_sigma: float = 0.03
    jump_prob: float = 0.007
    jump_scale: float = 0.22
    informed_strength: float = 1.1
    noise_scale: float = 0.45
    herding_strength: float = 0.95
    trend_memory: float = 0.94
    mispricing_threshold: float = 0.02

    # Impact / depth
    base_depth: float = 6.5
    impact_lambda: float = 0.28
    impact_beta: float = 0.92
    no_move_threshold: float = 0.85

    # Stage-2 extensions
    volatility_reference: float = 0.03
    activity_reference: float = 8.0
    vol_participation: float = 0.0
    activity_participation: float = 0.0
    depth_activity_coef: float = 0.0
    depth_deadline_coef: float = 0.0
    depth_interaction_coef: float = 0.0

    # Stage-3 extension (optional, thin-market transient dislocation)
    dislocation_prob: float = 0.0
    dislocation_scale: float = 0.0
    dislocation_decay: float = 0.0
    dislocation_activity_threshold: float = 6.0

    def with_updates(self, **updates: float | int | str) -> "SimulationParams":
        return replace(self, **updates)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationConfig:
    """Simple stage-wise calibration configuration."""

    stage1_draws: int = 16
    stage2_draws: int = 12
    stage3_draws: int = 8
    calibration_seeds: tuple[int, ...] = (101, 202)
    final_seeds: tuple[int, ...] = (401, 402, 403)
    calibration_sample_n: int = 15_000
    random_seed: int = 20260325


STAGE1_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "baseline_participation": (0.04, 0.18),
    "deadline_ramp": (0.50, 2.80),
    "anchor_phi": (0.60, 0.98),
    "anchor_sigma": (0.003, 0.045),
    "jump_prob": (0.0008, 0.020),
    "jump_scale": (0.08, 0.90),
    "informed_strength": (0.40, 2.10),
    "noise_scale": (0.08, 0.70),
    "herding_strength": (0.20, 1.60),
    "trend_memory": (0.80, 0.985),
    "base_depth": (1.5, 10.0),
    "impact_lambda": (0.08, 0.60),
    "no_move_threshold": (0.10, 1.40),
}

STAGE2_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "vol_participation": (0.05, 0.15),
    "activity_participation": (0.08, 0.22),
    "depth_activity_coef": (0.70, 1.60),
    "depth_deadline_coef": (0.20, 0.45),
    "depth_interaction_coef": (0.08, 0.15),
}

STAGE3_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "dislocation_prob": (0.05, 0.70),
    "dislocation_scale": (0.03, 0.30),
    "dislocation_decay": (0.35, 0.92),
    "dislocation_activity_threshold": (2.0, 16.0),
}


def ensure_output_dirs() -> None:
    ABM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ABM_EMPIRICAL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def logit_clip_bounds() -> tuple[float, float]:
    lo = float(np.log(0.001 / (1.0 - 0.001)))
    hi = float(np.log(0.999 / (1.0 - 0.999)))
    return lo, hi
