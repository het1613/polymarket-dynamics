"""
Empirical target computation and calibration utilities for the ABM.

Calibration is intentionally light-weight:

- panel-level simulated method of moments / distance matching
- random search, optionally centred around an anchor parameter vector
- emphasis on robust tails, volatility clustering, deadline pressure,
  and stage-2 interaction patterns
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.stattools import acf as _acf

from src.abm.model import (
    ABMParams,
    panel_template_from_reference,
    simulate_panel,
)


def _safe_acf(series: pd.Series, lag: int) -> float | None:
    clean = series.dropna()
    if len(clean) <= lag + 5:
        return None
    return float(_acf(clean.values, nlags=lag, fft=True)[lag])


def compute_quantile_regression_targets(
    df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95),
    sample_size: int = 20_000,
) -> dict[str, float]:
    """
    Quantile-regression targets used as stage-2 diagnostics.

    These are kept separate from ``compute_targets`` so we can inspect them
    without forcing every calibration loop to fit four quantile regressions.
    """

    cols = ["abs_logit_return", "trade_rate_1h", "time_to_resolution"]
    sub = df[cols].dropna()
    if len(sub) < 500:
        return {}

    if len(sub) > sample_size:
        sub = sub.sample(sample_size, random_state=0)

    X = sm.add_constant(sub[["trade_rate_1h", "time_to_resolution"]])
    y = sub["abs_logit_return"]

    out: dict[str, float] = {}
    for q in quantiles:
        res = QuantReg(y, X).fit(q=q)
        tag = str(q).replace(".", "p")
        out[f"qreg_trade_{tag}"] = float(res.params["trade_rate_1h"])
        out[f"qreg_ttr_{tag}"] = float(res.params["time_to_resolution"])
    return out


def compute_targets(
    df: pd.DataFrame,
    include_interaction: bool = False,
) -> dict[str, float]:
    """Compute calibration targets from empirical or simulated feature panels."""

    lr = df["logit_return"].dropna()
    alr = df["abs_logit_return"].dropna()
    ttr = df["time_to_resolution"]

    t: dict[str, float] = {}

    if len(alr) > 30:
        t["q90_abs_r"] = float(alr.quantile(0.90))
        t["q95_abs_r"] = float(alr.quantile(0.95))
        t["q99_abs_r"] = float(alr.quantile(0.99))

    for lag in (1, 3, 5, 10, 24):
        val = _safe_acf(alr, lag)
        if val is not None:
            t[f"acf{lag}_abs_r"] = val

    early_mask = ttr > 30
    late_mask = ttr <= 30
    early_alr = df.loc[early_mask, "abs_logit_return"].dropna()
    late_alr = df.loc[late_mask, "abs_logit_return"].dropna()
    early_lr = df.loc[early_mask, "logit_return"].dropna()
    late_lr = df.loc[late_mask, "logit_return"].dropna()

    if len(early_alr) > 20:
        t["mean_abs_r_early"] = float(early_alr.mean())
        t["std_lr_early"] = float(early_lr.std())
        val = _safe_acf(early_alr, 1)
        if val is not None:
            t["acf1_early"] = val

    if len(late_alr) > 20:
        t["mean_abs_r_late"] = float(late_alr.mean())
        t["std_lr_late"] = float(late_lr.std())
        val = _safe_acf(late_alr, 1)
        if val is not None:
            t["acf1_late"] = val

    if "mean_abs_r_early" in t and "mean_abs_r_late" in t and t["mean_abs_r_early"] > 0:
        t["late_early_mean_ratio"] = t["mean_abs_r_late"] / t["mean_abs_r_early"]
    if "std_lr_early" in t and "std_lr_late" in t and t["std_lr_early"] > 0:
        t["late_early_std_ratio"] = t["std_lr_late"] / t["std_lr_early"]

    if include_interaction and "trade_rate_1h" in df.columns:
        sub = df[["abs_logit_return", "trade_rate_1h", "time_to_resolution"]].dropna()
        if len(sub) > 100:
            sub = sub.copy()
            sub["phase"] = np.where(sub["time_to_resolution"] <= 30, "late", "early")
            try:
                sub["act_bin"] = pd.qcut(
                    sub["trade_rate_1h"],
                    3,
                    labels=["low", "mid", "high"],
                    duplicates="drop",
                )
            except ValueError:
                return t

            cell_vals: dict[str, float] = {}
            for phase in ("early", "late"):
                for ab in ("low", "mid", "high"):
                    mask = (sub["phase"] == phase) & (sub["act_bin"] == ab)
                    vals = sub.loc[mask, "abs_logit_return"]
                    key = f"interact_{phase}_{ab}"
                    if len(vals) > 5:
                        cell_vals[key] = float(vals.mean())
                        t[key] = cell_vals[key]

            if "interact_early_high" in cell_vals and "interact_early_low" in cell_vals:
                t["interact_slope_early"] = cell_vals["interact_early_high"] - cell_vals["interact_early_low"]
            if "interact_late_high" in cell_vals and "interact_late_low" in cell_vals:
                t["interact_slope_late"] = cell_vals["interact_late_high"] - cell_vals["interact_late_low"]
            if "interact_late_low" in cell_vals and "interact_early_low" in cell_vals:
                t["deadline_low"] = cell_vals["interact_late_low"] - cell_vals["interact_early_low"]
            if "interact_late_mid" in cell_vals and "interact_early_mid" in cell_vals:
                t["deadline_mid"] = cell_vals["interact_late_mid"] - cell_vals["interact_early_mid"]
            if "interact_late_high" in cell_vals and "interact_early_high" in cell_vals:
                t["deadline_high"] = cell_vals["interact_late_high"] - cell_vals["interact_early_high"]

    return t


STAGE1_WEIGHTS: dict[str, float] = {
    "q90_abs_r": 0.8,
    "q95_abs_r": 2.2,
    "q99_abs_r": 2.5,
    "acf1_abs_r": 1.8,
    "acf3_abs_r": 1.2,
    "acf5_abs_r": 1.0,
    "acf10_abs_r": 0.7,
    "acf24_abs_r": 0.5,
    "mean_abs_r_early": 2.0,
    "mean_abs_r_late": 2.3,
    "std_lr_early": 1.6,
    "std_lr_late": 1.8,
    "acf1_early": 0.9,
    "acf1_late": 1.1,
    "late_early_mean_ratio": 2.3,
    "late_early_std_ratio": 1.4,
}

STAGE2_WEIGHTS: dict[str, float] = {
    **STAGE1_WEIGHTS,
    "interact_early_low": 0.4,
    "interact_early_mid": 0.4,
    "interact_early_high": 0.4,
    "interact_late_low": 0.5,
    "interact_late_mid": 0.6,
    "interact_late_high": 0.6,
    "interact_slope_early": 2.0,
    "interact_slope_late": 2.6,
    "deadline_low": 0.8,
    "deadline_mid": 1.2,
    "deadline_high": 1.2,
}


def compute_distance(
    sim_targets: dict[str, float],
    emp_targets: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted mean squared relative error across available targets."""
    if weights is None:
        weights = STAGE1_WEIGHTS

    total = 0.0
    w_sum = 0.0
    for key, w in weights.items():
        e = emp_targets.get(key)
        s = sim_targets.get(key)
        if e is None or s is None or np.isnan(e) or np.isnan(s):
            continue
        denom = max(abs(e), 1e-6)
        total += w * ((s - e) / denom) ** 2
        w_sum += w
    return total / max(w_sum, 1e-12)


PARAM_RANGES: dict[str, tuple[float, float]] = {
    "sigma_star": (0.002, 0.020),
    "jump_prob": (0.0005, 0.020),
    "jump_size_std": (0.04, 0.45),
    "jump_decay": (0.55, 0.97),
    "n_informed": (8, 30),
    "n_noise": (12, 45),
    "n_herding": (5, 24),
    "p_informed": (0.05, 0.30),
    "p_noise": (0.02, 0.20),
    "p_herding": (0.02, 0.18),
    "deadline_ramp": (0.30, 4.80),
    "informed_sensitivity": (0.60, 3.00),
    "noise_scale": (0.60, 3.80),
    "herding_lookback": (2, 18),
    "herding_strength": (0.20, 2.50),
    "max_position": (6.0, 22.0),
    "max_order": (0.60, 4.00),
    "lambda_impact": (0.0008, 0.020),
    "beta_impact": (0.80, 1.80),
    "no_move_threshold": (0.10, 4.00),
    "market_heterogeneity": (0.15, 1.10),
}

INT_PARAMS = {"n_informed", "n_noise", "n_herding", "herding_lookback"}


def _sample_uniform_params(rng: np.random.Generator, stage: int) -> ABMParams:
    sampled: dict[str, float | int] = {}
    for key, (lo, hi) in PARAM_RANGES.items():
        if key in INT_PARAMS:
            sampled[key] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            sampled[key] = float(rng.uniform(lo, hi))

    params = ABMParams(**sampled)
    if stage >= 2:
        params.participation_excite = float(rng.uniform(0.15, 1.80))
        params.depth_activity_scale = float(rng.uniform(0.20, 2.80))
    return params


def _sample_local_params(
    rng: np.random.Generator,
    anchor: ABMParams,
    stage: int,
    perturb_scale: float,
) -> ABMParams:
    sampled: dict[str, float | int] = {}
    anchor_dict = anchor.to_dict()

    for key, (lo, hi) in PARAM_RANGES.items():
        base = anchor_dict[key]
        span = hi - lo
        if key in INT_PARAMS:
            half_width = max(1, int(round(span * perturb_scale)))
            low_i = max(int(lo), int(base) - half_width)
            high_i = min(int(hi), int(base) + half_width)
            sampled[key] = int(rng.integers(low_i, high_i + 1))
        else:
            low_f = max(lo, float(base) - span * perturb_scale)
            high_f = min(hi, float(base) + span * perturb_scale)
            sampled[key] = float(rng.uniform(low_f, high_f))

    params = ABMParams(**sampled)

    if stage >= 2:
        if anchor.participation_excite > 0:
            lo = max(0.05, anchor.participation_excite - 1.5 * perturb_scale)
            hi = anchor.participation_excite + 1.5 * perturb_scale
            params.participation_excite = float(rng.uniform(lo, hi))
        else:
            params.participation_excite = float(rng.uniform(0.15, 1.80))

        if anchor.depth_activity_scale > 0:
            lo = max(0.05, anchor.depth_activity_scale - 2.0 * perturb_scale)
            hi = anchor.depth_activity_scale + 2.0 * perturb_scale
            params.depth_activity_scale = float(rng.uniform(lo, hi))
        else:
            params.depth_activity_scale = float(rng.uniform(0.20, 2.80))

    return params


def evaluate_params(
    params: ABMParams,
    emp_targets: dict[str, float],
    *,
    stage: int = 1,
    n_seeds: int = 2,
    n_markets: int = 10,
    market_specs: list[dict] | None = None,
    base_seed: int = 0,
) -> tuple[float, list[dict[str, float]], pd.DataFrame]:
    """Evaluate one parameter vector over one or more simulation seeds."""

    include_interaction = stage >= 2
    weights = STAGE2_WEIGHTS if include_interaction else STAGE1_WEIGHTS

    panels: list[pd.DataFrame] = []
    dists: list[float] = []
    targets_list: list[dict[str, float]] = []

    for s in range(n_seeds):
        panel = simulate_panel(
            params,
            n_markets=n_markets,
            stage=stage,
            base_seed=base_seed + s * 1_003,
            market_specs=market_specs,
        )
        panel_targets = compute_targets(panel, include_interaction=include_interaction)
        dists.append(compute_distance(panel_targets, emp_targets, weights))
        targets_list.append(panel_targets)
        panels.append(panel)

    mean_panel = pd.concat(panels, ignore_index=True)
    return float(np.mean(dists)), targets_list, mean_panel


def random_search(
    emp_targets: dict[str, float],
    *,
    n_iter: int = 200,
    n_seeds: int = 2,
    stage: int = 1,
    T_hours: int | None = None,
    n_markets: int = 10,
    market_specs: list[dict] | None = None,
    anchor_params: ABMParams | None = None,
    perturb_scale: float = 0.20,
    verbose: bool = True,
) -> tuple[ABMParams, float, list[dict]]:
    """
    Random-search calibration over panel simulations.

    If ``anchor_params`` is supplied, search is local around that vector.
    """

    rng = np.random.default_rng(42 + stage)
    include_interaction = stage >= 2
    weights = STAGE2_WEIGHTS if include_interaction else STAGE1_WEIGHTS

    best_dist = np.inf
    best_params: ABMParams | None = None
    results: list[dict] = []

    for i in range(n_iter):
        if anchor_params is None:
            params = _sample_uniform_params(rng, stage=stage)
        else:
            params = _sample_local_params(
                rng,
                anchor=anchor_params,
                stage=stage,
                perturb_scale=perturb_scale,
            )

        if T_hours is not None and market_specs is None:
            params.T_hours = T_hours

        dists: list[float] = []
        for s in range(n_seeds):
            try:
                panel = simulate_panel(
                    params,
                    n_markets=n_markets,
                    stage=stage,
                    base_seed=7 + 1_003 * s + i * 17,
                    market_specs=market_specs,
                )
                targets = compute_targets(panel, include_interaction=include_interaction)
                dists.append(compute_distance(targets, emp_targets, weights))
            except Exception:
                dists.append(1e6)

        mean_dist = float(np.mean(dists))
        results.append({"params": params, "distance": mean_dist, "iter": i})

        if mean_dist < best_dist:
            best_dist = mean_dist
            best_params = params
            if verbose:
                print(f"  iter {i:4d}  new best distance = {best_dist:.6f}")

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{n_iter}] best so far = {best_dist:.6f}")

    assert best_params is not None
    return best_params, best_dist, results


__all__ = [
    "ABMParams",
    "STAGE1_WEIGHTS",
    "STAGE2_WEIGHTS",
    "compute_distance",
    "compute_quantile_regression_targets",
    "compute_targets",
    "evaluate_params",
    "panel_template_from_reference",
    "random_search",
]
