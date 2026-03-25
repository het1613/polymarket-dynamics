"""
Empirical target computation and calibration via simulated method of moments.

Targets are computed identically from empirical *and* simulated feature
DataFrames, then compared with a weighted normalised distance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf as _acf

from src.abm.model import ABMParams, simulate_market, sim_to_features


# ── target computation ───────────────────────────────────────────────────────

def compute_targets(
    df: pd.DataFrame,
    include_interaction: bool = False,
) -> dict[str, float]:
    """Compute summary statistics used as calibration targets.

    Works on both the empirical ``features_politics.parquet`` and
    simulated feature panels.
    """
    lr = df["logit_return"].dropna()
    alr = df["abs_logit_return"].dropna()
    ttr = df["time_to_resolution"]

    t: dict[str, float] = {}

    # ── tail quantiles ───────────────────────────────────────────────
    if len(alr) > 30:
        t["q90_abs_r"] = float(alr.quantile(0.90))
        t["q95_abs_r"] = float(alr.quantile(0.95))
        t["q99_abs_r"] = float(alr.quantile(0.99))

    # ── ACF of |r| ──────────────────────────────────────────────────
    if len(alr) >= 50:
        acf_vals = _acf(alr.values, nlags=24, fft=True)
        t["acf1_abs_r"] = float(acf_vals[1])
        t["acf3_abs_r"] = float(acf_vals[3])
        t["acf5_abs_r"] = float(acf_vals[5])
        t["acf10_abs_r"] = float(acf_vals[10])
        t["acf24_abs_r"] = float(acf_vals[24])

    # ── early vs late split (30-day threshold) ───────────────────────
    early_alr = df.loc[ttr > 30, "abs_logit_return"].dropna()
    late_alr = df.loc[ttr <= 30, "abs_logit_return"].dropna()
    early_lr = df.loc[ttr > 30, "logit_return"].dropna()
    late_lr = df.loc[ttr <= 30, "logit_return"].dropna()

    if len(early_alr) > 20:
        t["mean_abs_r_early"] = float(early_alr.mean())
        t["std_lr_early"] = float(early_lr.std())
        if len(early_alr) >= 30:
            t["acf1_early"] = float(_acf(early_alr.values, nlags=1, fft=True)[1])
    if len(late_alr) > 20:
        t["mean_abs_r_late"] = float(late_alr.mean())
        t["std_lr_late"] = float(late_lr.std())
        if len(late_alr) >= 30:
            t["acf1_late"] = float(_acf(late_alr.values, nlags=1, fft=True)[1])

    # ── activity × deadline interaction (stage 2) ────────────────────
    if include_interaction and "trade_rate_1h" in df.columns:
        sub = df[["abs_logit_return", "trade_rate_1h", "time_to_resolution"]].dropna()
        if len(sub) > 100:
            sub = sub.copy()
            sub["phase"] = np.where(sub["time_to_resolution"] <= 30, "late", "early")
            try:
                sub["act_bin"] = pd.qcut(
                    sub["trade_rate_1h"], 3,
                    labels=["low", "mid", "high"],
                    duplicates="drop",
                )
            except ValueError:
                return t
            for phase in ("early", "late"):
                for ab in ("low", "mid", "high"):
                    mask = (sub["phase"] == phase) & (sub["act_bin"] == ab)
                    vals = sub.loc[mask, "abs_logit_return"]
                    t[f"interact_{phase}_{ab}"] = float(vals.mean()) if len(vals) > 5 else np.nan

    return t


# ── distance metric ──────────────────────────────────────────────────────────

STAGE1_WEIGHTS: dict[str, float] = {
    "q90_abs_r": 1.0,
    "q95_abs_r": 1.5,
    "q99_abs_r": 0.5,
    "acf1_abs_r": 2.0,
    "acf3_abs_r": 1.0,
    "acf5_abs_r": 0.8,
    "acf10_abs_r": 0.5,
    "acf24_abs_r": 0.3,
    "mean_abs_r_early": 2.0,
    "mean_abs_r_late": 2.0,
    "std_lr_early": 1.0,
    "std_lr_late": 1.0,
    "acf1_early": 0.8,
    "acf1_late": 0.8,
}

STAGE2_WEIGHTS: dict[str, float] = {
    **STAGE1_WEIGHTS,
    "interact_early_low": 1.0,
    "interact_early_mid": 1.0,
    "interact_early_high": 1.0,
    "interact_late_low": 1.0,
    "interact_late_mid": 1.0,
    "interact_late_high": 1.0,
}


def compute_distance(
    sim_targets: dict[str, float],
    emp_targets: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted mean squared relative error across target components."""
    if weights is None:
        weights = STAGE1_WEIGHTS

    total = 0.0
    w_sum = 0.0
    for key, w in weights.items():
        e = emp_targets.get(key)
        s = sim_targets.get(key)
        if e is None or s is None or np.isnan(e) or np.isnan(s) or abs(e) < 1e-12:
            continue
        total += w * ((s - e) / abs(e)) ** 2
        w_sum += w
    return total / max(w_sum, 1e-12)


# ── parameter sampling ───────────────────────────────────────────────────────

PARAM_RANGES: dict[str, tuple[float, float]] = {
    "sigma_star": (0.002, 0.015),
    "jump_prob": (0.001, 0.015),
    "jump_size_std": (0.03, 0.25),
    "n_informed": (8, 25),
    "n_noise": (15, 45),
    "n_herding": (5, 20),
    "p_informed": (0.25, 0.65),
    "p_noise": (0.12, 0.45),
    "p_herding": (0.12, 0.45),
    "deadline_ramp": (0.5, 5.0),
    "informed_sensitivity": (0.15, 1.5),
    "noise_scale": (0.4, 2.5),
    "herding_lookback": (2, 12),
    "herding_strength": (0.2, 1.5),
    "max_position": (5.0, 18.0),
    "max_order": (0.8, 5.0),
    "lambda_impact": (0.0005, 0.010),
    "beta_impact": (0.3, 0.75),
    "no_move_threshold": (0.15, 2.0),
}

INT_PARAMS = {"n_informed", "n_noise", "n_herding", "herding_lookback"}


def _sample_params(rng: np.random.Generator, stage: int = 1) -> ABMParams:
    d: dict = {}
    for k, (lo, hi) in PARAM_RANGES.items():
        if k in INT_PARAMS:
            d[k] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            d[k] = float(rng.uniform(lo, hi))
    params = ABMParams(**d)
    if stage >= 2:
        params.participation_excite = float(rng.uniform(0.3, 4.0))
        params.depth_activity_scale = float(rng.uniform(0.1, 2.5))
    return params


# ── search routine ───────────────────────────────────────────────────────────

def random_search(
    emp_targets: dict[str, float],
    *,
    n_iter: int = 200,
    n_seeds: int = 3,
    stage: int = 1,
    T_hours: int | None = None,
    verbose: bool = True,
) -> tuple[ABMParams, float, list[dict]]:
    """Random-search calibration.

    Returns ``(best_params, best_distance, all_results)``.
    """
    rng = np.random.default_rng(42)
    weights = STAGE2_WEIGHTS if stage >= 2 else STAGE1_WEIGHTS
    include_interaction = stage >= 2

    best_dist = np.inf
    best_params: ABMParams | None = None
    results: list[dict] = []

    for i in range(n_iter):
        params = _sample_params(rng, stage=stage)
        if T_hours is not None:
            params.T_hours = T_hours

        dists: list[float] = []
        for s in range(n_seeds):
            try:
                res = simulate_market(params, seed=s * 13 + 7)
                df = sim_to_features(res, params, seed=s, stage=stage)
                st = compute_targets(df, include_interaction=include_interaction)
                d = compute_distance(st, emp_targets, weights)
                dists.append(d)
            except Exception:
                dists.append(1e6)

        mean_d = float(np.mean(dists))
        results.append({"params": params, "distance": mean_d, "iter": i})

        if mean_d < best_dist:
            best_dist = mean_d
            best_params = params
            if verbose:
                print(f"  iter {i:4d}  new best distance = {best_dist:.6f}")

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_iter}] best so far = {best_dist:.6f}")

    assert best_params is not None
    return best_params, best_dist, results
