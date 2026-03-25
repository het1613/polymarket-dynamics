"""
Logit-space agent-based model for prediction market dynamics.

Three trader classes (informed, noise, herding) interact through
reduced-form price impact.  The state evolves at hourly time steps on
an unbounded logit scale; a latent anchor drives informed-trader demand.

Stage 1 — fixed base participation + deterministic deadline ramp.
Stage 2 — adds self-exciting participation + state-dependent depth.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd

# ── constants (match empirical pipeline in src/transform.py) ─────────────────
CLAMP_LO = 0.001
CLAMP_HI = 0.999
LOGIT_LO = float(np.log(CLAMP_LO / (1.0 - CLAMP_LO)))  # ≈ −6.91
LOGIT_HI = float(np.log(CLAMP_HI / (1.0 - CLAMP_HI)))  # ≈ +6.91


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


# ── parameter container ──────────────────────────────────────────────────────

@dataclass
class ABMParams:
    """All tuneable model knobs.  Stage-2 extensions default to *off*."""

    # simulation length
    T_hours: int = 7200

    # --- latent anchor / news ---
    sigma_star: float = 0.006
    jump_prob: float = 0.005
    jump_size_std: float = 0.12

    # --- agent counts ---
    n_informed: int = 15
    n_noise: int = 30
    n_herding: int = 12

    # --- base participation probabilities ---
    p_informed: float = 0.40
    p_noise: float = 0.25
    p_herding: float = 0.25

    # --- deadline ramp (multiplicative boost, last 30 simulated days) ---
    deadline_ramp: float = 2.5

    # --- agent behaviour ---
    informed_sensitivity: float = 0.6
    noise_scale: float = 1.2
    herding_lookback: int = 5
    herding_strength: float = 0.7

    # --- position / order limits ---
    max_position: float = 10.0
    max_order: float = 3.0

    # --- price impact ---
    lambda_impact: float = 0.003
    beta_impact: float = 0.5
    no_move_threshold: float = 0.5

    # --- stage-2 additions (set > 0 to activate) ---
    participation_excite: float = 0.0
    depth_activity_scale: float = 0.0
    vol_lookback: int = 24

    # --- initial conditions ---
    x0: float = 0.0
    x_star_0: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── core simulation ──────────────────────────────────────────────────────────

def simulate_market(params: ABMParams, seed: int = 42) -> dict[str, np.ndarray]:
    """Run one market simulation.  Returns a dict of 1-D arrays (length T)."""

    rng = np.random.default_rng(seed)
    T = params.T_hours

    n_inf = params.n_informed
    n_noi = params.n_noise
    n_herd = params.n_herding
    n_total = n_inf + n_noi + n_herd
    i_end = n_inf
    n_end = i_end + n_noi

    # --- state ---
    x = np.zeros(T)
    x_star = np.zeros(T)
    x[0] = params.x0
    x_star[0] = params.x_star_0
    positions = np.zeros(n_total)

    # --- output recorders ---
    trade_count = np.zeros(T, dtype=np.int32)
    volume_arr = np.zeros(T)
    net_imb_arr = np.zeros(T)
    eff_depth_arr = np.ones(T)
    act_inf = np.zeros(T, dtype=np.int32)
    act_noi = np.zeros(T, dtype=np.int32)
    act_herd = np.zeros(T, dtype=np.int32)

    # --- base participation vector ---
    p_base = np.empty(n_total)
    p_base[:i_end] = params.p_informed
    p_base[i_end:n_end] = params.p_noise
    p_base[n_end:] = params.p_herding

    # pre-draw all random numbers for the anchor (faster than per-step)
    star_noise = params.sigma_star * rng.standard_normal(T)
    star_jumps = params.jump_size_std * rng.standard_normal(T)
    star_jump_mask = rng.random(T) < params.jump_prob

    for t in range(1, T):
        # 1. anchor
        x_star[t] = x_star[t - 1] + star_noise[t]
        if star_jump_mask[t]:
            x_star[t] += star_jumps[t]

        # 2. deadline factor
        tau = (T - t) / 24.0
        dl = 1.0 + params.deadline_ramp * max(0.0, (30.0 - tau) / 30.0) if tau < 30.0 else 1.0

        # 3. self-exciting factor (stage 2)
        exc = 1.0
        if params.participation_excite > 0.0 and t > params.vol_lookback:
            w = x[t - params.vol_lookback: t]
            rv = np.std(np.diff(w)) if len(w) > 2 else 0.0
            exc = 1.0 + params.participation_excite * rv / max(params.sigma_star, 1e-8)

        # 4. participation draw
        p_active = np.minimum(p_base * dl * exc, 1.0)
        active = rng.random(n_total) < p_active

        # 5. orders
        orders = np.zeros(n_total)

        mispricing = x_star[t] - x[t - 1]
        orders[:i_end] = params.informed_sensitivity * mispricing * (0.5 + 0.5 * rng.random(n_inf))

        orders[i_end:n_end] = params.noise_scale * rng.standard_normal(n_noi)

        lb = min(params.herding_lookback, t)
        if lb > 0:
            recent_ret = x[t - 1] - x[max(0, t - 1 - lb)]
            h_sig = np.sign(recent_ret) if abs(recent_ret) > 1e-8 else 0.0
        else:
            h_sig = 0.0
        orders[n_end:] = params.herding_strength * h_sig * (0.5 + 0.5 * rng.random(n_herd))

        # clip by max order
        np.clip(orders, -params.max_order, params.max_order, out=orders)

        # clip by position limits
        room_hi = params.max_position - positions
        room_lo = -params.max_position - positions
        np.maximum(orders, room_lo, out=orders)
        np.minimum(orders, room_hi, out=orders)

        # zero-out inactive
        orders *= active

        # 6. aggregate
        I = float(orders.sum())
        n_act = int(active.sum())
        vol = float(np.abs(orders[active]).sum()) if n_act > 0 else 0.0

        # 7. state-dependent depth (stage 2)
        depth = 1.0
        if params.depth_activity_scale > 0.0 and n_act > 0:
            depth = 1.0 + params.depth_activity_scale * n_act / n_total
        lam = params.lambda_impact / depth

        # 8. impact
        if abs(I) < params.no_move_threshold:
            x[t] = x[t - 1]
        else:
            x[t] = x[t - 1] + lam * np.sign(I) * (abs(I) ** params.beta_impact)

        x[t] = np.clip(x[t], LOGIT_LO, LOGIT_HI)

        # 9. update positions
        positions += orders

        # 10. record
        trade_count[t] = n_act
        volume_arr[t] = vol
        net_imb_arr[t] = I
        eff_depth_arr[t] = depth
        act_inf[t] = int(active[:i_end].sum())
        act_noi[t] = int(active[i_end:n_end].sum())
        act_herd[t] = int(active[n_end:].sum())

    return {
        "x": x,
        "x_star": x_star,
        "trade_count": trade_count,
        "volume": volume_arr,
        "net_imbalance": net_imb_arr,
        "effective_depth": eff_depth_arr,
        "active_informed": act_inf,
        "active_noise": act_noi,
        "active_herding": act_herd,
    }


# ── feature-table conversion ────────────────────────────────────────────────

def sim_to_features(
    result: dict[str, np.ndarray],
    params: ABMParams,
    sim_market: str = "sim_0",
    seed: int = 0,
    stage: int = 1,
) -> pd.DataFrame:
    """Convert raw arrays to a feature DataFrame matching the empirical
    ``features_politics.parquet`` schema."""

    x = result["x"]
    T = len(x)

    price = np.clip(sigmoid(x), CLAMP_LO, CLAMP_HI)
    lr = np.empty(T)
    lr[0] = np.nan
    lr[1:] = np.diff(x)

    return pd.DataFrame({
        "timestamp": np.arange(T),
        "price": price,
        "logit_price": x,
        "logit_return": lr,
        "abs_logit_return": np.abs(lr),
        "trade_count": result["trade_count"],
        "trade_rate_1h": result["trade_count"].astype(float),
        "volume_rate_1h": result["volume"],
        "time_to_resolution": (T - np.arange(T)) / 24.0,
        "sim_market": sim_market,
        "stage": stage,
        "seed": seed,
        "x_star": result["x_star"],
        "net_imbalance": result["net_imbalance"],
        "effective_depth": result["effective_depth"],
        "active_informed": result["active_informed"],
        "active_noise": result["active_noise"],
        "active_herding": result["active_herding"],
    })


def simulate_panel(
    params: ABMParams,
    n_markets: int = 5,
    stage: int = 1,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Run *n_markets* independent simulations and stack into one panel."""
    parts: list[pd.DataFrame] = []
    for i in range(n_markets):
        seed_i = base_seed + i * 137
        res = simulate_market(params, seed=seed_i)
        df = sim_to_features(res, params, sim_market=f"sim_{i}", seed=seed_i, stage=stage)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
