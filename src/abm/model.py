"""
Logit-space agent-based model for prediction market dynamics.

The model is intentionally reduced-form:

- Hourly time steps on a logit price scale
- Three trader classes only: informed, noise, herding
- Hard position / order limits
- A latent anchor with clustered news shocks
- Stage 1: static participation + deterministic deadline ramp
- Stage 2: endogenous attention + state-dependent depth

The simulator writes feature-style panels so the empirical analysis
pipeline can be reused on synthetic data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

# Match empirical clipping in ``src/transform.py``
CLAMP_LO = 0.001
CLAMP_HI = 0.999
LOGIT_LO = float(np.log(CLAMP_LO / (1.0 - CLAMP_LO)))
LOGIT_HI = float(np.log(CLAMP_HI / (1.0 - CLAMP_HI)))

MIN_EXECUTION = 0.05


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Stable sigmoid with clipping."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def _stage2_enabled(params: "ABMParams") -> bool:
    return params.participation_excite > 0.0 or params.depth_activity_scale > 0.0


def _deadline_pressure(tau_days: float) -> float:
    """Linear pressure ramp in the final 30 days."""
    if tau_days >= 30.0:
        return 0.0
    return max(0.0, (30.0 - tau_days) / 30.0)


def _build_market_context(
    rng: np.random.Generator,
    params: "ABMParams",
) -> dict[str, float]:
    """
    Draw one light-weight market-level context.

    A single latent popularity draw makes some markets both more active and
    deeper, while a separate news draw varies shockiness across markets.
    This is the minimum heterogeneity needed to mimic the pooled politics
    panel without introducing market-type hierarchies.
    """

    h = params.market_heterogeneity
    popularity = float(np.exp(h * rng.normal()))
    fragility = float(np.exp(0.60 * h * rng.normal()))
    news = float(np.exp(0.50 * h * rng.normal()))

    depth_scale = np.clip(popularity**1.10 / fragility, 0.35, 5.50)
    impact_scale = np.clip(fragility / (popularity**0.55), 0.25, 4.50)
    count_scale = np.clip(4.5 * popularity, 0.8, 20.0)
    volume_scale = np.clip(2_500.0 * popularity, 250.0, 25_000.0)

    return {
        "activity_scale": np.clip(popularity, 0.45, 4.50),
        "depth_scale": float(depth_scale),
        "impact_scale": float(impact_scale),
        "news_scale": np.clip(news, 0.45, 3.50),
        "count_scale": float(count_scale),
        "volume_scale": float(volume_scale),
        "x0_shift": float(np.clip(rng.normal(0.0, 0.55), -2.2, 2.2)),
        "x_star_shift": float(np.clip(rng.normal(0.0, 0.55), -2.2, 2.2)),
    }


@dataclass
class ABMParams:
    """Tuneable ABM parameters.

    The parameter vector remains compact by letting a few coefficients govern
    multiple stage-2 channels internally.
    """

    # Simulation length
    T_hours: int = 7200

    # Latent anchor / news
    sigma_star: float = 0.006
    jump_prob: float = 0.004
    jump_size_std: float = 0.18
    jump_decay: float = 0.82

    # Agent counts
    n_informed: int = 18
    n_noise: int = 28
    n_herding: int = 14

    # Baseline participation probabilities
    p_informed: float = 0.18
    p_noise: float = 0.08
    p_herding: float = 0.08

    # Deadline ramp
    deadline_ramp: float = 2.25

    # Trading behaviour
    informed_sensitivity: float = 1.60
    noise_scale: float = 1.75
    herding_lookback: int = 8
    herding_strength: float = 1.10

    # Inventory / order limits
    max_position: float = 12.0
    max_order: float = 2.50

    # Price impact
    lambda_impact: float = 0.0045
    beta_impact: float = 1.15
    no_move_threshold: float = 1.10

    # Stage-2 additions
    participation_excite: float = 0.0
    depth_activity_scale: float = 0.0
    vol_lookback: int = 24

    # Panel heterogeneity
    market_heterogeneity: float = 0.55

    # Initial conditions
    x0: float = 0.0
    x_star_0: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def simulate_market(
    params: ABMParams,
    seed: int = 42,
    market_context: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Run one simulated market path.

    The agent logic uses target inventories rather than raw one-shot orders.
    That change removes the old saw-tooth artifacts and makes the position
    constraints behave more naturally.
    """

    rng = np.random.default_rng(seed)
    T = int(params.T_hours)
    stage2 = _stage2_enabled(params)

    context = market_context or {}
    activity_scale = float(context.get("activity_scale", 1.0))
    depth_scale = float(context.get("depth_scale", 1.0))
    impact_scale = float(context.get("impact_scale", 1.0))
    news_scale = float(context.get("news_scale", 1.0))
    count_scale = float(context.get("count_scale", 4.5))
    volume_scale = float(context.get("volume_scale", 2_500.0))

    n_inf = int(params.n_informed)
    n_noi = int(params.n_noise)
    n_herd = int(params.n_herding)
    n_total = n_inf + n_noi + n_herd
    i_end = n_inf
    n_end = i_end + n_noi

    # Static cross-agent heterogeneity
    inf_skill = np.exp(0.25 * rng.normal(size=n_inf)) if n_inf else np.empty(0)
    noi_skill = np.exp(0.30 * rng.normal(size=n_noi)) if n_noi else np.empty(0)
    herd_skill = np.exp(0.25 * rng.normal(size=n_herd)) if n_herd else np.empty(0)

    # State
    x = np.zeros(T)
    x_star = np.zeros(T)
    x[0] = np.clip(params.x0 + float(context.get("x0_shift", 0.0)), LOGIT_LO, LOGIT_HI)
    x_star[0] = np.clip(
        params.x_star_0 + float(context.get("x_star_shift", 0.0)),
        LOGIT_LO,
        LOGIT_HI,
    )
    positions = np.zeros(n_total)

    # Recorder arrays
    trade_count = np.zeros(T, dtype=np.int32)
    volume_arr = np.zeros(T)
    net_imb_arr = np.zeros(T)
    eff_depth_arr = np.ones(T)
    act_inf = np.zeros(T, dtype=np.int32)
    act_noi = np.zeros(T, dtype=np.int32)
    act_herd = np.zeros(T, dtype=np.int32)
    attention_arr = np.zeros(T)
    deadline_arr = np.zeros(T)

    news_flow = 0.0
    attention = 0.0
    prev_imbalance = 0.0

    for t in range(1, T):
        tau_days = (T - t) / 24.0
        deadline_pressure = _deadline_pressure(tau_days)
        deadline_arr[t] = deadline_pressure

        # Clustered latent-news process: a decaying shock with rare heavy-tailed jumps.
        news_flow = params.jump_decay * news_flow
        news_flow += params.sigma_star * news_scale * rng.standard_normal()
        jump_event = 0.0
        if rng.random() < params.jump_prob:
            jump = params.jump_size_std * news_scale * rng.standard_t(df=3)
            news_flow += jump
            jump_event = min(abs(jump) / max(params.jump_size_std, 1e-8), 8.0)

        x_star[t] = np.clip(x_star[t - 1] + news_flow, LOGIT_LO, LOGIT_HI)

        lookback = min(params.vol_lookback, t)
        if lookback > 0:
            recent_returns = np.diff(x[max(0, t - lookback): t + 1])
            recent_abs = float(np.mean(np.abs(recent_returns))) if len(recent_returns) else 0.0
        else:
            recent_abs = 0.0
        vol_signal = min(recent_abs / 0.02, 4.0)
        imbalance_signal = min(abs(prev_imbalance) / max(n_total * params.max_order, 1e-8), 4.0)

        if stage2:
            attention = 0.85 * attention
            attention += params.participation_excite * (
                0.30 * vol_signal + 0.15 * imbalance_signal + 0.20 * jump_event
            )
            attention = float(np.clip(attention, 0.0, 3.0))
        else:
            attention = 0.0
        attention_arr[t] = attention

        # Participation is type-specific. Herders and noise receive a stronger
        # deadline ramp, which is the minimal way to encode deadline crowding.
        p_inf = params.p_informed * activity_scale * (1.0 + params.deadline_ramp * 0.55 * deadline_pressure)
        p_noi = params.p_noise * activity_scale * (1.0 + params.deadline_ramp * 0.95 * deadline_pressure)
        p_herd = params.p_herding * activity_scale * (1.0 + params.deadline_ramp * 1.35 * deadline_pressure)

        mispricing = x_star[t] - x[t - 1]
        mispricing_signal = min(abs(mispricing) / 1.25, 4.0)

        if stage2:
            p_inf *= 1.0 + 0.20 * mispricing_signal + 0.10 * attention
            p_noi *= 1.0 + 0.05 * attention
            p_herd *= 1.0 + 0.05 * attention + 0.16 * attention * deadline_pressure

        p_inf = float(np.clip(p_inf, 0.0, 0.985))
        p_noi = float(np.clip(p_noi, 0.0, 0.985))
        p_herd = float(np.clip(p_herd, 0.0, 0.985))

        active_inf = rng.random(n_inf) < p_inf if n_inf else np.zeros(0, dtype=bool)
        active_noi = rng.random(n_noi) < p_noi if n_noi else np.zeros(0, dtype=bool)
        active_herd = rng.random(n_herd) < p_herd if n_herd else np.zeros(0, dtype=bool)

        orders = np.zeros(n_total)

        # Informed traders move toward a target inventory based on the latent anchor gap.
        if n_inf:
            target_inf = params.max_position * np.tanh(
                params.informed_sensitivity * mispricing / 1.6
            ) * inf_skill
            np.clip(target_inf, -params.max_position, params.max_position, out=target_inf)
            orders[:i_end] = 0.42 * (target_inf - positions[:i_end])

        # Noise traders submit transient target positions drawn from a heavy-tailed shock.
        if n_noi:
            noise_draw = params.noise_scale * noi_skill * rng.standard_t(df=3, size=n_noi)
            target_noi = params.max_position * np.tanh(noise_draw / 3.6)
            if stage2:
                target_noi *= 1.0 + 0.04 * attention + 0.08 * deadline_pressure
            np.clip(target_noi, -params.max_position, params.max_position, out=target_noi)
            orders[i_end:n_end] = 0.58 * (target_noi - positions[i_end:n_end])

        # Herders respond to recent trend and partial order-flow persistence.
        lb = min(params.herding_lookback, t)
        if lb > 0:
            trend = x[t - 1] - x[max(0, t - 1 - lb)]
        else:
            trend = 0.0
        if n_herd:
            herd_signal = np.tanh(params.herding_strength * (trend + 0.18 * prev_imbalance) / 0.38)
            target_herd = params.max_position * herd_signal * herd_skill
            if stage2:
                target_herd *= 1.0 + 0.08 * attention + 0.22 * deadline_pressure
            np.clip(target_herd, -params.max_position, params.max_position, out=target_herd)
            orders[n_end:] = 0.52 * (target_herd - positions[n_end:])

        # Deadline urgency only becomes endogenous in stage 2.
        max_order_t = params.max_order * (1.0 + (0.10 * deadline_pressure + 0.04 * attention if stage2 else 0.0))
        np.clip(orders, -max_order_t, max_order_t, out=orders)

        # Enforce activity draws.
        if n_inf:
            orders[:i_end] *= active_inf
        if n_noi:
            orders[i_end:n_end] *= active_noi
        if n_herd:
            orders[n_end:] *= active_herd

        n_participants = int(active_inf.sum()) + int(active_noi.sum()) + int(active_herd.sum())

        # Enforce hard position limits.
        room_hi = params.max_position - positions
        room_lo = -params.max_position - positions
        np.maximum(orders, room_lo, out=orders)
        np.minimum(orders, room_hi, out=orders)

        # No-move possibility starts at the order level: tiny desired adjustments
        # simply do not execute.
        orders[np.abs(orders) < MIN_EXECUTION] = 0.0
        executed = np.abs(orders) >= MIN_EXECUTION

        I = float(orders.sum())
        abs_vol = float(np.abs(orders[executed]).sum()) if executed.any() else 0.0
        n_exec = int(executed.sum())

        depth = depth_scale
        if stage2:
            depth = 1.0 + (depth - 1.0) * (1.0 - 0.75 * deadline_pressure)
        if stage2 and n_exec > 0:
            active_share = n_exec / max(n_total, 1)
            inf_share = float(executed[:i_end].sum()) / n_exec if n_inf else 0.0
            herd_share = float(executed[n_end:].sum()) / n_exec if n_herd else 0.0

            depth *= 1.0 + params.depth_activity_scale * active_share * (1.60 + 1.20 * inf_share)
            depth /= 1.0 + params.depth_activity_scale * (
                0.14 * deadline_pressure
                + 0.55 * deadline_pressure * active_share * (1.0 + herd_share)
                + 0.10 * attention * herd_share
                + 0.08 * imbalance_signal
            )
            depth = max(depth, 0.20)

        lam = params.lambda_impact * impact_scale / depth

        excess_imb = max(abs(I) - params.no_move_threshold, 0.0)
        if excess_imb > 0.0:
            delta_x = lam * np.sign(I) * (excess_imb ** params.beta_impact)
            x[t] = np.clip(x[t - 1] + delta_x, LOGIT_LO, LOGIT_HI)
        else:
            x[t] = x[t - 1]

        positions += orders
        prev_imbalance = I

        if n_exec > 0:
            flow_floor = 0.30 * activity_scale * n_total * (
                params.p_informed + params.p_noise + params.p_herding
            )
            base_trade_intensity = count_scale * (flow_floor + 0.40 * n_participants + 0.15 * n_exec)
            split_intensity = 0.20 * count_scale * abs_vol / max(params.max_order, 1e-8)
            trade_count[t] = int(max(n_exec, rng.poisson(base_trade_intensity + split_intensity)))
            volume_arr[t] = volume_scale * abs_vol

        net_imb_arr[t] = I
        eff_depth_arr[t] = depth
        act_inf[t] = int(executed[:i_end].sum()) if n_inf else 0
        act_noi[t] = int(executed[i_end:n_end].sum()) if n_noi else 0
        act_herd[t] = int(executed[n_end:].sum()) if n_herd else 0

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
        "attention_state": attention_arr,
        "deadline_pressure": deadline_arr,
    }


def sim_to_features(
    result: dict[str, np.ndarray],
    params: ABMParams,
    sim_market: str = "sim_0",
    seed: int = 0,
    stage: int = 1,
) -> pd.DataFrame:
    """
    Convert raw simulation output to a feature-style table.

    The schema intentionally mirrors the empirical politics feature panel so
    existing analysis notebooks can run on the synthetic data with minimal
    branching.
    """

    x = result["x"]
    T = len(x)
    price = np.clip(sigmoid(x), CLAMP_LO, CLAMP_HI)

    logit_return = np.empty(T)
    logit_return[0] = np.nan
    logit_return[1:] = np.diff(x)

    df = pd.DataFrame(
        {
            "timestamp": np.arange(T, dtype=np.int32),
            "price": price,
            "logit_price": x,
            "logit_return": logit_return,
            "abs_logit_return": np.abs(logit_return),
            "trade_count": result["trade_count"],
            "trade_rate_1h": result["trade_count"].astype(float),
            "volume_rate_1h": result["volume"],
            "time_to_resolution": (T - np.arange(T)) / 24.0,
            "sim_market": sim_market,
            "market_slug": sim_market,
            "stage": stage,
            "seed": seed,
            "x_star": result["x_star"],
            "net_imbalance": result["net_imbalance"],
            "effective_depth": result["effective_depth"],
            "active_informed": result["active_informed"],
            "active_noise": result["active_noise"],
            "active_herding": result["active_herding"],
            "attention_state": result["attention_state"],
            "deadline_pressure": result["deadline_pressure"],
            "category": "politics",
            "triad_id": "abm",
            "market_type": "simulated",
            "role": "simulated",
        }
    )

    df["trade_rate_24h"] = df["trade_rate_1h"].rolling(24, min_periods=1).sum()
    df["volume_rate_24h"] = df["volume_rate_1h"].rolling(24, min_periods=1).sum()
    return df


def panel_template_from_reference(
    reference: pd.DataFrame,
    market_col: str = "market_slug",
) -> list[dict[str, Any]]:
    """Extract market lengths from an empirical stacked panel."""
    if market_col not in reference.columns:
        raise KeyError(f"Missing market column '{market_col}' in reference DataFrame")

    specs: list[dict[str, Any]] = []
    for idx, (name, grp) in enumerate(reference.groupby(market_col, sort=False)):
        specs.append(
            {
                "name": f"sim_{idx}",
                "source_market": str(name),
                "T_hours": int(len(grp)),
            }
        )
    return specs


def simulate_panel(
    params: ABMParams,
    n_markets: int = 5,
    stage: int = 1,
    base_seed: int = 0,
    market_specs: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Run a heterogeneous synthetic panel.

    If ``market_specs`` is supplied, each spec may contain ``T_hours`` and
    ``name``. Otherwise ``n_markets`` identical-length markets are drawn.
    """

    panel_rng = np.random.default_rng(base_seed + 10_003)

    if market_specs is None:
        specs = [
            {
                "name": f"sim_{i}",
                "T_hours": int(params.T_hours),
            }
            for i in range(n_markets)
        ]
    else:
        specs = market_specs

    parts: list[pd.DataFrame] = []
    for i, spec in enumerate(specs):
        ctx = _build_market_context(panel_rng, params)
        params_i = ABMParams(**params.to_dict())
        params_i.T_hours = int(spec.get("T_hours", params.T_hours))

        seed_i = base_seed + i * 137 + 7
        res = simulate_market(params_i, seed=seed_i, market_context=ctx)
        parts.append(
            sim_to_features(
                res,
                params_i,
                sim_market=str(spec.get("name", f"sim_{i}")),
                seed=seed_i,
                stage=stage,
            )
        )

    return pd.concat(parts, ignore_index=True)
