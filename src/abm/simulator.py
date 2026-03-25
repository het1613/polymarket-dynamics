"""Minimal staged politics-first ABM in logit space."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from src.abm.config import MarketSkeleton, SimulationParams, logit_clip_bounds

LOGIT_LO, LOGIT_HI = logit_clip_bounds()

TYPE_COUNTS = {
    "informed": "n_informed",
    "noise": "n_noise",
    "herding": "n_herding",
}

TYPE_TRADE_WEIGHTS = {
    "informed": "trade_weight_informed",
    "noise": "trade_weight_noise",
    "herding": "trade_weight_herding",
}

BASE_PARTICIPATION_MULTIPLIER = {
    "informed": 0.80,
    "noise": 1.00,
    "herding": 0.85,
}

VOL_RESPONSE_MULTIPLIER = {
    "informed": 0.35,
    "noise": 0.80,
    "herding": 1.10,
}

ACTIVITY_RESPONSE_MULTIPLIER = {
    "informed": 0.20,
    "noise": 0.55,
    "herding": 1.00,
}


def _exposure_cost(position: float, price: float) -> float:
    """Approximate collateral consumed by a yes/no exposure."""
    if position >= 0:
        return float(position * price)
    return float((-position) * (1.0 - price))


def _execute_order(
    position: float,
    cash: float,
    desired: float,
    price: float,
    params: SimulationParams,
) -> tuple[float, float, float]:
    """Apply order-size, position, and budget limits to one desired trade."""
    if abs(desired) < 1e-12:
        return 0.0, position, cash

    desired = float(np.clip(desired, -params.max_order, params.max_order))
    target = float(np.clip(position + desired, -params.max_position, params.max_position))
    delta = target - position
    if abs(delta) < 1e-12:
        return 0.0, position, cash

    current_cost = _exposure_cost(position, price)
    proposed_cost = _exposure_cost(target, price)
    incremental_cost = proposed_cost - current_cost

    if incremental_cost <= cash + 1e-12:
        return delta, target, cash - incremental_cost

    if incremental_cost <= 0:
        return delta, target, cash - incremental_cost

    lo, hi = 0.0, 1.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        candidate = position + mid * delta
        candidate_cost = _exposure_cost(candidate, price)
        if candidate_cost - current_cost <= cash:
            lo = mid
        else:
            hi = mid

    executed = lo * delta
    final_position = position + executed
    final_cost = _exposure_cost(final_position, price)
    final_cash = cash - (final_cost - current_cost)
    return executed, final_position, final_cash


def _participation_probability(
    params: SimulationParams,
    trader_type: str,
    progress: float,
    recent_vol: float,
    recent_activity: float,
    attention_state: float,
) -> float:
    """Stage-wise participation function."""
    base = (
        params.baseline_participation
        * BASE_PARTICIPATION_MULTIPLIER[trader_type]
        * (1.0 + params.deadline_ramp * progress)
    )

    if params.stage in {"stage2", "stage3"}:
        vol_signal = np.clip(recent_vol / max(params.volatility_reference, 1e-6), 0.0, 3.0)
        act_signal = np.clip(recent_activity / max(params.activity_reference, 1e-6), 0.0, 3.0)
        att_signal = np.clip(attention_state, 0.0, 3.0)
        if trader_type == "noise":
            vol_weight = VOL_RESPONSE_MULTIPLIER[trader_type] * (0.95 - 0.35 * progress)
            act_weight = ACTIVITY_RESPONSE_MULTIPLIER[trader_type] * (1.35 - 1.10 * progress)
            attention_weight = 0.65 * (1.20 - 0.80 * progress)
        elif trader_type == "herding":
            vol_weight = VOL_RESPONSE_MULTIPLIER[trader_type] * (0.35 + 1.35 * progress)
            act_weight = ACTIVITY_RESPONSE_MULTIPLIER[trader_type] * (0.05 + 2.10 * progress)
            attention_weight = 0.10 * (0.20 + 1.10 * progress)
        else:
            vol_weight = VOL_RESPONSE_MULTIPLIER[trader_type] * (0.85 + 0.20 * progress)
            act_weight = ACTIVITY_RESPONSE_MULTIPLIER[trader_type] * (0.55 + 0.20 * progress)
            attention_weight = 0.20

        base *= 1.0 + params.vol_participation * vol_weight * vol_signal
        base *= 1.0 + params.activity_participation * act_weight * act_signal
        base *= 1.0 + params.activity_participation * attention_weight * att_signal

    return float(np.clip(base, 0.0, 0.95))


def _effective_depth(
    params: SimulationParams,
    progress: float,
    recent_activity: float,
    attention_state: float,
) -> float:
    """Stage-1 constant depth, then stage-2 state-dependent depth."""
    if params.stage not in {"stage2", "stage3"}:
        return params.base_depth

    act_signal = np.clip(
        recent_activity / max(params.activity_reference, 1e-6) + 0.75 * attention_state,
        0.0,
        4.0,
    )
    depth_multiplier = (
        1.0
        + params.depth_activity_coef * act_signal * (1.0 - progress)
        - params.depth_deadline_coef * progress
    )
    depth_multiplier = float(np.clip(depth_multiplier, 0.15, 6.0))
    return float(np.clip(params.base_depth * depth_multiplier, 0.50, 6.0 * params.base_depth))


def _desired_orders(
    trader_type: str,
    n_agents: int,
    params: SimulationParams,
    observed_x: float,
    x_star: float,
    trend_state: float,
    progress: float,
    recent_activity: float,
    attention_state: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Type-specific desired orders before caps and budgets."""
    size_scale = 1.0
    if params.stage in {"stage2", "stage3"}:
        activity_signal = np.clip(
            recent_activity / max(params.activity_reference, 1e-6) + 0.50 * attention_state,
            0.0,
            3.0,
        )
        if trader_type == "noise":
            size_scale /= 1.0 + 0.35 * params.depth_activity_coef * activity_signal * (1.0 - progress)
        elif trader_type == "informed":
            size_scale /= 1.0 + 0.20 * params.depth_activity_coef * activity_signal * (1.0 - progress)
        else:
            size_scale *= 1.0 + 0.15 * params.depth_interaction_coef * activity_signal * progress

    if trader_type == "informed":
        edge = x_star - observed_x
        if abs(edge) < params.mispricing_threshold:
            return np.zeros(n_agents, dtype=float)
        magnitude = np.tanh((abs(edge) / 0.22) * params.informed_strength)
        sizes = (
            params.max_order
            * size_scale
            * magnitude
            * rng.uniform(0.55, 1.00, size=n_agents)
        )
        return np.sign(edge) * sizes

    if trader_type == "noise":
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_agents)
        sizes = rng.gamma(shape=1.5, scale=max(params.noise_scale, 1e-3), size=n_agents)
        sizes = np.clip(sizes * size_scale, 0.0, params.max_order)
        # Preserve a genuine no-trade possibility.
        sizes[rng.random(n_agents) < 0.15] = 0.0
        return signs * sizes

    trend_mag = np.tanh((abs(trend_state) / 0.05) * params.herding_strength)
    if abs(trend_state) < 0.001 or trend_mag < 0.015:
        return np.zeros(n_agents, dtype=float)
    sizes = (
        params.max_order
        * size_scale
        * trend_mag
        * rng.uniform(0.45, 1.00, size=n_agents)
    )
    return np.sign(trend_state) * sizes


def simulate_market(
    skeleton: MarketSkeleton,
    params: SimulationParams,
    rng: np.random.Generator,
    *,
    seed: int,
) -> pd.DataFrame:
    """Simulate one market on the empirical hourly scaffold."""
    n_steps = skeleton.n_steps
    timestamps = pd.to_datetime(list(skeleton.timestamps), utc=True)
    ttr = np.asarray(skeleton.time_to_resolution, dtype=float)
    max_ttr = max(float(ttr.max()), 1e-6)

    positions = {
        name: np.zeros(getattr(params, field), dtype=float)
        for name, field in TYPE_COUNTS.items()
    }
    cash = {
        name: np.full(getattr(params, field), params.initial_cash, dtype=float)
        for name, field in TYPE_COUNTS.items()
    }

    logit_price = np.zeros(n_steps, dtype=float)
    price = np.zeros(n_steps, dtype=float)
    x_star_series = np.zeros(n_steps, dtype=float)
    net_imbalance = np.zeros(n_steps, dtype=float)
    effective_depth = np.zeros(n_steps, dtype=float)
    trade_count = np.zeros(n_steps, dtype=int)
    volume = np.zeros(n_steps, dtype=float)
    active_counts = {name: np.zeros(n_steps, dtype=int) for name in TYPE_COUNTS}

    x_core = float(np.clip(skeleton.initial_logit + rng.normal(0.0, 0.05), LOGIT_LO, LOGIT_HI))
    x_star = float(np.clip(x_core + rng.normal(0.0, params.anchor_sigma), LOGIT_LO, LOGIT_HI))
    news_impulse = 0.0
    dislocation = 0.0
    trend_state = 0.0
    recent_vol = 0.0
    recent_activity = 0.0
    attention_state = 0.0

    observed_x = float(np.clip(x_core + dislocation, LOGIT_LO, LOGIT_HI))
    logit_price[0] = observed_x
    price[0] = float(np.clip(expit(observed_x), 0.001, 0.999))
    x_star_series[0] = x_star
    effective_depth[0] = params.base_depth

    for t in range(1, n_steps):
        progress = float(np.clip(1.0 - (ttr[t] / max_ttr), 0.0, 1.0))
        price_prev = float(np.clip(expit(observed_x), 0.001, 0.999))

        if params.stage in {"stage2", "stage3"}:
            attention_burst = rng.gamma(shape=1.4, scale=0.45) * (1.20 - 0.50 * progress)
            attention_state = (
                0.92 * attention_state
                + 0.08 * attention_burst
                + 0.03 * np.clip(recent_activity / max(params.activity_reference, 1e-6), 0.0, 3.0)
            )
            attention_state = float(np.clip(attention_state, 0.0, 3.0))
        else:
            attention_state = 0.0

        news_impulse = params.anchor_phi * news_impulse + rng.normal(0.0, params.anchor_sigma)
        if rng.random() < params.jump_prob:
            news_impulse += rng.normal(0.0, params.jump_scale)
        x_star = 0.985 * x_star + news_impulse
        x_star = float(np.clip(x_star, LOGIT_LO, LOGIT_HI))

        depth_t = _effective_depth(params, progress, recent_activity, attention_state)
        effective_depth[t] = depth_t

        total_orders = []
        total_weighted_trades = 0
        total_raw_trades = 0
        total_volume = 0.0

        for trader_type, field in TYPE_COUNTS.items():
            n_agents = getattr(params, field)
            p_active = _participation_probability(
                params,
                trader_type,
                progress,
                recent_vol,
                recent_activity,
                attention_state,
            )
            active_mask = rng.random(n_agents) < p_active
            active_idx = np.flatnonzero(active_mask)
            if active_idx.size == 0:
                continue

            desired = _desired_orders(
                trader_type,
                active_idx.size,
                params,
                observed_x,
                x_star,
                trend_state,
                progress,
                recent_activity,
                attention_state,
                rng,
            )
            submitted_count = int(np.count_nonzero(np.abs(desired) > 1e-10))
            executed_count = 0
            orders = np.zeros(n_agents, dtype=float)
            for i, agent_idx in enumerate(active_idx):
                executed, new_position, new_cash = _execute_order(
                    positions[trader_type][agent_idx],
                    cash[trader_type][agent_idx],
                    desired[i],
                    price_prev,
                    params,
                )
                if abs(executed) < 1e-10:
                    continue
                positions[trader_type][agent_idx] = new_position
                cash[trader_type][agent_idx] = new_cash
                orders[agent_idx] = executed
                executed_count += 1
                total_volume += abs(executed)

            if executed_count:
                total_orders.append(float(orders.sum()))
            if submitted_count:
                reported_weight = float(getattr(params, TYPE_TRADE_WEIGHTS[trader_type]))
                if params.stage in {"stage2", "stage3"}:
                    if trader_type == "noise":
                        reported_weight *= 1.0 + 0.60 * attention_state * (1.0 - progress)
                    elif trader_type == "informed":
                        reported_weight *= 1.0 + 0.20 * attention_state * (1.0 - 0.50 * progress)
                total_weighted_trades += submitted_count * reported_weight
                total_raw_trades += submitted_count
                active_counts[trader_type][t] = submitted_count

        imbalance = float(np.sum(total_orders)) if total_orders else 0.0
        net_imbalance[t] = imbalance

        delta_x = 0.0
        excess_imbalance = max(abs(imbalance) - params.no_move_threshold, 0.0)
        if excess_imbalance > 0:
            activity_signal = np.clip(
                (0.60 * recent_activity + 0.40 * total_raw_trades)
                / max(params.activity_reference, 1e-6),
                0.0,
                4.0,
            )
            total_active = max(total_raw_trades, 1)
            noise_share = active_counts["noise"][t] / total_active
            herding_share = active_counts["herding"][t] / total_active
            gross_flow = max(total_volume, excess_imbalance, 1e-6)
            balance_ratio = float(np.clip(excess_imbalance / gross_flow, 0.05, 1.0))

            early_buffer = 1.0
            crowding_multiplier = 1.0
            thin_market_multiplier = 1.0
            if params.stage in {"stage2", "stage3"}:
                early_buffer += (
                    params.depth_activity_coef
                    * activity_signal
                    * (1.0 - progress)
                    * (1.10 - balance_ratio)
                    * (0.70 + 0.90 * noise_share)
                )
                crowding_multiplier += (
                    params.depth_interaction_coef
                    * activity_signal
                    * progress
                    * (balance_ratio + herding_share)
                )
                thin_market_multiplier += 2.75 * max(0.0, 1.0 - activity_signal) * (
                    0.90 - 0.35 * progress
                )

            delta_x = (
                params.impact_lambda
                * np.sign(imbalance)
                * (excess_imbalance**params.impact_beta)
                * np.sqrt(balance_ratio)
                * crowding_multiplier
                * thin_market_multiplier
                / max(depth_t * early_buffer, 1e-6)
            )

        x_core = float(np.clip(observed_x - dislocation + delta_x, LOGIT_LO, LOGIT_HI))
        dislocation *= params.dislocation_decay

        if (
            params.stage == "stage3"
            and total_raw_trades <= params.dislocation_activity_threshold
            and abs(imbalance) >= params.no_move_threshold * 1.5
            and rng.random() < params.dislocation_prob
        ):
            dislocation += params.dislocation_scale * np.sign(imbalance) * (
                abs(imbalance) / max(depth_t, 1e-6)
            ) ** params.impact_beta

        observed_x = float(np.clip(x_core + dislocation, LOGIT_LO, LOGIT_HI))
        logit_price[t] = observed_x
        price[t] = float(np.clip(expit(observed_x), 0.001, 0.999))
        x_star_series[t] = x_star

        trade_count[t] = int(total_weighted_trades)
        volume[t] = float(total_volume * params.volume_scale)

        logit_return_t = observed_x - logit_price[t - 1]
        trend_state = float(
            np.clip(params.trend_memory * trend_state + logit_return_t, -1.5, 1.5)
        )
        recent_vol = 0.92 * recent_vol + 0.08 * abs(logit_return_t)
        recent_activity = 0.88 * recent_activity + 0.12 * total_raw_trades

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "hour_index": np.arange(n_steps, dtype=int),
            "open": np.r_[price[0], price[:-1]],
            "high": np.maximum(np.r_[price[0], price[:-1]], price),
            "low": np.minimum(np.r_[price[0], price[:-1]], price),
            "price": price,
            "n_obs": 1,
            "market_slug": skeleton.market_slug,
            "sim_market": skeleton.market_slug,
            "category": "politics",
            "market_type": skeleton.market_type,
            "logit_price": logit_price,
            "logit_return": np.r_[np.nan, np.diff(logit_price)],
            "abs_logit_return": np.r_[np.nan, np.abs(np.diff(logit_price))],
            "time_to_resolution": ttr,
            "trade_count": trade_count,
            "trade_rate_1h": trade_count,
            "volume_rate_1h": volume,
            "x_star": x_star_series,
            "net_imbalance": net_imbalance,
            "effective_depth": effective_depth,
            "active_informed": active_counts["informed"],
            "active_noise": active_counts["noise"],
            "active_herding": active_counts["herding"],
            "stage": params.stage,
            "seed": seed,
            "role": "",
            "triad_id": "",
        }
    )
    df["trade_rate_24h"] = df["trade_rate_1h"].rolling(24, min_periods=1).sum()
    df["volume_rate_24h"] = df["volume_rate_1h"].rolling(24, min_periods=1).sum()
    return df


def simulate_panel(
    skeletons: list[MarketSkeleton],
    params: SimulationParams,
    *,
    seed: int,
) -> pd.DataFrame:
    """Simulate the full politics panel for one parameter vector and seed."""
    root_rng = np.random.default_rng(seed)
    panels = []
    for idx, skeleton in enumerate(skeletons):
        market_seed = int(root_rng.integers(0, 2**32 - 1)) + idx
        market_rng = np.random.default_rng(market_seed)
        panels.append(simulate_market(skeleton, params, market_rng, seed=seed))
    return pd.concat(panels, ignore_index=True)
