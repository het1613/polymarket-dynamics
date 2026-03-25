"""Evaluation, target extraction, and stage-wise scoring for the ABM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from scipy.stats import qmc
from statsmodels.tsa.stattools import acf

from src.abm.config import (
    ACTIVITY_LABELS,
    ACF_LAGS,
    EARLY_DEADLINE_DAYS,
    QUANTILES,
    QUANTILES_CALIBRATION,
    REVERSAL_HORIZONS,
)
from src.analyze.return_distribution import return_stats
from src.analyze.reversal import conditional_returns
from src.analyze.volatility import abs_return_acf


def latin_hypercube_draws(
    search_space: dict[str, tuple[float, float]],
    n_draws: int,
    seed: int,
) -> list[dict[str, float]]:
    """Draw Latin-hypercube candidates from a bounded box."""
    if n_draws <= 0:
        return []

    keys = list(search_space)
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    raw = sampler.random(n=n_draws)
    scaled = qmc.scale(
        raw,
        [search_space[key][0] for key in keys],
        [search_space[key][1] for key in keys],
    )
    return [dict(zip(keys, row)) for row in scaled]


def compute_tail_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Robust tail summaries for overall, early, and late market life."""
    rows = []
    periods = {
        "Overall": df,
        "Earlier": df.loc[df["time_to_resolution"] > EARLY_DEADLINE_DAYS],
        "Last 30d": df.loc[df["time_to_resolution"] <= EARLY_DEADLINE_DAYS],
    }
    for period, subset in periods.items():
        alr = subset["abs_logit_return"].dropna()
        lr = subset["logit_return"].dropna()
        if len(alr) == 0 or len(lr) == 0:
            continue
        rows.append(
            {
                "period": period,
                "q90": float(alr.quantile(0.90)),
                "q95": float(alr.quantile(0.95)),
                "q99": float(alr.quantile(0.99)),
                "mean_abs_r": float(alr.mean()),
                "std_r": float(lr.std()),
            }
        )
    return pd.DataFrame(rows).set_index("period")


def pooled_deadline_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Match the pooled last-30-days vs earlier summary used in RQ3."""
    rows = []
    for label, subset in (
        ("Earlier", df.loc[df["time_to_resolution"] > EARLY_DEADLINE_DAYS]),
        ("Last 30d", df.loc[df["time_to_resolution"] <= EARLY_DEADLINE_DAYS]),
    ):
        lr = subset["logit_return"].dropna()
        alr = subset["abs_logit_return"].dropna()
        if len(lr) == 0 or len(alr) == 0:
            continue
        rows.append(
            {
                "period": label,
                "n": int(len(lr)),
                "mean_abs_r": float(alr.mean()),
                "std_r": float(lr.std()),
                "kurtosis": float(sp_stats.kurtosis(lr.values)),
                "skewness": float(sp_stats.skew(lr.values)),
                "acf1_abs_r": float(acf(alr.values, nlags=1, fft=True)[1]),
                "mean_trade_rate": float(subset["trade_rate_1h"].dropna().mean()),
            }
        )
    return pd.DataFrame(rows).set_index("period")


def interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled 2 x 3 cross-tab used in the RQ3 interaction section."""
    clean = df[
        ["logit_return", "abs_logit_return", "trade_rate_1h", "time_to_resolution"]
    ].dropna()
    clean = clean.copy()
    clean["ttr_phase"] = np.where(
        clean["time_to_resolution"] <= EARLY_DEADLINE_DAYS,
        "Last 30d",
        "Earlier",
    )
    clean["activity_bin"] = pd.qcut(
        clean["trade_rate_1h"],
        3,
        labels=list(ACTIVITY_LABELS),
        duplicates="drop",
    )
    out = clean.groupby(["ttr_phase", "activity_bin"], observed=True).agg(
        mean_abs_r=("abs_logit_return", "mean"),
        std_r=("logit_return", "std"),
        n=("logit_return", "count"),
    )
    return out


def quantile_regression_table(
    df: pd.DataFrame,
    *,
    standardize: bool = False,
    quantiles: tuple[float, ...] = QUANTILES,
    sample_n: int | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Quantile regression table matching the RQ3 notebook."""
    clean = df[["abs_logit_return", "trade_rate_1h", "time_to_resolution"]].dropna()
    if sample_n is not None and len(clean) > sample_n:
        clean = clean.sample(sample_n, random_state=random_state)

    if standardize:
        clean = clean.copy()
        for col in ("trade_rate_1h", "time_to_resolution"):
            scale = clean[col].std(ddof=0)
            scale = scale if scale > 1e-12 else 1.0
            clean[col] = (clean[col] - clean[col].mean()) / scale

    y = clean["abs_logit_return"]
    X = sm.add_constant(clean[["trade_rate_1h", "time_to_resolution"]])
    rows = []
    for q in quantiles:
        qr_model = sm.QuantReg(y, X).fit(q=q)
        rows.append(
            {
                "quantile": float(q),
                "trade_rate_coef": float(qr_model.params["trade_rate_1h"]),
                "trade_rate_p": float(qr_model.pvalues["trade_rate_1h"]),
                "ttr_coef": float(qr_model.params["time_to_resolution"]),
                "ttr_p": float(qr_model.pvalues["time_to_resolution"]),
                "pseudo_r2": float(qr_model.prsquared),
            }
        )
    return pd.DataFrame(rows).set_index("quantile")


def reversal_by_condition(
    df: pd.DataFrame,
    condition_col: str,
    split_val: float,
    label_lo: str,
    label_hi: str,
) -> pd.DataFrame:
    """Match the pooled RQ3 reversal split helper."""
    clean = df[["logit_return", condition_col]].dropna()
    lo = clean.loc[clean[condition_col] <= split_val, ["logit_return"]].copy()
    hi = clean.loc[clean[condition_col] > split_val, ["logit_return"]].copy()

    parts = []
    for subset, label in ((lo, label_lo), (hi, label_hi)):
        if len(subset) < 50:
            continue
        cr = conditional_returns(
            subset, sigma_mult=2.0, horizons=list(REVERSAL_HORIZONS)
        )
        cr["condition"] = label
        parts.append(cr.reset_index())
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).set_index(["condition", "horizon"])


def compute_panel_metrics(
    df: pd.DataFrame,
    *,
    sample_n_for_quantile: int | None,
) -> dict[str, Any]:
    """Compute the pooled RQ2/RQ3 metrics used for staged ABM evaluation."""
    metrics: dict[str, Any] = {}
    metrics["return_stats"] = return_stats(df)
    metrics["tail_quantiles"] = compute_tail_quantiles(df)
    metrics["acf_abs"] = abs_return_acf(df, nlags=ACF_LAGS).set_index("lag")
    metrics["deadline_summary"] = pooled_deadline_summary(df)
    metrics["interaction"] = interaction_table(df)
    metrics["quantile_regression_raw"] = quantile_regression_table(
        df,
        standardize=False,
        quantiles=QUANTILES,
        sample_n=sample_n_for_quantile,
        random_state=11,
    )
    metrics["quantile_regression_std"] = quantile_regression_table(
        df,
        standardize=True,
        quantiles=QUANTILES_CALIBRATION,
        sample_n=sample_n_for_quantile,
        random_state=23,
    )

    median_tr = float(df[["trade_rate_1h"]].dropna()["trade_rate_1h"].median())
    metrics["reversal_activity"] = reversal_by_condition(
        df,
        "trade_rate_1h",
        median_tr,
        f"Low (≤{median_tr:.0f})",
        f"High (>{median_tr:.0f})",
    )
    metrics["reversal_deadline"] = reversal_by_condition(
        df,
        "time_to_resolution",
        EARLY_DEADLINE_DAYS,
        "Last 30d",
        "Earlier (>30d)",
    )
    return metrics


def _relative_error(sim: np.ndarray, emp: np.ndarray, floor: float = 1e-6) -> float:
    denom = np.maximum(np.abs(emp), floor)
    return float(np.mean(np.abs(sim - emp) / denom))


def _acf_error(sim_metrics: dict[str, Any], emp_metrics: dict[str, Any]) -> float:
    sim = sim_metrics["acf_abs"].loc[1:ACF_LAGS, "acf"].to_numpy()
    emp = emp_metrics["acf_abs"].loc[1:ACF_LAGS, "acf"].to_numpy()
    scale = np.sqrt(np.mean(emp**2)) + 1e-6
    return float(np.sqrt(np.mean((sim - emp) ** 2)) / scale)


def stage1_score(
    sim_metrics: dict[str, Any],
    emp_metrics: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Primary stage-1 distance: tails, ACF shape, and late-life volatility."""
    tail_cols = ["q90", "q95", "q99"]
    tail_sim = (
        sim_metrics["tail_quantiles"]
        .loc[["Overall", "Earlier", "Last 30d"], tail_cols]
        .to_numpy()
        .ravel()
    )
    tail_emp = (
        emp_metrics["tail_quantiles"]
        .loc[["Overall", "Earlier", "Last 30d"], tail_cols]
        .to_numpy()
        .ravel()
    )
    tail_error = _relative_error(tail_sim, tail_emp)

    deadline_cols = ["mean_abs_r", "std_r", "acf1_abs_r"]
    deadline_sim = (
        sim_metrics["deadline_summary"]
        .loc[["Earlier", "Last 30d"], deadline_cols]
        .to_numpy()
        .ravel()
    )
    deadline_emp = (
        emp_metrics["deadline_summary"]
        .loc[["Earlier", "Last 30d"], deadline_cols]
        .to_numpy()
        .ravel()
    )
    deadline_error = _relative_error(deadline_sim, deadline_emp)
    acf_error = _acf_error(sim_metrics, emp_metrics)

    sim_ratio = float(
        sim_metrics["deadline_summary"].loc["Last 30d", "mean_abs_r"]
        / max(sim_metrics["deadline_summary"].loc["Earlier", "mean_abs_r"], 1e-6)
    )
    emp_ratio = float(
        emp_metrics["deadline_summary"].loc["Last 30d", "mean_abs_r"]
        / max(emp_metrics["deadline_summary"].loc["Earlier", "mean_abs_r"], 1e-6)
    )
    ratio_error = float(abs(sim_ratio - emp_ratio) / max(abs(emp_ratio), 1e-6))
    monotonic_penalty = 0.0
    if sim_metrics["deadline_summary"].loc["Last 30d", "mean_abs_r"] <= sim_metrics["deadline_summary"].loc["Earlier", "mean_abs_r"]:
        monotonic_penalty += 1.0
    if sim_metrics["deadline_summary"].loc["Last 30d", "std_r"] <= sim_metrics["deadline_summary"].loc["Earlier", "std_r"]:
        monotonic_penalty += 1.0
    monotonic_penalty /= 2.0

    score = (
        0.35 * tail_error
        + 0.25 * acf_error
        + 0.25 * deadline_error
        + 0.10 * ratio_error
        + 0.05 * monotonic_penalty
    )
    return score, {
        "tail_error": tail_error,
        "acf_error": acf_error,
        "deadline_error": deadline_error,
        "ratio_error": ratio_error,
        "monotonic_penalty": monotonic_penalty,
        "total_score": score,
    }


def _interaction_pattern_penalty(table: pd.DataFrame) -> float:
    pivot = table["mean_abs_r"].unstack("activity_bin")
    penalty = 0.0
    if {"Low", "High"}.issubset(pivot.columns):
        penalty += 0.0 if pivot.loc["Earlier", "Low"] > pivot.loc["Earlier", "High"] else 1.0
    if {"Low", "Mid"}.issubset(pivot.columns):
        penalty += 0.0 if pivot.loc["Last 30d", "Mid"] > pivot.loc["Last 30d", "Low"] else 1.0
    if {"Low", "High"}.issubset(pivot.columns):
        penalty += 0.0 if pivot.loc["Last 30d", "High"] > pivot.loc["Last 30d", "Low"] else 1.0
    return penalty / 3.0


def _activity_pattern_match(table: pd.DataFrame) -> bool:
    pivot = table["mean_abs_r"].unstack("activity_bin")
    if not {"Low", "High", "Mid"}.issubset(pivot.columns):
        return False
    return bool(
        pivot.loc["Earlier", "Low"] > pivot.loc["Earlier", "High"]
        and pivot.loc["Last 30d", "Mid"] > pivot.loc["Last 30d", "Low"]
        and pivot.loc["Last 30d", "High"] > pivot.loc["Last 30d", "Low"]
    )


def stage2_score(
    sim_metrics: dict[str, Any],
    emp_metrics: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Stage-2 distance: preserve stage-1 fit and add conditionals."""
    stage1_total, stage1_parts = stage1_score(sim_metrics, emp_metrics)

    interaction_cols = ["mean_abs_r", "std_r"]
    sim_inter = sim_metrics["interaction"][interaction_cols].to_numpy().ravel()
    emp_inter = emp_metrics["interaction"][interaction_cols].to_numpy().ravel()
    interaction_error = _relative_error(sim_inter, emp_inter)
    pattern_penalty = _interaction_pattern_penalty(sim_metrics["interaction"])

    sim_qr = sim_metrics["quantile_regression_std"].loc[
        list(QUANTILES_CALIBRATION), ["trade_rate_coef", "ttr_coef"]
    ]
    emp_qr = emp_metrics["quantile_regression_std"].loc[
        list(QUANTILES_CALIBRATION), ["trade_rate_coef", "ttr_coef"]
    ]
    qr_scale = np.maximum(np.ptp(emp_qr.to_numpy(), axis=0), 1e-6)
    qr_error = float(
        np.mean(np.abs(sim_qr.to_numpy() - emp_qr.to_numpy()) / qr_scale)
    )

    score = (
        0.55 * stage1_total
        + 0.25 * interaction_error
        + 0.10 * pattern_penalty
        + 0.10 * qr_error
    )
    return score, {
        **stage1_parts,
        "interaction_error": interaction_error,
        "interaction_pattern_penalty": pattern_penalty,
        "quantile_error": qr_error,
        "total_score": score,
    }


def _reversal_activity_gap(metrics: dict[str, Any]) -> tuple[float, float]:
    rev = metrics["reversal_activity"]
    if rev.empty:
        return 0.0, 0.0
    try:
        low = rev.xs(next(idx for idx in rev.index.levels[0] if idx.startswith("Low")), level=0)
        high = rev.xs(next(idx for idx in rev.index.levels[0] if idx.startswith("High")), level=0)
    except StopIteration:
        return 0.0, 0.0
    pos_gap = float(high.loc[6, "mean_after_pos"] - low.loc[6, "mean_after_pos"])
    neg_gap = float(low.loc[6, "mean_after_neg"] - high.loc[6, "mean_after_neg"])
    return pos_gap, neg_gap


def _low_activity_reversal_match(metrics: dict[str, Any]) -> bool:
    rev = metrics["reversal_activity"]
    if rev.empty:
        return False
    try:
        low = rev.xs(next(idx for idx in rev.index.levels[0] if idx.startswith("Low")), level=0)
        high = rev.xs(next(idx for idx in rev.index.levels[0] if idx.startswith("High")), level=0)
    except StopIteration:
        return False

    low_pos = float(low.loc[6, "mean_after_pos"])
    high_pos = float(high.loc[6, "mean_after_pos"])
    low_neg = float(low.loc[6, "mean_after_neg"])
    high_neg = float(high.loc[6, "mean_after_neg"])
    return bool(
        low_pos < 0.0
        and low_neg > 0.0
        and high_pos > low_pos
        and high_neg < low_neg
    )


def stage3_score(
    sim_metrics: dict[str, Any],
    emp_metrics: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Optional stage-3 distance: add low-activity reversal asymmetry."""
    stage2_total, stage2_parts = stage2_score(sim_metrics, emp_metrics)
    sim_gap = np.array(_reversal_activity_gap(sim_metrics))
    emp_gap = np.array(_reversal_activity_gap(emp_metrics))
    reversal_error = _relative_error(sim_gap, emp_gap, floor=0.01)
    score = 0.80 * stage2_total + 0.20 * reversal_error
    return score, {
        **stage2_parts,
        "reversal_error": reversal_error,
        "total_score": score,
    }


def stage_match_flags(
    stage: str,
    sim_metrics: dict[str, Any],
    emp_metrics: dict[str, Any],
) -> dict[str, bool]:
    """Human-readable stage-fit flags used in the markdown summaries."""
    if stage == "stage1":
        _, parts = stage1_score(sim_metrics, emp_metrics)
        sim_deadline = sim_metrics["deadline_summary"]
        late_ratio = (
            sim_deadline.loc["Last 30d", "mean_abs_r"]
            / max(sim_deadline.loc["Earlier", "mean_abs_r"], 1e-6)
        )
        return {
            "heavy_tails": parts["tail_error"] < 0.50,
            "volatility_clustering": parts["acf_error"] < 0.60,
            "late_life_volatility": parts["deadline_error"] < 0.50 and late_ratio > 1.2,
        }

    if stage == "stage2":
        _, parts = stage2_score(sim_metrics, emp_metrics)
        qr = sim_metrics["quantile_regression_std"]
        trade_progressive = bool(
            qr.loc[0.95, "trade_rate_coef"] > qr.loc[0.50, "trade_rate_coef"]
        )
        ttr_more_negative = bool(
            qr.loc[0.95, "ttr_coef"] < qr.loc[0.50, "ttr_coef"]
        )
        return {
            **stage_match_flags("stage1", sim_metrics, emp_metrics),
            "activity_deadline_pattern": _activity_pattern_match(sim_metrics["interaction"]),
            "quantile_pattern": trade_progressive and ttr_more_negative,
        }

    return {
        **stage_match_flags("stage2", sim_metrics, emp_metrics),
        "low_activity_reversal": _low_activity_reversal_match(sim_metrics),
    }


def needs_stage3(sim_metrics: dict[str, Any]) -> bool:
    """Decide whether stage 3 is justified after the stage-2 run."""
    return not _low_activity_reversal_match(sim_metrics)


def save_metric_tables(output_dir: Path, metrics: dict[str, Any]) -> None:
    """Persist metric tables for notebook/report reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics["return_stats"]]).to_csv(
        output_dir / "return_stats.csv", index=False
    )
    metrics["tail_quantiles"].to_csv(output_dir / "tail_quantiles.csv")
    metrics["acf_abs"].to_csv(output_dir / "acf_abs.csv")
    metrics["deadline_summary"].to_csv(output_dir / "deadline_summary.csv")
    metrics["interaction"].to_csv(output_dir / "interaction.csv")
    metrics["quantile_regression_raw"].to_csv(
        output_dir / "quantile_regression_raw.csv"
    )
    metrics["quantile_regression_std"].to_csv(
        output_dir / "quantile_regression_std.csv"
    )

    for name in ("reversal_activity", "reversal_deadline"):
        table = metrics[name]
        if table.empty:
            (output_dir / f"{name}.csv").write_text("", encoding="utf-8")
        else:
            table.to_csv(output_dir / f"{name}.csv")


def save_score_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Persist calibration search results."""
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "search_results.csv", index=False)


def save_params(output_dir: Path, params: dict[str, Any]) -> None:
    """Persist the best parameter vector for a stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
