"""
Cross-triad consistency analysis.

Aggregates per-market diagnostics into within-triad summaries, checks
consistency across the three triads, produces comparison plots, and
builds a Stage 2 readiness decision table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from src.analyze.return_distribution import return_stats, fit_tails
from src.analyze.volatility import (
    ljung_box_test,
    burst_summary,
    detect_bursts,
    abs_return_acf,
)
from src.analyze.reversal import conditional_returns


# ── Per-market diagnostic row ────────────────────────────────────────────────

def _safe_fit_tails(df: pd.DataFrame) -> dict:
    try:
        info = fit_tails(df)
        return {"alpha": info["alpha"], "xmin": info["xmin"]}
    except Exception:
        return {"alpha": np.nan, "xmin": np.nan}


def market_diagnostic(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a full diagnostic row for one market's feature DataFrame."""
    slug = df["market_slug"].iloc[0] if "market_slug" in df.columns else "?"
    triad = df["triad_id"].iloc[0] if "triad_id" in df.columns else "?"
    role = df["role"].iloc[0] if "role" in df.columns else "?"

    rs = return_stats(df)

    # Ljung-Box at lag 10, 20
    try:
        lb = ljung_box_test(df, lags=[10, 20])
        lb10_p = lb.loc[lb["lag"] == 10, "lb_pvalue"].values
        lb20_p = lb.loc[lb["lag"] == 20, "lb_pvalue"].values
        lb10_pval = float(lb10_p[0]) if len(lb10_p) else np.nan
        lb20_pval = float(lb20_p[0]) if len(lb20_p) else np.nan
    except Exception:
        lb10_pval = lb20_pval = np.nan

    # Bursts
    try:
        bursts = detect_bursts(df)
        bs = burst_summary(bursts)
    except Exception:
        bs = {"n_bursts": 0, "mean_duration_h": 0, "mean_intensity": 0}

    # Power-law tail
    tail = _safe_fit_tails(df)

    # Reversal: sign of mean conditional return at h=6 after large + and -
    try:
        cr = conditional_returns(df, sigma_mult=2.0, horizons=[6])
        rev_pos = cr["mean_after_pos"].iloc[0] if len(cr) else np.nan
        rev_neg = cr["mean_after_neg"].iloc[0] if len(cr) else np.nan
    except Exception:
        rev_pos = rev_neg = np.nan

    return {
        "market_slug": slug,
        "triad_id": triad,
        "role": role,
        "n_obs": rs["n"],
        "mean_return": rs["mean"],
        "std_return": rs["std"],
        "skewness": rs["skewness"],
        "excess_kurtosis": rs["kurtosis"],
        "jarque_bera_p": rs["jarque_bera_p"],
        "ljung_box_10_p": lb10_pval,
        "ljung_box_20_p": lb20_pval,
        "n_bursts": bs["n_bursts"],
        "mean_burst_duration_h": bs["mean_duration_h"],
        "tail_alpha": tail["alpha"],
        "tail_xmin": tail["xmin"],
        "reversal_after_pos_h6": rev_pos,
        "reversal_after_neg_h6": rev_neg,
    }


# ── Within-triad summary ────────────────────────────────────────────────────

def within_triad_summary(
    feature_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute diagnostic rows for every market in a single triad.

    Parameters
    ----------
    feature_dfs : ``{slug: features_df}`` for the 3 markets of one triad.
    """
    rows = [market_diagnostic(df) for df in feature_dfs.values()]
    return pd.DataFrame(rows).set_index("market_slug")


# ── Cross-triad consistency ──────────────────────────────────────────────────

def cross_triad_consistency(
    triad_summaries: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate within-triad summaries into a cross-triad consistency report.

    For each numerical diagnostic column, reports the per-triad mean and the
    overall coefficient of variation (CV) across triads.  Low CV → consistent.

    Parameters
    ----------
    triad_summaries : ``{triad_id: within_triad_summary_df}``
    """
    means: dict[str, dict] = {}
    for tid, tdf in triad_summaries.items():
        numeric = tdf.select_dtypes(include="number")
        means[tid] = numeric.mean().to_dict()

    agg = pd.DataFrame(means).T
    agg.index.name = "triad_id"

    summary_rows = []
    for col in agg.columns:
        vals = agg[col].dropna()
        if len(vals) < 2:
            continue
        m = vals.mean()
        s = vals.std()
        cv = s / abs(m) if abs(m) > 1e-12 else np.nan
        summary_rows.append({
            "metric": col,
            "overall_mean": m,
            "overall_std": s,
            "cv": cv,
            **{f"triad_{tid}": vals.get(tid, np.nan) for tid in triad_summaries},
        })

    return pd.DataFrame(summary_rows).set_index("metric")


def _bool_flag(value: float, threshold: float, direction: str = "below") -> str:
    if np.isnan(value):
        return "?"
    if direction == "below":
        return "YES" if value < threshold else "no"
    return "YES" if value > threshold else "no"


def stage2_decision_table(
    consistency: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a Stage 2 readiness table: for each key diagnostic, flag
    whether the cross-triad evidence is consistent enough to proceed.

    Checks:
    - Heavy tails: excess kurtosis consistently > 0 across triads
    - Volatility clustering: Ljung-Box p < 0.05 across triads
    - Reversal evidence: negative mean after large + (sign consistency)
    """
    checks: list[dict] = []

    def _get(metric: str, col: str = "overall_mean") -> float:
        if metric in consistency.index:
            return consistency.loc[metric, col]
        return np.nan

    # 1. Heavy tails
    ek = _get("excess_kurtosis")
    checks.append({
        "check": "Heavy tails (excess kurtosis > 0)",
        "overall_mean": ek,
        "cv": _get("excess_kurtosis", "cv"),
        "pass": _bool_flag(ek, 0, "above"),
    })

    # 2. Non-normality
    jb = _get("jarque_bera_p")
    checks.append({
        "check": "Non-normality (Jarque-Bera p < 0.05)",
        "overall_mean": jb,
        "cv": _get("jarque_bera_p", "cv"),
        "pass": _bool_flag(jb, 0.05, "below"),
    })

    # 3. Volatility clustering
    lb10 = _get("ljung_box_10_p")
    checks.append({
        "check": "Vol clustering (LB-10 p < 0.05)",
        "overall_mean": lb10,
        "cv": _get("ljung_box_10_p", "cv"),
        "pass": _bool_flag(lb10, 0.05, "below"),
    })

    # 4. Reversal after large positive
    rev = _get("reversal_after_pos_h6")
    checks.append({
        "check": "Reversal after large + (mean < 0 at h=6)",
        "overall_mean": rev,
        "cv": _get("reversal_after_pos_h6", "cv"),
        "pass": _bool_flag(rev, 0, "below"),
    })

    return pd.DataFrame(checks)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_signature_comparison(
    triad_summaries: dict[str, pd.DataFrame],
    metrics: list[str] | None = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing key metrics across triads.

    Each cluster is one metric; bars are coloured by triad.
    """
    if metrics is None:
        metrics = [
            "excess_kurtosis",
            "n_bursts",
            "tail_alpha",
            "reversal_after_pos_h6",
        ]

    n_metrics = len(metrics)
    n_triads = len(triad_summaries)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_triads, 1)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_metrics), 5))

    for i, (tid, tdf) in enumerate(triad_summaries.items()):
        means = tdf.select_dtypes(include="number").mean()
        vals = [means.get(m, np.nan) for m in metrics]
        offset = (i - (n_triads - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=tid, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.legend(title="Triad")
    ax.set_title("Cross-Triad Diagnostic Comparison")
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    return fig


def plot_kurtosis_comparison(
    triad_summaries: dict[str, pd.DataFrame],
) -> plt.Figure:
    """Horizontal bar chart of excess kurtosis, colour-coded by triad."""
    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * sum(len(t) for t in triad_summaries.values()))))

    y_labels, y_vals, colors = [], [], []
    palette = plt.cm.Set2.colors  # type: ignore[attr-defined]

    for i, (tid, tdf) in enumerate(triad_summaries.items()):
        for slug in tdf.index:
            y_labels.append(f"[{tid}] {slug[:35]}")
            y_vals.append(tdf.loc[slug, "excess_kurtosis"])
            colors.append(palette[i % len(palette)])

    y_pos = np.arange(len(y_labels))
    ax.barh(y_pos, y_vals, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Excess Kurtosis")
    ax.set_title("Excess Kurtosis by Market (grouped by triad)")
    fig.tight_layout()
    return fig
