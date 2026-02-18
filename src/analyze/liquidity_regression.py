"""
Liquidity and activity regressions.

Bin stratification by trade-rate and spread, OLS regressions with
time-to-resolution controls, and deadline-effect visualisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf


# ── Bin stratification ───────────────────────────────────────────────────────

def _acf1(series: pd.Series) -> float:
    """ACF at lag 1 for a series, NaN-safe."""
    clean = series.dropna()
    if len(clean) < 10:
        return np.nan
    return float(acf(clean, nlags=1, fft=True)[1])


def bin_stratification(
    df: pd.DataFrame,
    bin_col: str,
    n_bins: int = 3,
    return_col: str = "abs_logit_return",
) -> pd.DataFrame:
    """
    Partition observations into quantile bins of *bin_col* and report
    within-bin statistics for |r_t|.

    Returns a DataFrame with columns: ``bin``, ``mean_abs_r``,
    ``kurtosis``, ``acf1_abs_r``, ``n``.
    """
    clean = df[[bin_col, return_col, "logit_return"]].dropna()
    if clean.empty:
        return pd.DataFrame()

    clean["bin"] = pd.qcut(clean[bin_col], n_bins, labels=False, duplicates="drop")

    rows = []
    for b, grp in clean.groupby("bin"):
        rows.append(
            {
                "bin": int(b),
                f"{bin_col}_lo": grp[bin_col].min(),
                f"{bin_col}_hi": grp[bin_col].max(),
                "mean_abs_r": grp[return_col].mean(),
                "kurtosis": float(grp["logit_return"].kurtosis()),
                "acf1_abs_r": _acf1(grp[return_col]),
                "n": len(grp),
            }
        )
    return pd.DataFrame(rows)


# ── OLS regressions ──────────────────────────────────────────────────────────

def ols_volatility_activity(
    df: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    ``|r_t| ~ trade_rate_1h + time_to_resolution``

    Uses the stacked ``features_all.parquet`` with market dummies.
    """
    cols = ["abs_logit_return", "trade_rate_1h", "time_to_resolution"]
    if "role" in df.columns:
        cols.append("role")

    clean = df[cols].dropna()
    if clean.empty:
        raise ValueError("No valid rows for OLS")

    y = clean["abs_logit_return"]
    X = clean[["trade_rate_1h", "time_to_resolution"]].copy()

    if "role" in clean.columns:
        dummies = pd.get_dummies(clean["role"], drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)
    return sm.OLS(y, X).fit(cov_type="HC1")


def ols_volatility_spread(
    df: pd.DataFrame,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    ``|r_t| ~ spread + time_to_resolution``

    Spread is a static per-market value, so this mainly tests cross-market
    variation.  Market dummies are therefore omitted (collinear with spread).
    """
    cols = ["abs_logit_return", "spread", "time_to_resolution"]
    clean = df[cols].dropna()
    if clean.empty:
        raise ValueError("No valid rows for OLS")

    y = clean["abs_logit_return"]
    X = sm.add_constant(clean[["spread", "time_to_resolution"]])
    return sm.OLS(y, X).fit(cov_type="HC1")


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_bin_stratification(
    feature_dfs: dict[str, pd.DataFrame],
    bin_col: str = "trade_rate_1h",
    n_bins: int = 3,
) -> plt.Figure:
    """Bar charts of mean |r| by trade-rate tercile, one panel per market."""
    slugs = list(feature_dfs.keys())
    fig, axes = plt.subplots(1, len(slugs), figsize=(5 * len(slugs), 4), sharey=True)
    if len(slugs) == 1:
        axes = [axes]

    for ax, slug in zip(axes, slugs):
        bs = bin_stratification(feature_dfs[slug], bin_col, n_bins)
        if bs.empty:
            ax.set_title(f"{slug} (no data)")
            continue
        ax.bar(bs["bin"], bs["mean_abs_r"], alpha=0.7)
        ax.set_xticks(bs["bin"])
        ax.set_xticklabels([f"T{int(b)+1}" for b in bs["bin"]])
        ax.set_xlabel(f"{bin_col} tercile")
        ax.set_title(slug, fontsize=10)

    axes[0].set_ylabel("Mean |logit return|")
    fig.suptitle(f"Volatility by {bin_col} Tercile", fontsize=12)
    fig.tight_layout()
    return fig


def plot_deadline_effect(
    feature_dfs: dict[str, pd.DataFrame],
) -> plt.Figure:
    """
    Scatter of |r_t| vs. time-to-resolution with a LOWESS smoother,
    one panel per market.
    """
    slugs = list(feature_dfs.keys())
    fig, axes = plt.subplots(1, len(slugs), figsize=(5 * len(slugs), 4), sharey=True)
    if len(slugs) == 1:
        axes = [axes]

    for ax, slug in zip(axes, slugs):
        df = feature_dfs[slug].dropna(
            subset=["abs_logit_return", "time_to_resolution"]
        )
        if df.empty:
            ax.set_title(f"{slug} (no data)")
            continue

        ax.scatter(
            df["time_to_resolution"],
            df["abs_logit_return"],
            s=2,
            alpha=0.3,
        )

        # LOWESS smoother
        lw = lowess(
            df["abs_logit_return"].values,
            df["time_to_resolution"].values,
            frac=0.15,
            return_sorted=True,
        )
        ax.plot(lw[:, 0], lw[:, 1], "r-", linewidth=2)

        ax.set_xlabel("Days to resolution")
        ax.set_title(slug, fontsize=10)

    axes[0].set_ylabel("|Logit return|")
    fig.suptitle("Deadline Effect: Volatility vs. Time-to-Resolution", fontsize=12)
    fig.tight_layout()
    return fig
