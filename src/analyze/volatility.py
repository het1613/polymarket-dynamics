"""
Volatility-clustering analysis.

ACF of absolute returns, Ljung-Box tests, rolling-volatility time series,
and burst detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox


# ── ACF computation ──────────────────────────────────────────────────────────

def abs_return_acf(
    df: pd.DataFrame,
    col: str = "abs_logit_return",
    nlags: int = 100,
) -> pd.DataFrame:
    """
    Compute the ACF of |r_t| with confidence bounds.

    Returns a DataFrame with columns ``lag``, ``acf``, ``ci_lower``,
    ``ci_upper``.
    """
    series = df[col].dropna().values
    acf_vals, ci = acf(series, nlags=nlags, alpha=0.05)
    out = pd.DataFrame(
        {
            "lag": np.arange(nlags + 1),
            "acf": acf_vals,
            "ci_lower": ci[:, 0] - acf_vals,
            "ci_upper": ci[:, 1] - acf_vals,
        }
    )
    return out


# ── Ljung-Box test ───────────────────────────────────────────────────────────

def ljung_box_test(
    df: pd.DataFrame,
    col: str = "abs_logit_return",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Run the Ljung-Box portmanteau test on |r_t| at specified lags.

    Returns a DataFrame with ``lag``, ``lb_stat``, ``lb_pvalue``.
    """
    if lags is None:
        lags = [10, 20, 50]
    series = df[col].dropna()
    result = acorr_ljungbox(series, lags=lags, return_df=True)
    result.index.name = "lag"
    return result.reset_index()


# ── Rolling volatility ───────────────────────────────────────────────────────

def rolling_volatility(
    df: pd.DataFrame,
    col: str = "logit_return",
    window: int = 24,
) -> pd.Series:
    """Rolling standard deviation of logit returns (default 24-hour window)."""
    return df[col].rolling(window, min_periods=1).std()


# ── Burst detection ──────────────────────────────────────────────────────────

def detect_bursts(
    df: pd.DataFrame,
    col: str = "logit_return",
    window: int = 24,
    sigma_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Flag contiguous periods where rolling volatility exceeds
    ``mean + sigma_mult * std`` of the rolling-vol series.

    Returns a DataFrame with one row per burst episode:
    ``start``, ``end``, ``duration_hours``, ``mean_vol``.
    """
    vol = rolling_volatility(df, col=col, window=window)
    threshold = vol.mean() + sigma_mult * vol.std()
    is_burst = vol > threshold

    # Label contiguous burst blocks
    block_id = (is_burst != is_burst.shift()).cumsum()
    burst_blocks = block_id[is_burst]

    if burst_blocks.empty:
        return pd.DataFrame(
            columns=["start", "end", "duration_hours", "mean_vol"]
        )

    episodes = []
    for _, grp in df.loc[burst_blocks.index].groupby(burst_blocks):
        ts = grp["timestamp"]
        v = vol.loc[grp.index]
        episodes.append(
            {
                "start": ts.iloc[0],
                "end": ts.iloc[-1],
                "duration_hours": len(grp),
                "mean_vol": v.mean(),
            }
        )
    return pd.DataFrame(episodes)


def burst_summary(bursts: pd.DataFrame) -> dict[str, float]:
    """Quick summary stats from a burst DataFrame."""
    if bursts.empty:
        return {"n_bursts": 0, "mean_duration_h": 0, "mean_intensity": 0}
    return {
        "n_bursts": len(bursts),
        "mean_duration_h": bursts["duration_hours"].mean(),
        "mean_intensity": bursts["mean_vol"].mean(),
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_acf_panels(
    feature_dfs: dict[str, pd.DataFrame],
    col: str = "abs_logit_return",
    nlags: int = 100,
) -> plt.Figure:
    """Three-panel ACF plot of |r_t|."""
    slugs = list(feature_dfs.keys())
    fig, axes = plt.subplots(1, len(slugs), figsize=(5 * len(slugs), 4), sharey=True)
    if len(slugs) == 1:
        axes = [axes]

    for ax, slug in zip(axes, slugs):
        acf_df = abs_return_acf(feature_dfs[slug], col=col, nlags=nlags)
        ax.bar(acf_df["lag"], acf_df["acf"], width=1, alpha=0.6)
        ax.fill_between(
            acf_df["lag"],
            acf_df["ci_lower"],
            acf_df["ci_upper"],
            alpha=0.15,
            color="grey",
        )
        ax.set_title(slug, fontsize=10)
        ax.set_xlabel("Lag (hours)")

    axes[0].set_ylabel("ACF of |r|")
    fig.suptitle("Volatility Clustering: ACF of |Logit Returns|", fontsize=12)
    fig.tight_layout()
    return fig


def plot_acf_overlay(
    feature_dfs: dict[str, pd.DataFrame],
    col: str = "abs_logit_return",
    nlags: int = 100,
) -> plt.Figure:
    """All triad ACF curves on a single axis for direct comparison."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for slug, df in feature_dfs.items():
        acf_df = abs_return_acf(df, col=col, nlags=nlags)
        ax.plot(acf_df["lag"], acf_df["acf"], label=slug, alpha=0.8)

    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF of |r|")
    ax.set_title("Cross-Triad ACF Comparison")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    return fig


def plot_rolling_vol_with_price(
    df: pd.DataFrame,
    slug: str,
    window: int = 24,
) -> plt.Figure:
    """
    Two-panel figure: top = price, bottom = rolling volatility with
    burst threshold.
    """
    vol = rolling_volatility(df, window=window)
    threshold = vol.mean() + 2.0 * vol.std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    ax1.plot(df["timestamp"], df["price"], linewidth=0.7)
    ax1.set_ylabel("Price (prob)")
    ax1.set_title(f"{slug} — Price & Rolling Volatility ({window}h)")

    ax2.plot(df["timestamp"], vol, linewidth=0.7)
    ax2.axhline(threshold, color="r", linestyle="--", linewidth=0.8, label="Burst threshold")
    ax2.set_ylabel("Rolling σ (logit)")
    ax2.set_xlabel("Time")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    return fig
