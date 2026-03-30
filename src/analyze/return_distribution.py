"""
Return-distribution analysis.

Heavy-tail diagnostics, QQ plots, power-law fitting, and cross-triad
comparison for logit-space returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    import powerlaw
except ImportError:
    powerlaw = None  # type: ignore[assignment]


# ── Descriptive statistics ───────────────────────────────────────────────────

def return_stats(df: pd.DataFrame, col: str = "logit_return") -> dict[str, float]:
    """Compute summary statistics for one market's return series."""
    r = df[col].dropna()
    return {
        "n": len(r),
        "mean": r.mean(),
        "std": r.std(),
        "skewness": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r)),  # excess kurtosis
        "min": r.min(),
        "max": r.max(),
        "jarque_bera_stat": float(stats.jarque_bera(r).statistic),
        "jarque_bera_p": float(stats.jarque_bera(r).pvalue),
    }


def cross_triad_stats(
    feature_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Return a summary DataFrame with one row per market slug.

    Parameters
    ----------
    feature_dfs : ``{slug: features_DataFrame}``
    """
    rows = []
    for slug, df in feature_dfs.items():
        row = return_stats(df)
        row["market_slug"] = slug
        row["role"] = df["role"].iloc[0] if "role" in df.columns else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("market_slug")


# ── Power-law tail fitting ───────────────────────────────────────────────────

def fit_tails(
    df: pd.DataFrame, col: str = "logit_return"
) -> dict[str, Any]:
    """
    Fit power-law, lognormal, and exponential to |r| and return
    fit objects + log-likelihood comparisons.

    Requires the ``powerlaw`` package.
    """
    if powerlaw is None:
        raise ImportError("Install the `powerlaw` package.")

    abs_r = df[col].dropna().abs().values
    abs_r = abs_r[abs_r > 0]

    fit = powerlaw.Fit(abs_r, discrete=False, verbose=False)

    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential")

    return {
        "alpha": fit.alpha,
        "xmin": fit.xmin,
        "sigma": fit.sigma,
        "R_vs_lognormal": R_ln,
        "p_vs_lognormal": p_ln,
        "R_vs_exponential": R_exp,
        "p_vs_exponential": p_exp,
        "fit_object": fit,
    }


def cross_triad_tail_table(
    feature_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """One-row-per-market tail-fit comparison table."""
    if powerlaw is None:
        raise ImportError("Install the `powerlaw` package.")

    rows = []
    for slug, df in feature_dfs.items():
        info = fit_tails(df)
        rows.append(
            {
                "market_slug": slug,
                "alpha": info["alpha"],
                "xmin": info["xmin"],
                "R_vs_lognormal": info["R_vs_lognormal"],
                "p_vs_lognormal": info["p_vs_lognormal"],
                "R_vs_exponential": info["R_vs_exponential"],
                "p_vs_exponential": info["p_vs_exponential"],
            }
        )
    return pd.DataFrame(rows).set_index("market_slug")


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_return_histograms(
    feature_dfs: dict[str, pd.DataFrame],
    col: str = "logit_return",
    bins: int = 80,
) -> plt.Figure:
    """
    Three-panel histogram of logit returns with a normal overlay.
    """
    slugs = list(feature_dfs.keys())
    fig, axes = plt.subplots(1, len(slugs), figsize=(5 * len(slugs), 4), sharey=True)
    if len(slugs) == 1:
        axes = [axes]

    for ax, slug in zip(axes, slugs):
        r = feature_dfs[slug][col].dropna()
        ax.hist(r, bins=bins, density=True, alpha=0.6, edgecolor="none")

        # Normal overlay
        x = np.linspace(r.min(), r.max(), 300)
        ax.plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r-", lw=1.5)

        ax.set_title(slug, fontsize=10)
        ax.set_xlabel("Logit return")

    axes[0].set_ylabel("Density")
    fig.suptitle("Logit-Return Distributions", fontsize=12)
    fig.tight_layout()
    return fig


def plot_qq(
    feature_dfs: dict[str, pd.DataFrame],
    col: str = "logit_return",
) -> plt.Figure:
    """QQ plots against a normal distribution, one panel per market."""
    slugs = list(feature_dfs.keys())
    fig, axes = plt.subplots(1, len(slugs), figsize=(5 * len(slugs), 4))
    if len(slugs) == 1:
        axes = [axes]

    for ax, slug in zip(axes, slugs):
        r = feature_dfs[slug][col].dropna().values
        stats.probplot(r, dist="norm", plot=ax)
        ax.set_title(slug, fontsize=10)

    fig.suptitle("QQ Plots vs. Normal", fontsize=12)
    fig.tight_layout()
    return fig


def plot_tail_ccdf(
    feature_dfs: dict[str, pd.DataFrame],
    col: str = "logit_return",
) -> plt.Figure:
    """
    Log-log complementary CDF of |r| for each market on a single axes,
    useful for visual power-law comparison.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for slug, df in feature_dfs.items():
        abs_r = np.sort(df[col].dropna().abs().values)[::-1]
        ccdf = np.arange(1, len(abs_r) + 1) / len(abs_r)
        ax.loglog(abs_r, ccdf, ".", markersize=2, label=slug)

    ax.set_xlabel("|Logit return|")
    ax.set_ylabel("P(X > x)")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize="small",
    )
    ax.set_title("Complementary CDF of |Logit Returns|")
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    return fig
