"""
Reversal / overshoot analysis.

Event-study of conditional returns following large positive and large
negative moves in logit space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats


HORIZONS = [1, 3, 6, 12, 24]


def _classify_events(
    returns: pd.Series, sigma_mult: float = 2.0
) -> pd.Series:
    """
    Label each observation as ``"large_pos"``, ``"large_neg"``, or ``None``.

    A large move is one where ``|r_t| > sigma_mult * sigma``.
    """
    sigma = returns.std()
    threshold = sigma_mult * sigma
    labels = pd.Series(None, index=returns.index, dtype=object)
    labels[returns > threshold] = "large_pos"
    labels[returns < -threshold] = "large_neg"
    return labels


def conditional_returns(
    df: pd.DataFrame,
    col: str = "logit_return",
    sigma_mult: float = 2.0,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute mean conditional returns at each horizon after large moves.

    Returns a DataFrame indexed by horizon with columns:

    - ``mean_after_pos``, ``se_after_pos``, ``n_pos``
    - ``mean_after_neg``, ``se_after_neg``, ``n_neg``
    - ``tstat_pos``, ``pval_pos`` (H0: mean = 0)
    - ``tstat_neg``, ``pval_neg``
    """
    if horizons is None:
        horizons = HORIZONS

    r = df[col].dropna().reset_index(drop=True)
    events = _classify_events(r, sigma_mult)

    rows = []
    for h in horizons:
        # Forward return over h steps
        fwd = r.shift(-h) - r  # cumulative return over next h periods
        # We want the cumulative logit return, but since logit returns
        # are additive, the sum of the next h single-step returns equals
        # the h-step return.  Using rolling sum is equivalent:
        fwd_h = r.rolling(h).sum().shift(-h)

        pos_idx = events[events == "large_pos"].index
        neg_idx = events[events == "large_neg"].index

        fwd_pos = fwd_h.loc[fwd_h.index.intersection(pos_idx)].dropna()
        fwd_neg = fwd_h.loc[fwd_h.index.intersection(neg_idx)].dropna()

        def _ttest(vals: pd.Series):
            if len(vals) < 2:
                return np.nan, np.nan
            t, p = sp_stats.ttest_1samp(vals, 0)
            return float(t), float(p)

        t_pos, p_pos = _ttest(fwd_pos)
        t_neg, p_neg = _ttest(fwd_neg)

        rows.append(
            {
                "horizon": h,
                "mean_after_pos": fwd_pos.mean() if len(fwd_pos) else np.nan,
                "se_after_pos": fwd_pos.sem() if len(fwd_pos) else np.nan,
                "n_pos": len(fwd_pos),
                "tstat_pos": t_pos,
                "pval_pos": p_pos,
                "mean_after_neg": fwd_neg.mean() if len(fwd_neg) else np.nan,
                "se_after_neg": fwd_neg.sem() if len(fwd_neg) else np.nan,
                "n_neg": len(fwd_neg),
                "tstat_neg": t_neg,
                "pval_neg": p_neg,
            }
        )
    return pd.DataFrame(rows).set_index("horizon")


def cross_triad_reversal_table(
    feature_dfs: dict[str, pd.DataFrame],
    sigma_mult: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Return ``{slug: conditional_returns_df}`` for every market."""
    return {
        slug: conditional_returns(df, sigma_mult=sigma_mult)
        for slug, df in feature_dfs.items()
    }


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_conditional_returns(
    feature_dfs: dict[str, pd.DataFrame],
    sigma_mult: float = 2.0,
) -> plt.Figure:
    """
    Event-study style plot: mean conditional return path after large
    positive and large negative moves, one panel per market.
    """
    slugs = list(feature_dfs.keys())
    fig, axes = plt.subplots(1, len(slugs), figsize=(5 * len(slugs), 4), sharey=True)
    if len(slugs) == 1:
        axes = [axes]

    for ax, slug in zip(axes, slugs):
        cr = conditional_returns(feature_dfs[slug], sigma_mult=sigma_mult)

        ax.errorbar(
            cr.index,
            cr["mean_after_pos"],
            yerr=1.96 * cr["se_after_pos"],
            marker="o",
            capsize=3,
            label="After large +",
        )
        ax.errorbar(
            cr.index,
            cr["mean_after_neg"],
            yerr=1.96 * cr["se_after_neg"],
            marker="s",
            capsize=3,
            label="After large −",
        )
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_title(slug, fontsize=10)
        ax.set_xlabel("Horizon (hours)")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Mean conditional return (logit)")
    fig.suptitle(
        f"Reversal Analysis (|r| > {sigma_mult}σ events)", fontsize=12
    )
    fig.tight_layout()
    return fig
