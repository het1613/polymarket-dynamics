"""Plot writers for staged RQ4 outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.abm.config import ABM_PLOT_PREFIX, PLOTS_DIR, REPRESENTATIVE_PLOT_MARKETS


def _plot_path(stage: str, suffix: str) -> Path:
    return PLOTS_DIR / f"{ABM_PLOT_PREFIX}_{stage}_{suffix}.png"


def plot_tail_quantiles(
    empirical: pd.DataFrame,
    simulated: pd.DataFrame,
    stage: str,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    periods = ["Overall", "Earlier", "Last 30d"]
    quantiles = ["q90", "q95", "q99"]

    for ax, period in zip(axes, periods):
        emp_vals = empirical.loc[period, quantiles].to_numpy()
        sim_vals = simulated.loc[period, quantiles].to_numpy()
        x = np.arange(len(quantiles))
        ax.bar(x - 0.18, emp_vals, width=0.36, label="Empirical", alpha=0.8)
        ax.bar(x + 0.18, sim_vals, width=0.36, label="Simulated", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(quantiles)
        ax.set_title(period)

    axes[0].set_ylabel("|logit return| quantile")
    axes[-1].legend()
    fig.suptitle(f"Stage {stage[-1]} Tail Quantiles")
    fig.tight_layout()
    out = _plot_path(stage, "tail_quantiles")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_acf_comparison(
    empirical: pd.DataFrame,
    simulated: pd.DataFrame,
    stage: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(empirical.index, empirical["acf"], label="Empirical", linewidth=2)
    ax.plot(simulated.index, simulated["acf"], label="Simulated", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlim(1, simulated.index.max())
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF of |r|")
    ax.set_title(f"Stage {stage[-1]} ACF Comparison")
    ax.legend()
    fig.tight_layout()
    out = _plot_path(stage, "acf")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_price_paths(panel: pd.DataFrame, stage: str) -> Path:
    market_order = [
        slug for slug in REPRESENTATIVE_PLOT_MARKETS if slug in set(panel["market_slug"])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    for ax, slug in zip(axes.flatten(), market_order):
        mdf = panel.loc[panel["market_slug"] == slug].sort_values("timestamp")
        ax.plot(mdf["timestamp"], mdf["price"], linewidth=1.0)
        ax.set_title(slug.replace("will-", "").replace("-", " ")[:42])
        ax.set_ylim(0, 1)
    fig.suptitle(f"Stage {stage[-1]} Representative Price Paths")
    fig.tight_layout()
    out = _plot_path(stage, "price_paths")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_trade_rate_timeseries(panel: pd.DataFrame, stage: str) -> Path:
    market_order = [
        slug for slug in REPRESENTATIVE_PLOT_MARKETS if slug in set(panel["market_slug"])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    for ax, slug in zip(axes.flatten(), market_order):
        mdf = panel.loc[panel["market_slug"] == slug].sort_values("timestamp")
        ax.plot(mdf["timestamp"], mdf["trade_rate_1h"], linewidth=0.9)
        ax.set_title(slug.replace("will-", "").replace("-", " ")[:42])
        ax.set_ylabel("trade_rate_1h")
    fig.suptitle(f"Stage {stage[-1]} Activity Time Series")
    fig.tight_layout()
    out = _plot_path(stage, "trade_rate")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_activity_heatmap(
    empirical: pd.DataFrame,
    simulated: pd.DataFrame,
    stage: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, table, title in zip(
        axes,
        (empirical, simulated),
        ("Empirical", "Simulated"),
    ):
        pivot = table["mean_abs_r"].unstack("activity_bin")
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes, fraction=0.045, pad=0.04, label="Mean |r|")
    fig.suptitle(f"Stage {stage[-1]} Activity x Deadline Heatmap")
    fig.tight_layout()
    out = _plot_path(stage, "activity_heatmap")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_quantile_regression(
    empirical: pd.DataFrame,
    simulated: pd.DataFrame,
    stage: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, title in zip(
        axes,
        ("trade_rate_coef", "ttr_coef"),
        ("Trade-rate coefficient", "Time-to-resolution coefficient"),
    ):
        ax.plot(empirical.index, empirical[column], marker="o", label="Empirical")
        ax.plot(simulated.index, simulated[column], marker="s", label="Simulated")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("Quantile")
    axes[0].set_ylabel("Coefficient")
    axes[0].legend()
    fig.suptitle(f"Stage {stage[-1]} Quantile Regression")
    fig.tight_layout()
    out = _plot_path(stage, "quantile_regression")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_reversal_comparison(
    empirical: pd.DataFrame,
    simulated: pd.DataFrame,
    stage: str,
    suffix: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, table, title in zip(
        axes,
        (empirical, simulated),
        ("Empirical", "Simulated"),
    ):
        if table.empty:
            ax.set_title(f"{title} (no data)")
            continue
        for condition in table.index.get_level_values(0).unique():
            subset = table.xs(condition, level=0)
            ax.plot(subset.index, subset["mean_after_pos"], marker="o", label=f"{condition} after +")
            ax.plot(subset.index, subset["mean_after_neg"], marker="s", linestyle="--", label=f"{condition} after -")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("Horizon (hours)")
    axes[0].set_ylabel("Mean conditional return")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Stage {stage[-1]} Reversal Comparison")
    fig.tight_layout()
    out = _plot_path(stage, suffix)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out
