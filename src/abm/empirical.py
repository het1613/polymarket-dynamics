"""Load empirical politics-panel data and build ABM reference objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.abm.config import ABM_EMPIRICAL_DIR, MarketSkeleton, ensure_output_dirs
from src.abm.evaluation import compute_panel_metrics, save_metric_tables
from src.config import DATA_PROCESSED, POLITICS_MARKETS


def load_politics_panel(path: Path | None = None) -> pd.DataFrame:
    """Load the RQ3 politics feature panel."""
    if path is None:
        path = DATA_PROCESSED / "features_politics.parquet"

    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["market_slug", "timestamp"]).reset_index(drop=True)


def build_market_skeletons(df: pd.DataFrame) -> list[MarketSkeleton]:
    """Mirror the empirical market lengths, timestamps, and TTR paths."""
    skeletons: list[MarketSkeleton] = []

    for market in POLITICS_MARKETS:
        slug = market["slug"]
        mdf = (
            df.loc[df["market_slug"] == slug]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if mdf.empty:
            continue

        skeletons.append(
            MarketSkeleton(
                market_slug=slug,
                market_type=str(market.get("market_type", "")),
                timestamps=tuple(mdf["timestamp"].tolist()),
                time_to_resolution=tuple(mdf["time_to_resolution"].astype(float).tolist()),
                initial_price=float(mdf["price"].iloc[0]),
                initial_logit=float(mdf["logit_price"].iloc[0]),
            )
        )

    return skeletons


def load_empirical_reference() -> dict[str, Any]:
    """Load empirical panel, market skeletons, and benchmark metrics."""
    df = load_politics_panel()
    skeletons = build_market_skeletons(df)
    metrics = compute_panel_metrics(df, sample_n_for_quantile=None)
    return {"panel": df, "skeletons": skeletons, "metrics": metrics}


def write_empirical_reference(reference: dict[str, Any]) -> None:
    """Persist empirical benchmark tables for the RQ4 notebook."""
    ensure_output_dirs()
    metrics = reference["metrics"]
    save_metric_tables(ABM_EMPIRICAL_DIR, metrics)

    summary = {
        "n_rows": int(len(reference["panel"])),
        "n_markets": int(reference["panel"]["market_slug"].nunique()),
        "market_slugs": sorted(reference["panel"]["market_slug"].unique().tolist()),
    }
    with (ABM_EMPIRICAL_DIR / "reference.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
