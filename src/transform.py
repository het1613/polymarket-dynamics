"""
Data transformation: logit mapping, return computation, and feature merge.

Run with::

    python -m src.transform
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on sys.path when invoked directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import DATA_PROCESSED, ALL_MARKETS, TRIADS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLAMP_LO = 0.001
CLAMP_HI = 0.999


def logit(p: pd.Series) -> pd.Series:
    """Map bounded probability to unbounded logit scale."""
    p_clamped = p.clip(CLAMP_LO, CLAMP_HI)
    return np.log(p_clamped / (1.0 - p_clamped))


def build_features(market: dict) -> pd.DataFrame | None:
    """
    For one market, load price + activity + order-book data, compute
    logit returns and time-to-resolution, and merge into a single
    analysis-ready DataFrame.
    """
    slug = market["slug"]
    end_date_str = market["end_date"]
    role = market["role"]
    category = market["category"]
    triad_id = market.get("triad_id", "")

    price_path = DATA_PROCESSED / f"prices_{slug}_hourly.parquet"
    if not price_path.exists():
        logger.warning("Missing hourly prices for %s – skipping", slug)
        return None
    df = pd.read_parquet(price_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Logit transform & returns
    df["logit_price"] = logit(df["price"])
    df["logit_return"] = df["logit_price"].diff()
    df["abs_logit_return"] = df["logit_return"].abs()

    # Time-to-resolution (days)
    try:
        resolve_dt = pd.to_datetime(end_date_str, utc=True)
    except Exception:
        resolve_dt = pd.Timestamp(end_date_str, tz="UTC")

    df["time_to_resolution"] = (
        (resolve_dt - df["timestamp"]).dt.total_seconds() / 86400.0
    )

    # Merge activity proxies
    activity_path = DATA_PROCESSED / f"activity_{slug}.parquet"
    if activity_path.exists():
        act = pd.read_parquet(activity_path)
        act["timestamp"] = pd.to_datetime(act["timestamp"], utc=True)
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            act[["timestamp", "trade_rate_1h", "volume_rate_1h",
                 "trade_rate_24h", "volume_rate_24h"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )
    else:
        logger.warning("No activity data for %s – columns will be NaN", slug)
        for col in ("trade_rate_1h", "volume_rate_1h",
                     "trade_rate_24h", "volume_rate_24h"):
            df[col] = np.nan

    # Merge order-book snapshot (static)
    ob_path = DATA_PROCESSED / f"orderbook_{slug}.parquet"
    if ob_path.exists():
        ob = pd.read_parquet(ob_path)
        for col in ("spread", "midpoint", "depth_1c_bid", "depth_1c_ask",
                     "depth_5c_bid", "depth_5c_ask"):
            df[col] = ob[col].iloc[0] if col in ob.columns else np.nan
    else:
        logger.warning("No order-book data for %s – columns will be NaN", slug)

    # Metadata columns
    df["market_slug"] = slug
    df["role"] = role
    df["category"] = category
    df["triad_id"] = triad_id

    return df


def transform_all() -> None:
    """Build feature tables for every market, per-triad stacks, and a global stack."""
    if not ALL_MARKETS:
        logger.error("ALL_MARKETS is empty. Run: python -m src.collect.select_markets")
        sys.exit(1)

    # Per-market feature files
    for mkt in ALL_MARKETS:
        slug = mkt["slug"]
        out_path = DATA_PROCESSED / f"features_{slug}.parquet"

        logger.info("Transforming %s …", slug)
        df = build_features(mkt)
        if df is None or df.empty:
            logger.warning("No features produced for %s", slug)
            continue

        df.to_parquet(out_path, index=False)
        logger.info("Wrote %d rows × %d cols → %s", len(df), len(df.columns), out_path)

    # Per-triad stacked files
    for triad_id, markets in TRIADS.items():
        parts = []
        for mkt in markets:
            p = DATA_PROCESSED / f"features_{mkt['slug']}.parquet"
            if p.exists():
                parts.append(pd.read_parquet(p))
        if parts:
            stacked = pd.concat(parts, ignore_index=True)
            triad_path = DATA_PROCESSED / f"features_triad_{triad_id}.parquet"
            stacked.to_parquet(triad_path, index=False)
            logger.info("Wrote triad '%s' features (%d rows) → %s",
                        triad_id, len(stacked), triad_path)

    # Global stacked file
    all_parts = []
    for mkt in ALL_MARKETS:
        p = DATA_PROCESSED / f"features_{mkt['slug']}.parquet"
        if p.exists():
            all_parts.append(pd.read_parquet(p))
    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        combined_path = DATA_PROCESSED / "features_all.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info("Wrote combined features (%d rows) → %s",
                    len(combined), combined_path)


if __name__ == "__main__":
    transform_all()
