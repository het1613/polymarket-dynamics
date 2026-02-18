"""
Data transformation: logit mapping, return computation, and feature merge.

Run with::

    python -m src.transform
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, TRIAD_MARKETS

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
    logit returns and time-to-resolution, and merge everything into a
    single analysis-ready DataFrame.
    """
    slug = market["slug"]
    end_date_str = market["end_date"]
    role = market["role"]
    category = market["category"]

    # ── Load hourly prices ───────────────────────────────────────────
    price_path = DATA_PROCESSED / f"prices_{slug}_hourly.parquet"
    if not price_path.exists():
        logger.warning("Missing hourly prices for %s – skipping", slug)
        return None
    df = pd.read_parquet(price_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Logit transform & returns ────────────────────────────────────
    df["logit_price"] = logit(df["price"])
    df["logit_return"] = df["logit_price"].diff()
    df["abs_logit_return"] = df["logit_return"].abs()

    # ── Time-to-resolution (days) ────────────────────────────────────
    try:
        resolve_dt = pd.to_datetime(end_date_str, utc=True)
    except Exception:
        resolve_dt = pd.Timestamp(end_date_str, tz="UTC")

    df["time_to_resolution"] = (
        (resolve_dt - df["timestamp"]).dt.total_seconds() / 86400.0
    )

    # ── Merge activity proxies ───────────────────────────────────────
    activity_path = DATA_PROCESSED / f"activity_{slug}.parquet"
    if activity_path.exists():
        act = pd.read_parquet(activity_path)
        act["timestamp"] = pd.to_datetime(act["timestamp"], utc=True)
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            act[
                [
                    "timestamp",
                    "trade_rate_1h",
                    "volume_rate_1h",
                    "trade_rate_24h",
                    "volume_rate_24h",
                ]
            ].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )
    else:
        logger.warning("No activity data for %s – columns will be NaN", slug)
        for col in (
            "trade_rate_1h",
            "volume_rate_1h",
            "trade_rate_24h",
            "volume_rate_24h",
        ):
            df[col] = np.nan

    # ── Merge order-book snapshot (static) ───────────────────────────
    ob_path = DATA_PROCESSED / f"orderbook_{slug}.parquet"
    if ob_path.exists():
        ob = pd.read_parquet(ob_path)
        for col in ("spread", "midpoint", "depth_1c_bid", "depth_1c_ask",
                     "depth_5c_bid", "depth_5c_ask"):
            df[col] = ob[col].iloc[0] if col in ob.columns else np.nan
    else:
        logger.warning("No order-book data for %s – columns will be NaN", slug)

    # ── Metadata columns ─────────────────────────────────────────────
    df["market_slug"] = slug
    df["role"] = role
    df["category"] = category

    return df


def transform_all() -> None:
    """Build feature tables for every triad market."""
    if not TRIAD_MARKETS:
        logger.error(
            "TRIAD_MARKETS is empty. Run the discovery notebook first."
        )
        sys.exit(1)

    for mkt in TRIAD_MARKETS:
        slug = mkt["slug"]
        out_path = DATA_PROCESSED / f"features_{slug}.parquet"

        logger.info("Transforming %s …", slug)
        df = build_features(mkt)
        if df is None or df.empty:
            logger.warning("No features produced for %s", slug)
            continue

        df.to_parquet(out_path, index=False)
        logger.info(
            "Wrote %d rows × %d cols → %s", len(df), len(df.columns), out_path
        )

    # Also produce a single stacked file for cross-triad analysis
    parts = []
    for mkt in TRIAD_MARKETS:
        p = DATA_PROCESSED / f"features_{mkt['slug']}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        combined_path = DATA_PROCESSED / "features_all.parquet"
        combined.to_parquet(combined_path, index=False)
        logger.info(
            "Wrote combined features (%d rows) → %s",
            len(combined),
            combined_path,
        )


if __name__ == "__main__":
    transform_all()
