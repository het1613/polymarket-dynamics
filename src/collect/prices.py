"""
Historical price fetcher.

Downloads full-history (hourly) and recent high-frequency (5-min) price
series for each triad market and writes Parquet files.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta

import pandas as pd

from src.config import (
    DATA_PROCESSED,
    PRICE_FIDELITY_HOURLY,
    PRICE_FIDELITY_HIGHFREQ,
    HIGHFREQ_WINDOW_DAYS,
    TRIAD_MARKETS,
)
from src.client import clob_get, save_raw_json

logger = logging.getLogger(__name__)


def fetch_price_history(
    token_id: str,
    slug: str,
    *,
    fidelity: int = PRICE_FIDELITY_HOURLY,
    start_ts: int | None = None,
    end_ts: int | None = None,
    interval: str | None = "max",
) -> pd.DataFrame:
    """
    Fetch price history for a single CLOB token and return a DataFrame.

    Parameters
    ----------
    token_id : CLOB token ID (YES token).
    slug : human-readable market slug (used for filenames).
    fidelity : resolution in minutes (60 = hourly, 5 = high-freq).
    start_ts / end_ts : explicit UNIX timestamps (mutually exclusive
        with *interval*).
    interval : convenience window such as ``"max"``, ``"1w"``, ``"1d"``.
    """
    params: dict = {"market": token_id, "fidelity": fidelity}
    if start_ts is not None and end_ts is not None:
        params["startTs"] = start_ts
        params["endTs"] = end_ts
    elif interval is not None:
        params["interval"] = interval

    raw = clob_get("/prices-history", params=params)
    if raw is None:
        logger.error("No data returned for token %s (%s)", token_id, slug)
        return pd.DataFrame()

    save_raw_json(raw, f"prices_{slug}_f{fidelity}.json")

    history = raw.get("history", [])
    if not history:
        logger.warning("Empty history for %s (fidelity=%d)", slug, fidelity)
        return pd.DataFrame()

    df = pd.DataFrame(history)
    df.rename(columns={"t": "timestamp", "p": "price"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def collect_prices_for_market(market: dict, *, force: bool = False) -> None:
    """Download hourly + high-freq price series for one market dict."""
    slug = market["slug"]
    token_id = market["clob_token_id"]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # -- Hourly (full history) --
    hourly_path = DATA_PROCESSED / f"prices_{slug}_hourly.parquet"
    if hourly_path.exists() and not force:
        logger.info("Skipping hourly prices for %s (already exists)", slug)
    else:
        logger.info("Fetching hourly prices for %s …", slug)
        df_hourly = fetch_price_history(
            token_id, slug, fidelity=PRICE_FIDELITY_HOURLY, interval="max"
        )
        if not df_hourly.empty:
            df_hourly["market_slug"] = slug
            df_hourly["category"] = market["category"]
            df_hourly.to_parquet(hourly_path, index=False)
            logger.info(
                "Wrote %d hourly rows → %s", len(df_hourly), hourly_path
            )

    # -- High-frequency (recent window) --
    hf_path = DATA_PROCESSED / f"prices_{slug}_highfreq.parquet"
    if hf_path.exists() and not force:
        logger.info("Skipping high-freq prices for %s (already exists)", slug)
    else:
        logger.info("Fetching 5-min prices for %s (last %d days) …", slug, HIGHFREQ_WINDOW_DAYS)
        now = int(datetime.now(timezone.utc).timestamp())
        start = now - HIGHFREQ_WINDOW_DAYS * 86400
        df_hf = fetch_price_history(
            token_id,
            slug,
            fidelity=PRICE_FIDELITY_HIGHFREQ,
            start_ts=start,
            end_ts=now,
            interval=None,
        )
        if not df_hf.empty:
            df_hf["market_slug"] = slug
            df_hf["category"] = market["category"]
            df_hf.to_parquet(hf_path, index=False)
            logger.info("Wrote %d high-freq rows → %s", len(df_hf), hf_path)


def collect_all_prices(*, force: bool = False) -> None:
    """Download prices for every market in the configured triad."""
    if not TRIAD_MARKETS:
        logger.error(
            "TRIAD_MARKETS is empty. Run the discovery notebook first."
        )
        return
    for mkt in TRIAD_MARKETS:
        collect_prices_for_market(mkt, force=force)
