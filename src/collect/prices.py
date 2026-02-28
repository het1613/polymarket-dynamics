"""
Historical price fetcher.

The CLOB ``/prices-history`` endpoint only returns data for ``fidelity=5``
(5-minute bars) in ≤14-day windows.  This module fetches the full lifecycle
by walking backwards from the resolution date in 14-day chunks, then
resamples to an hourly series.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from src.config import (
    DATA_PROCESSED,
    PRICE_FIDELITY_HIGHFREQ,
    HIGHFREQ_WINDOW_DAYS,
    ALL_MARKETS,
    REQUEST_DELAY_S,
)
from src.client import clob_get, save_raw_json

logger = logging.getLogger(__name__)

_CHUNK_DAYS = 14  # max window the CLOB API supports per request
_MAX_EMPTY_CHUNKS = 3  # stop after N consecutive empty responses


def _resolve_ts(market: dict) -> int:
    """UNIX timestamp for the market's resolution date (or now as fallback)."""
    try:
        return int(pd.to_datetime(market["end_date"], utc=True).timestamp())
    except Exception:
        return int(datetime.now(timezone.utc).timestamp())


def fetch_full_history_5min(token_id: str, slug: str, end_ts: int) -> pd.DataFrame:
    """
    Walk backwards in 14-day chunks from *end_ts* collecting 5-min bars
    until the API returns no data for ``_MAX_EMPTY_CHUNKS`` consecutive windows.
    """
    all_rows: list[dict] = []
    cursor = end_ts
    empty_streak = 0

    while empty_streak < _MAX_EMPTY_CHUNKS:
        start = cursor - _CHUNK_DAYS * 86400
        raw = clob_get("/prices-history", params={
            "market": token_id,
            "fidelity": PRICE_FIDELITY_HIGHFREQ,
            "startTs": start,
            "endTs": cursor,
        })
        history = (raw or {}).get("history", [])

        if not history:
            empty_streak += 1
            cursor = start
            continue

        empty_streak = 0
        all_rows.extend(history)
        cursor = start
        time.sleep(REQUEST_DELAY_S * 0.5)

    if not all_rows:
        logger.warning("No 5-min price data found for %s", slug)
        return pd.DataFrame()

    save_raw_json(all_rows[:200], f"prices_{slug[:80]}_5min_sample.json")

    df = pd.DataFrame(all_rows)
    df.rename(columns={"t": "timestamp", "p": "price"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def resample_to_hourly(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Resample a 5-min DataFrame to hourly OHLC + last-price."""
    if df_5min.empty:
        return pd.DataFrame()
    tmp = df_5min.set_index("timestamp").sort_index()
    hourly = tmp["price"].resample("h").agg(["first", "max", "min", "last", "count"])
    hourly.columns = ["open", "high", "low", "price", "n_obs"]
    hourly = hourly.dropna(subset=["price"]).reset_index()
    return hourly


def collect_prices_for_market(market: dict, *, force: bool = False) -> None:
    """Download 5-min history → hourly resample → last-14-days highfreq."""
    slug = market["slug"]
    token_id = market["clob_token_id"]
    end_ts = _resolve_ts(market)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Full 5-min history ────────────────────────────────────────────
    raw5_path = DATA_PROCESSED / f"prices_{slug}_5min.parquet"
    hourly_path = DATA_PROCESSED / f"prices_{slug}_hourly.parquet"

    if hourly_path.exists() and raw5_path.exists() and not force:
        logger.info("Skipping prices for %s (already exists)", slug)
    else:
        logger.info("Fetching full 5-min history for %s …", slug)
        df_5min = fetch_full_history_5min(token_id, slug, end_ts)

        if not df_5min.empty:
            df_5min["market_slug"] = slug
            df_5min["category"] = market["category"]
            df_5min.to_parquet(raw5_path, index=False)
            logger.info("Wrote %d 5-min rows → %s", len(df_5min), raw5_path)

            df_hourly = resample_to_hourly(df_5min)
            df_hourly["market_slug"] = slug
            df_hourly["category"] = market["category"]
            df_hourly.to_parquet(hourly_path, index=False)
            logger.info("Wrote %d hourly rows → %s", len(df_hourly), hourly_path)
        else:
            logger.warning("No price data at all for %s", slug)

    # ── High-frequency: last 14 days before resolution ────────────────
    hf_path = DATA_PROCESSED / f"prices_{slug}_highfreq.parquet"
    if hf_path.exists() and not force:
        logger.info("Skipping high-freq prices for %s (already exists)", slug)
    else:
        start_ts = end_ts - HIGHFREQ_WINDOW_DAYS * 86400
        logger.info("Fetching high-freq (%d days before resolution) for %s …",
                     HIGHFREQ_WINDOW_DAYS, slug)
        raw = clob_get("/prices-history", params={
            "market": token_id,
            "fidelity": PRICE_FIDELITY_HIGHFREQ,
            "startTs": start_ts,
            "endTs": end_ts,
        })
        history = (raw or {}).get("history", [])
        if history:
            df_hf = pd.DataFrame(history)
            df_hf.rename(columns={"t": "timestamp", "p": "price"}, inplace=True)
            df_hf["timestamp"] = pd.to_datetime(df_hf["timestamp"], unit="s", utc=True)
            df_hf["price"] = df_hf["price"].astype(float)
            df_hf = df_hf.sort_values("timestamp").reset_index(drop=True)
            df_hf["market_slug"] = slug
            df_hf["category"] = market["category"]
            df_hf.to_parquet(hf_path, index=False)
            logger.info("Wrote %d high-freq rows → %s", len(df_hf), hf_path)


def collect_all_prices(*, force: bool = False) -> None:
    """Download prices for every configured market."""
    if not ALL_MARKETS:
        logger.error("ALL_MARKETS is empty. Run select_markets first.")
        return
    for mkt in ALL_MARKETS:
        collect_prices_for_market(mkt, force=force)
