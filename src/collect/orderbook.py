"""
Order-book snapshot collector.

For resolved markets the book will be empty; we write a row with NaN
liquidity proxies so downstream code doesn't crash.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pandas as pd

from src.config import DATA_PROCESSED, ALL_MARKETS
from src.client import clob_get, save_raw_json

logger = logging.getLogger(__name__)


def _parse_levels(levels: list[dict]) -> list[tuple[float, float]]:
    return [(float(lv["price"]), float(lv["size"])) for lv in levels]


def _depth_within(
    levels: list[tuple[float, float]], midpoint: float, band: float
) -> float:
    return sum(sz for px, sz in levels if abs(px - midpoint) <= band)


def fetch_orderbook_snapshot(token_id: str, slug: str) -> dict:
    """Fetch live order-book snapshot; returns NaN row if book is empty."""
    try:
        raw = clob_get("/book", params={"token_id": token_id})
    except Exception as exc:
        logger.warning("Order-book request failed for %s: %s", slug, exc)
        raw = None

    if raw is None or (isinstance(raw, dict) and "error" in raw):
        logger.warning("No order book for %s (likely resolved) – writing NaN row", slug)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_slug": slug,
            "best_bid": math.nan, "best_ask": math.nan,
            "spread": math.nan, "midpoint": math.nan,
            "depth_1c_bid": math.nan, "depth_1c_ask": math.nan,
            "depth_5c_bid": math.nan, "depth_5c_ask": math.nan,
            "n_bid_levels": 0, "n_ask_levels": 0,
        }

    save_raw_json(raw, f"orderbook_{slug}.json")

    bids = _parse_levels(raw.get("bids", []))
    asks = _parse_levels(raw.get("asks", []))

    if not bids and not asks:
        logger.warning("Empty book for %s (resolved market)", slug)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_slug": slug,
            "best_bid": math.nan, "best_ask": math.nan,
            "spread": math.nan, "midpoint": math.nan,
            "depth_1c_bid": math.nan, "depth_1c_ask": math.nan,
            "depth_5c_bid": math.nan, "depth_5c_ask": math.nan,
            "n_bid_levels": 0, "n_ask_levels": 0,
        }

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 1.0
    spread = best_ask - best_bid
    midpoint = (best_bid + best_ask) / 2.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_slug": slug,
        "best_bid": best_bid, "best_ask": best_ask,
        "spread": spread, "midpoint": midpoint,
        "depth_1c_bid": _depth_within(bids, midpoint, 0.01),
        "depth_1c_ask": _depth_within(asks, midpoint, 0.01),
        "depth_5c_bid": _depth_within(bids, midpoint, 0.05),
        "depth_5c_ask": _depth_within(asks, midpoint, 0.05),
        "n_bid_levels": len(bids), "n_ask_levels": len(asks),
    }


def collect_orderbook_for_market(market: dict, *, force: bool = False) -> None:
    slug = market["slug"]
    token_id = market["clob_token_id"]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    out_path = DATA_PROCESSED / f"orderbook_{slug}.parquet"
    if out_path.exists() and not force:
        logger.info("Skipping order-book for %s (already exists)", slug)
        return

    logger.info("Fetching order-book snapshot for %s …", slug)
    row = fetch_orderbook_snapshot(token_id, slug)
    pd.DataFrame([row]).to_parquet(out_path, index=False)
    logger.info("Wrote order-book snapshot → %s", out_path)


def collect_all_orderbooks(*, force: bool = False) -> None:
    if not ALL_MARKETS:
        logger.error("ALL_MARKETS is empty. Run select_markets first.")
        return
    for mkt in ALL_MARKETS:
        collect_orderbook_for_market(mkt, force=force)
