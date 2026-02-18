"""
Order-book snapshot collector.

Fetches the current order book for each triad market, computes spread
and depth proxies, and writes a single-row-per-market Parquet file.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from src.config import DATA_PROCESSED, TRIAD_MARKETS
from src.client import clob_get, clob_post, save_raw_json

logger = logging.getLogger(__name__)


def _parse_levels(levels: list[dict]) -> list[tuple[float, float]]:
    """Convert raw bid/ask level dicts to ``[(price, size), …]``."""
    return [(float(lv["price"]), float(lv["size"])) for lv in levels]


def _depth_within(
    levels: list[tuple[float, float]], midpoint: float, band: float
) -> float:
    """Total size within *band* of *midpoint*."""
    return sum(sz for px, sz in levels if abs(px - midpoint) <= band)


def fetch_orderbook_snapshot(token_id: str, slug: str) -> dict:
    """
    Fetch the live order-book snapshot and derive liquidity proxies.

    Returns a flat dict ready to become one row of a DataFrame.
    """
    raw = clob_get("/book", params={"token_id": token_id})
    if raw is None:
        logger.error("No order book returned for %s", slug)
        return {}

    save_raw_json(raw, f"orderbook_{slug}.json")

    bids = _parse_levels(raw.get("bids", []))
    asks = _parse_levels(raw.get("asks", []))

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 1.0
    spread = best_ask - best_bid
    midpoint = (best_bid + best_ask) / 2.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_slug": slug,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "midpoint": midpoint,
        "depth_1c_bid": _depth_within(bids, midpoint, 0.01),
        "depth_1c_ask": _depth_within(asks, midpoint, 0.01),
        "depth_5c_bid": _depth_within(bids, midpoint, 0.05),
        "depth_5c_ask": _depth_within(asks, midpoint, 0.05),
        "n_bid_levels": len(bids),
        "n_ask_levels": len(asks),
    }


def fetch_spreads(token_ids: list[str]) -> dict[str, float]:
    """
    Batch-fetch spreads via POST /spreads as a cross-check.

    Returns ``{token_id: spread_value}``.
    """
    if not token_ids:
        return {}
    raw = clob_post("/spreads", [{"token_id": tid} for tid in token_ids])
    save_raw_json(raw, "spreads_batch.json")
    if isinstance(raw, dict):
        return {k: float(v) for k, v in raw.items()}
    return {}


def collect_orderbook_for_market(
    market: dict, *, force: bool = False
) -> None:
    """Snapshot the order book for one market and write Parquet."""
    slug = market["slug"]
    token_id = market["clob_token_id"]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    out_path = DATA_PROCESSED / f"orderbook_{slug}.parquet"
    if out_path.exists() and not force:
        logger.info("Skipping order-book for %s (already exists)", slug)
        return

    logger.info("Fetching order-book snapshot for %s …", slug)
    row = fetch_orderbook_snapshot(token_id, slug)
    if not row:
        return

    df = pd.DataFrame([row])
    df.to_parquet(out_path, index=False)
    logger.info("Wrote order-book snapshot → %s", out_path)


def collect_all_orderbooks(*, force: bool = False) -> None:
    """Snapshot order books for every market in the configured triad."""
    if not TRIAD_MARKETS:
        logger.error(
            "TRIAD_MARKETS is empty. Run the discovery notebook first."
        )
        return

    for mkt in TRIAD_MARKETS:
        collect_orderbook_for_market(mkt, force=force)

    # Cross-check via /spreads
    token_ids = [m["clob_token_id"] for m in TRIAD_MARKETS]
    spreads = fetch_spreads(token_ids)
    if spreads:
        logger.info("Spread cross-check: %s", spreads)
