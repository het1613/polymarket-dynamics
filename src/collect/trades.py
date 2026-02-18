"""
Paginated trade-history fetcher.

Downloads all trades for each triad market from the Data API, derives
activity proxies (trade-rate, volume-rate), and writes Parquet.
"""

from __future__ import annotations

import logging

import pandas as pd
from tqdm import tqdm

from src.config import DATA_PROCESSED, TRIAD_MARKETS
from src.client import data_get, save_raw_json

logger = logging.getLogger(__name__)

_PAGE_SIZE = 10_000
_MAX_OFFSET = 10_000  # Data API hard cap


def fetch_trades(condition_id: str, slug: str) -> pd.DataFrame:
    """
    Paginate through ``/trades`` until exhausted or offset cap reached.

    Returns a DataFrame of individual trades.
    """
    all_rows: list[dict] = []
    offset = 0

    pbar = tqdm(desc=f"trades/{slug}", unit="trades")

    while True:
        params = {
            "market": condition_id,
            "limit": _PAGE_SIZE,
            "offset": offset,
        }
        page = data_get("/trades", params=params)

        if page is None or (isinstance(page, list) and len(page) == 0):
            break
        if isinstance(page, dict):
            page = page.get("data", page.get("trades", []))
        if not page:
            break

        all_rows.extend(page)
        pbar.update(len(page))

        if len(page) < _PAGE_SIZE:
            break

        offset += _PAGE_SIZE
        if offset >= _MAX_OFFSET:
            logger.warning(
                "Hit offset cap (%d) for %s – %d trades collected",
                _MAX_OFFSET,
                slug,
                len(all_rows),
            )
            break

    pbar.close()

    if not all_rows:
        logger.warning("No trades returned for %s", slug)
        return pd.DataFrame()

    save_raw_json(
        all_rows[:100], f"trades_{slug}_sample.json"
    )  # save a sample for inspection

    df = pd.DataFrame(all_rows)

    # Normalise column names that vary across API versions
    col_map = {}
    for candidate, target in [
        ("timestamp", "timestamp"),
        ("matchTime", "timestamp"),
        ("match_time", "timestamp"),
        ("createdAt", "timestamp"),
    ]:
        if candidate in df.columns:
            col_map[candidate] = target
            break
    for candidate, target in [
        ("size", "size"),
        ("matchSize", "size"),
        ("amount", "size"),
    ]:
        if candidate in df.columns:
            col_map[candidate] = target
            break
    for candidate, target in [
        ("price", "price"),
        ("matchPrice", "price"),
        ("outcomePrice", "price"),
    ]:
        if candidate in df.columns:
            col_map[candidate] = target
            break

    df.rename(columns=col_map, inplace=True)

    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if ts.dtype == "object":
            df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")
        else:
            # Assume unix seconds (or ms if values are huge)
            numeric = pd.to_numeric(ts, errors="coerce")
            if numeric.median() > 1e12:
                df["timestamp"] = pd.to_datetime(numeric, unit="ms", utc=True)
            else:
                df["timestamp"] = pd.to_datetime(numeric, unit="s", utc=True)

    for col in ("price", "size"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Fetched %d trades for %s", len(df), slug)
    return df


def derive_activity_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a raw trades DataFrame, compute rolling activity proxies aligned
    to an hourly grid:

    - ``trade_rate_1h``: number of trades in the preceding hour
    - ``volume_rate_1h``: total size traded in the preceding hour
    - ``trade_rate_24h``: trades in the preceding 24 h
    - ``volume_rate_24h``: size in the preceding 24 h
    """
    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    df = df.set_index("timestamp").sort_index()

    size_col = "size" if "size" in df.columns else None

    hourly = pd.DataFrame(
        index=pd.date_range(
            df.index.min().ceil("h"),
            df.index.max().floor("h"),
            freq="h",
        )
    )
    hourly.index.name = "timestamp"

    # Count trades per hourly bucket
    counts = df.resample("h").size().rename("trade_count")
    volume = (
        df["size"].resample("h").sum().rename("volume")
        if size_col
        else pd.Series(dtype=float, name="volume")
    )

    hourly = hourly.join(counts).join(volume).fillna(0)

    hourly["trade_rate_1h"] = hourly["trade_count"]
    hourly["volume_rate_1h"] = hourly["volume"]
    hourly["trade_rate_24h"] = (
        hourly["trade_count"].rolling(24, min_periods=1).sum()
    )
    hourly["volume_rate_24h"] = (
        hourly["volume"].rolling(24, min_periods=1).sum()
    )

    return hourly.reset_index()


def collect_trades_for_market(market: dict, *, force: bool = False) -> None:
    """Download trades + derive proxies for one market."""
    slug = market["slug"]
    cond = market["condition_id"]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    trades_path = DATA_PROCESSED / f"trades_{slug}.parquet"
    proxy_path = DATA_PROCESSED / f"activity_{slug}.parquet"

    if trades_path.exists() and not force:
        logger.info("Skipping trades for %s (already exists)", slug)
        df = pd.read_parquet(trades_path)
    else:
        logger.info("Fetching trades for %s …", slug)
        df = fetch_trades(cond, slug)
        if df.empty:
            return
        df["market_slug"] = slug
        df.to_parquet(trades_path, index=False)
        logger.info("Wrote %d trades → %s", len(df), trades_path)

    if proxy_path.exists() and not force:
        logger.info("Skipping activity proxies for %s (already exists)", slug)
    else:
        proxies = derive_activity_proxies(df)
        if not proxies.empty:
            proxies["market_slug"] = slug
            proxies.to_parquet(proxy_path, index=False)
            logger.info("Wrote %d activity rows → %s", len(proxies), proxy_path)


def collect_all_trades(*, force: bool = False) -> None:
    """Download trades for every market in the configured triad."""
    if not TRIAD_MARKETS:
        logger.error(
            "TRIAD_MARKETS is empty. Run the discovery notebook first."
        )
        return
    for mkt in TRIAD_MARKETS:
        collect_trades_for_market(mkt, force=force)
