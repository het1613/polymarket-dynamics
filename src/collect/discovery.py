"""
Market discovery helpers.

Fetches available tags and markets from the Gamma API and returns tidy
DataFrames so the triad can be selected interactively in a notebook.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.client import gamma_get, save_raw_json

logger = logging.getLogger(__name__)


def fetch_tags() -> pd.DataFrame:
    """Return a DataFrame of all Polymarket category tags."""
    raw: list[dict[str, Any]] = gamma_get("/tags") or []
    save_raw_json(raw, "tags.json")
    df = pd.json_normalize(raw)
    logger.info("Fetched %d tags", len(df))
    return df


def fetch_markets_by_tag(
    tag_slug: str,
    *,
    limit: int = 50,
    active: bool = True,
    closed: bool = False,
) -> pd.DataFrame:
    """Fetch markets for a given tag, sorted by volume descending."""
    params: dict[str, Any] = {
        "tag_slug": tag_slug,
        "limit": limit,
        "active": active,
        "closed": closed,
        "order": "volume",
        "ascending": False,
    }
    raw: list[dict[str, Any]] = gamma_get("/markets", params=params) or []
    save_raw_json(raw, f"markets_{tag_slug}.json")

    if not raw:
        logger.warning("No markets returned for tag '%s'", tag_slug)
        return pd.DataFrame()

    rows = []
    for m in raw:
        clob_ids = m.get("clobTokenIds") or []
        # clobTokenIds is a JSON-encoded list string on some responses
        if isinstance(clob_ids, str):
            import json
            try:
                clob_ids = json.loads(clob_ids)
            except json.JSONDecodeError:
                clob_ids = []

        rows.append(
            {
                "question": m.get("question"),
                "slug": m.get("slug", m.get("condition_id", "")),
                "condition_id": m.get("conditionId") or m.get("condition_id", ""),
                "clob_token_yes": clob_ids[0] if len(clob_ids) > 0 else None,
                "clob_token_no": clob_ids[1] if len(clob_ids) > 1 else None,
                "end_date": m.get("endDate") or m.get("end_date_iso"),
                "volume": float(m.get("volume", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "active": m.get("active"),
                "closed": m.get("closed"),
                "tag": tag_slug,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("volume", ascending=False).reset_index(drop=True)
    logger.info(
        "Fetched %d markets for tag '%s' (top volume: %.0f)",
        len(df),
        tag_slug,
        df["volume"].iloc[0] if len(df) else 0,
    )
    return df


def fetch_all_markets(
    *,
    limit: int = 100,
    active: bool = True,
    closed: bool = False,
) -> pd.DataFrame:
    """Fetch top markets across all categories by volume."""
    params: dict[str, Any] = {
        "limit": limit,
        "active": active,
        "closed": closed,
        "order": "volume",
        "ascending": False,
    }
    raw: list[dict[str, Any]] = gamma_get("/markets", params=params) or []
    save_raw_json(raw, "markets_all.json")

    if not raw:
        return pd.DataFrame()

    rows = []
    for m in raw:
        clob_ids = m.get("clobTokenIds") or []
        if isinstance(clob_ids, str):
            import json
            try:
                clob_ids = json.loads(clob_ids)
            except json.JSONDecodeError:
                clob_ids = []

        rows.append(
            {
                "question": m.get("question"),
                "slug": m.get("slug", m.get("condition_id", "")),
                "condition_id": m.get("conditionId") or m.get("condition_id", ""),
                "clob_token_yes": clob_ids[0] if len(clob_ids) > 0 else None,
                "clob_token_no": clob_ids[1] if len(clob_ids) > 1 else None,
                "end_date": m.get("endDate") or m.get("end_date_iso"),
                "volume": float(m.get("volume", 0) or 0),
                "liquidity": float(m.get("liquidity", 0) or 0),
                "active": m.get("active"),
                "closed": m.get("closed"),
                "tags": str(m.get("tags", "")),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("volume", ascending=False).reset_index(drop=True)
    return df


def build_triad_config(selections: list[dict]) -> list[dict]:
    """
    Turn a list of three dicts (with keys slug, condition_id,
    clob_token_yes, end_date, category, role) into the format
    expected by ``config.TRIAD_MARKETS``.
    """
    triad = []
    for s in selections:
        triad.append(
            {
                "slug": s["slug"],
                "condition_id": s["condition_id"],
                "clob_token_id": s["clob_token_yes"],
                "end_date": s["end_date"],
                "category": s["category"],
                "role": s["role"],
            }
        )
    return triad
