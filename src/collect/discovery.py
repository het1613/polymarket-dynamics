"""
Market discovery helpers.

Uses the Gamma search endpoint (``/public-search``) to find resolved markets
by topic, since the tag-based ``/markets`` filter is unreliable.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.client import gamma_get, save_raw_json

logger = logging.getLogger(__name__)


# ── Low-level helpers ────────────────────────────────────────────────────────

def search_events(query: str, *, closed_only: bool = True) -> list[dict]:
    """Search Polymarket events and return market dicts from matching events."""
    raw = gamma_get("/public-search", params={"q": query})
    if raw is None:
        return []

    events = raw.get("events", [])
    if closed_only:
        events = [e for e in events if e.get("closed")]

    save_raw_json(events, f"search_{query.replace(' ', '_')[:40]}.json")

    markets: list[dict] = []
    for e in events:
        for m in e.get("markets", []):
            clob_raw = m.get("clobTokenIds", "[]")
            if isinstance(clob_raw, str):
                try:
                    clob_ids = json.loads(clob_raw)
                except json.JSONDecodeError:
                    clob_ids = []
            else:
                clob_ids = clob_raw

            markets.append({
                "question": m.get("question"),
                "slug": m.get("slug", ""),
                "condition_id": m.get("conditionId") or m.get("condition_id", ""),
                "clob_token_yes": clob_ids[0] if clob_ids else None,
                "end_date": m.get("endDate") or m.get("end_date_iso"),
                "volume": float(m.get("volume", 0) or 0),
                "active": m.get("active"),
                "closed": m.get("closed"),
            })

    markets.sort(key=lambda m: m["volume"], reverse=True)
    return markets


def _best_market(
    markets: list[dict],
    *,
    min_volume: float = 10_000,
    exclude_slugs: set[str] | None = None,
) -> dict | None:
    """Return the highest-volume market passing quality filters."""
    for m in markets:
        if m["volume"] < min_volume:
            continue
        if not m.get("clob_token_yes"):
            continue
        if not m.get("end_date"):
            continue
        if exclude_slugs and m["slug"] in exclude_slugs:
            continue
        return m
    return None


# ── Triad spec: search queries per role ──────────────────────────────────────

TRIAD_SPEC: list[dict[str, Any]] = [
    {
        "triad_id": "politics",
        "base_query": "Presidential Election Winner 2024",
        "base_category": "politics",
        "similar_query": "will Donald Trump win popular vote",
        "similar_category": "politics",
        "dissimilar_query": "Bitcoin 100k 2024",
        "dissimilar_category": "crypto",
    },
    {
        "triad_id": "crypto",
        "base_query": "Bitcoin 100k 2024",
        "base_category": "crypto",
        "similar_query": "ethereum merge proof of stake",
        "similar_category": "crypto",
        "dissimilar_query": "Oscars Best Picture 2025",
        "dissimilar_category": "entertainment",
    },
    {
        "triad_id": "sports",
        "base_query": "Super Bowl Champion 2025",
        "base_category": "sports",
        "similar_query": "2025 NBA Finals MVP",
        "similar_category": "sports",
        "dissimilar_query": "2024 Republican Iowa Caucus",
        "dissimilar_category": "politics",
    },
]


def auto_select_triads() -> dict[str, list[dict]]:
    """
    Search for resolved markets and build three triads across different
    domains.  Returns the ``TRIADS`` dict ready for ``src/config.py``.
    """
    used_slugs: set[str] = set()
    triads: dict[str, list[dict]] = {}

    for spec in TRIAD_SPEC:
        tid = spec["triad_id"]
        triad_markets: list[dict] = []

        for role, q_key, cat_key in [
            ("base", "base_query", "base_category"),
            ("similar", "similar_query", "similar_category"),
            ("dissimilar", "dissimilar_query", "dissimilar_category"),
        ]:
            query = spec[q_key]
            category = spec[cat_key]

            logger.info("Triad '%s' / %s: searching '%s' …", tid, role, query)
            candidates = search_events(query, closed_only=True)
            pick = _best_market(candidates, exclude_slugs=used_slugs)

            if pick is None:
                raise RuntimeError(
                    f"No suitable resolved market for triad={tid} "
                    f"role={role} query='{query}'"
                )

            used_slugs.add(pick["slug"])
            triad_markets.append({
                "slug": pick["slug"],
                "condition_id": pick["condition_id"],
                "clob_token_id": pick["clob_token_yes"],
                "end_date": pick["end_date"],
                "category": category,
                "role": role,
                "triad_id": tid,
            })
            logger.info(
                "  → %s  vol=$%.0f  slug=%s",
                role, pick["volume"], pick["slug"][:55],
            )

        triads[tid] = triad_markets

    return triads


# ── Config writer ────────────────────────────────────────────────────────────

def write_triads_to_config(triads: dict[str, list[dict]]) -> None:
    """Patch ``src/config.py`` with the selected triads."""
    config_path = Path(__file__).resolve().parent.parent / "config.py"
    text = config_path.read_text()

    triads_repr = repr(triads)
    all_markets_repr = repr([m for t in triads.values() for m in t])

    text = re.sub(
        r"^TRIADS: dict\[str, list\[dict\]\] = .*$",
        f"TRIADS: dict[str, list[dict]] = {triads_repr}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^ALL_MARKETS: list\[dict\] = .*$",
        f"ALL_MARKETS: list[dict] = {all_markets_repr}",
        text,
        flags=re.MULTILINE,
    )

    config_path.write_text(text)
    logger.info("Wrote %d triads (%d markets) → %s",
                len(triads),
                sum(len(t) for t in triads.values()),
                config_path)
