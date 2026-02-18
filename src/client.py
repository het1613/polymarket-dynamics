"""
Thin, rate-limit-aware HTTP helpers for the three Polymarket APIs.

Usage
-----
    from src.client import gamma_get, clob_get, clob_post, data_get

Every call respects ``config.REQUEST_DELAY_S`` by sleeping between requests.
Transient 429 / 5xx errors are retried with exponential back-off (up to 3
attempts).
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Any

import requests

from src.config import (
    GAMMA_API,
    CLOB_API,
    DATA_API,
    DATA_RAW,
    REQUEST_DELAY_S,
)

logger = logging.getLogger(__name__)

_session = requests.Session()
_last_call: float = 0.0

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0
_RETRYABLE = {429, 500, 502, 503, 504}


def _throttle() -> None:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < REQUEST_DELAY_S:
        time.sleep(REQUEST_DELAY_S - elapsed)
    _last_call = time.time()


def _request(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json_body: Any | None = None,
) -> Any:
    """Fire an HTTP request with throttling and retry logic."""
    for attempt in range(1, _MAX_RETRIES + 1):
        _throttle()
        try:
            resp = _session.request(
                method, url, params=params, json=json_body, timeout=30
            )
            if resp.status_code in _RETRYABLE and attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE**attempt
                logger.warning(
                    "HTTP %s from %s – retrying in %.1fs (attempt %d/%d)",
                    resp.status_code,
                    url,
                    wait,
                    attempt,
                    _MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt == _MAX_RETRIES:
                raise
            wait = _BACKOFF_BASE**attempt
            logger.warning(
                "Request error for %s – retrying in %.1fs (attempt %d/%d)",
                url,
                wait,
                attempt,
                _MAX_RETRIES,
            )
            time.sleep(wait)
    return None  # unreachable but keeps mypy happy


# ── Public helpers ───────────────────────────────────────────────────────────

def gamma_get(path: str, params: dict | None = None) -> Any:
    """GET from the Gamma (market discovery) API."""
    return _request("GET", f"{GAMMA_API}{path}", params=params)


def clob_get(path: str, params: dict | None = None) -> Any:
    """GET from the CLOB (order-book / pricing) API."""
    return _request("GET", f"{CLOB_API}{path}", params=params)


def clob_post(path: str, body: Any) -> Any:
    """POST to the CLOB API."""
    return _request("POST", f"{CLOB_API}{path}", json_body=body)


def data_get(path: str, params: dict | None = None) -> Any:
    """GET from the Data API (trades, etc.)."""
    return _request("GET", f"{DATA_API}{path}", params=params)


def save_raw_json(payload: Any, filename: str) -> Path:
    """Persist a raw API response to ``data/raw/`` for reproducibility."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW / filename
    dest.write_text(json.dumps(payload, indent=2))
    logger.info("Saved raw JSON → %s", dest)
    return dest
