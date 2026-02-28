"""
Pipeline orchestrator.

Run with::

    python -m src.collect.run_pipeline          # normal run (skip existing)
    python -m src.collect.run_pipeline --force   # re-download everything

Executes:  prices → order-book snapshots → trades  for all 9 markets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked directly
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import ALL_MARKETS
from src.collect.prices import collect_all_prices
from src.collect.orderbook import collect_all_orderbooks
from src.collect.trades import collect_all_trades

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 data collection pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Re-download all data even if files already exist")
    args = parser.parse_args()

    if not ALL_MARKETS:
        logger.error("ALL_MARKETS is empty in src/config.py. "
                      "Run: python -m src.collect.select_markets")
        sys.exit(1)

    logger.info("Starting collection for %d markets (force=%s)",
                len(ALL_MARKETS), args.force)
    for i, m in enumerate(ALL_MARKETS, 1):
        logger.info("  [%d/%d] %s (%s / %s / %s)",
                     i, len(ALL_MARKETS), m["slug"][:50],
                     m["triad_id"], m["role"], m["category"])

    logger.info("── Step 1/3: Historical prices ──")
    collect_all_prices(force=args.force)

    logger.info("── Step 2/3: Order-book snapshots ──")
    collect_all_orderbooks(force=args.force)

    logger.info("── Step 3/3: Trade history ──")
    collect_all_trades(force=args.force)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
