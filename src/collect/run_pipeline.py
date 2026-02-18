"""
Pipeline orchestrator.

Run with::

    python -m src.collect.run_pipeline          # normal run (skip existing)
    python -m src.collect.run_pipeline --force   # re-download everything

Executes:  prices → order-book snapshots → trades  for each triad market.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import TRIAD_MARKETS
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
    parser = argparse.ArgumentParser(
        description="Stage 1 data collection pipeline"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all data even if files already exist",
    )
    args = parser.parse_args()

    if not TRIAD_MARKETS:
        logger.error(
            "TRIAD_MARKETS is empty in src/config.py. "
            "Run notebooks/01_market_selection.ipynb first."
        )
        sys.exit(1)

    logger.info(
        "Starting collection for %d triad markets (force=%s)",
        len(TRIAD_MARKETS),
        args.force,
    )
    for i, m in enumerate(TRIAD_MARKETS, 1):
        logger.info(
            "  [%d/%d] %s (%s / %s)",
            i,
            len(TRIAD_MARKETS),
            m["slug"],
            m["role"],
            m["category"],
        )

    logger.info("── Step 1/3: Historical prices ──")
    collect_all_prices(force=args.force)

    logger.info("── Step 2/3: Order-book snapshots ──")
    collect_all_orderbooks(force=args.force)

    logger.info("── Step 3/3: Trade history ──")
    collect_all_trades(force=args.force)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
