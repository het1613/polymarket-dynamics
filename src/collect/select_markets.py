"""
Automated market selection.

Queries the Polymarket API for resolved markets, picks 9 markets across
3 triads in different domains, and writes the result to ``src/config.py``.

Run with::

    python -m src.collect.select_markets
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``src`` is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.collect.discovery import auto_select_triads, write_triads_to_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Selecting 3 triads of resolved markets …")
    triads = auto_select_triads()

    # Print summary
    total = 0
    for tid, markets in triads.items():
        print(f"\n{'═' * 70}")
        print(f"  Triad: {tid}")
        print(f"{'═' * 70}")
        for m in markets:
            print(
                f"  [{m['role']:11s}]  {m['category']:14s}  "
                f"vol=n/a  slug={m['slug'][:60]}"
            )
            total += 1

    print(f"\nTotal: {total} markets across {len(triads)} triads\n")

    write_triads_to_config(triads)
    logger.info("Done. Config updated.")


if __name__ == "__main__":
    main()
