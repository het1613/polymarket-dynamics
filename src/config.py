"""
Central configuration for the Stage 1 data pipeline.

API base URLs, triad market definitions, and collection settings live here
so every downstream module reads from one place.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# ── API base URLs ────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# ── Rate limiting ────────────────────────────────────────────────────────────
REQUEST_DELAY_S = 1.0  # seconds between API calls

# ── Price history settings ───────────────────────────────────────────────────
PRICE_FIDELITY_HOURLY = 60       # minutes
PRICE_FIDELITY_HIGHFREQ = 5     # minutes
HIGHFREQ_WINDOW_DAYS = 14       # recent window for 5-min data

# ── Triad market definitions ────────────────────────────────────────────────
# Populated after running the discovery notebook.
# Each entry: {
#   "slug": str,             # human-readable market slug
#   "condition_id": str,     # Gamma API condition ID (used by /trades)
#   "clob_token_id": str,    # YES-token CLOB token ID (used by /prices-history, /book)
#   "end_date": str,         # ISO-8601 resolution date
#   "category": str,         # tag / topic label
#   "role": str,             # "base" | "similar" | "dissimilar"
# }
TRIAD_MARKETS: list[dict] = [{'slug': 'will-vivek-ramaswamy-win-the-2028-republican-presidential-nomination', 'condition_id': '0xd997dc2a212a7d6673375a3b016db1fb214247142f8cde0cbf07f8e6d789877c', 'clob_token_id': '7847177314809025001842027337093046201021185723933452855314998541165696902642', 'end_date': '2028-11-07T00:00:00Z', 'category': 'politics', 'role': 'base'}, {'slug': 'will-delcy-rodrguez-be-the-leader-of-venezuela-end-of-2026', 'condition_id': '0xa01d48a973e40770719dab42faf1aeae5da4376d9eca46e77265c7551d1be0f7', 'clob_token_id': '38667196958602416137463628517439560119304765709104570192447644733106171420112', 'end_date': '2026-12-31T00:00:00Z', 'category': 'politics', 'role': 'similar'}, {'slug': 'nba-2025-26-clutch-player-of-the-year-jimmy-butler', 'condition_id': '0x4d9dc5a7e516dcec935b0381707ccd4c52d1e73fe8d90307ec8309e4728551c8', 'clob_token_id': '93287580662612374998000503364773686496854653048711183445257002067443625560147', 'end_date': '2026-06-30T00:00:00Z', 'category': 'sports', 'role': 'dissimilar'}]
