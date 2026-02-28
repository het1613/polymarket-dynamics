"""
Central configuration for the Stage 1 data pipeline.

API base URLs, triad definitions, and collection settings live here
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
HIGHFREQ_WINDOW_DAYS = 14       # days before resolution for 5-min data

# ── Triad definitions ───────────────────────────────────────────────────────
# Populated by running: python -m src.collect.select_markets
#
# Structure: {triad_id: [base_market, similar_market, dissimilar_market]}
# Each market dict contains:
#   slug, condition_id, clob_token_id, end_date, category, role, triad_id
#
# Triad A (politics): base=politics, similar=politics, dissimilar=crypto
# Triad B (crypto):   base=crypto,   similar=crypto,   dissimilar=pop-culture
# Triad C (sports):   base=sports,   similar=sports,   dissimilar=politics

TRIADS: dict[str, list[dict]] = {'politics': [{'slug': 'will-donald-trump-win-the-2024-us-presidential-election', 'condition_id': '0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917', 'clob_token_id': '21742633143463906290569050155826241533067272736897614950488156847949938836455', 'end_date': '2024-11-05T12:00:00Z', 'category': 'politics', 'role': 'base', 'triad_id': 'politics'}, {'slug': 'will-kamala-harris-win-the-popular-vote-in-the-2024-presidential-election', 'condition_id': '0x265366ede72d73e137b2b9095a6cdc9be6149290caa295738a95e3d881ad0865', 'clob_token_id': '21271000291843361249209065706097167029083067325856089903026951915683588703117', 'end_date': '2024-11-05T12:00:00Z', 'category': 'politics', 'role': 'similar', 'triad_id': 'politics'}, {'slug': 'will-bitcoin-hit-100k-in-2024', 'condition_id': '0x9c66114d2dfe2139325cc7a408a5cd5d2e73b55d919e2141b3a0ed83fc15895d', 'clob_token_id': '64903093311385616430821497488306433314807585397286521531639186532059591846310', 'end_date': '2024-12-30T12:00:00Z', 'category': 'crypto', 'role': 'dissimilar', 'triad_id': 'politics'}], 'crypto': [{'slug': 'will-bitcoin-hit-100k-again-in-2024-dec-23', 'condition_id': '0x97587c58a3407fcc9a8df6396aaa8b66eff8b0c799fdf81880f258755b7d529c', 'clob_token_id': '2373171113481420188172421609695981388987238078561291090755743560262815731854', 'end_date': '2024-12-31T12:00:00Z', 'category': 'crypto', 'role': 'base', 'triad_id': 'crypto'}, {'slug': 'will-microstrategy-purchase-more-bitcoin-in-2024-dec-16', 'condition_id': '0xea2ff9d0ba315a4edc9755f46c00ec16bd916ffac0b0a6b571357d8f773dddaf', 'clob_token_id': '114336723244967523692771633488815816513121181465478099926587038213645095513800', 'end_date': '2024-12-31T00:00:00Z', 'category': 'crypto', 'role': 'similar', 'triad_id': 'crypto'}, {'slug': 'oscars-best-picture-will-anora-win-best-picture-at-the-2025-oscars', 'condition_id': '0x75a8981659dc4efa2a2bdb4b3d31c5a5827500a2feb19f6a4cb3889893f1485c', 'clob_token_id': '1018044885195187396947687732754809918865343613107923057688573272106246886316', 'end_date': '2025-03-02T12:00:00Z', 'category': 'entertainment', 'role': 'dissimilar', 'triad_id': 'crypto'}], 'sports': [{'slug': 'will-saquon-barkley-win-super-bowl-lix-mvp', 'condition_id': '0x1d8821183636eb50838d5fb9bd7ee7b5a5d7c37a43eca8402a09d3a248a73717', 'clob_token_id': '22535062482817123000285472211110590288002335348533150289803030312393762193632', 'end_date': '2025-02-09T12:00:00Z', 'category': 'sports', 'role': 'base', 'triad_id': 'sports'}, {'slug': 'will-shai-gilgeous-alexander-win-the-2025-nba-finals-mvp', 'condition_id': '0xa6de2e10501f82a91e012ab54223416fcd3644b0ec054bdb22a68e2062527e50', 'clob_token_id': '89925387190278774582641776158919820836874249308405980593525773746053389719723', 'end_date': '2025-06-24T12:00:00Z', 'category': 'sports', 'role': 'similar', 'triad_id': 'sports'}, {'slug': 'will-nikki-haley-win-the-2024-republican-iowa-caucus', 'condition_id': '0x50a21e51cc42148ee129af89fecf0b5f3218d340d687521ef99cae84dd6b4291', 'clob_token_id': '13523137249951112077326627170171849224232457014547264934113561394395820729837', 'end_date': '2024-01-15T00:00:00Z', 'category': 'politics', 'role': 'dissimilar', 'triad_id': 'sports'}]}

# Flat list derived from TRIADS for collection / transform iteration
ALL_MARKETS: list[dict] = [{'slug': 'will-donald-trump-win-the-2024-us-presidential-election', 'condition_id': '0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917', 'clob_token_id': '21742633143463906290569050155826241533067272736897614950488156847949938836455', 'end_date': '2024-11-05T12:00:00Z', 'category': 'politics', 'role': 'base', 'triad_id': 'politics'}, {'slug': 'will-kamala-harris-win-the-popular-vote-in-the-2024-presidential-election', 'condition_id': '0x265366ede72d73e137b2b9095a6cdc9be6149290caa295738a95e3d881ad0865', 'clob_token_id': '21271000291843361249209065706097167029083067325856089903026951915683588703117', 'end_date': '2024-11-05T12:00:00Z', 'category': 'politics', 'role': 'similar', 'triad_id': 'politics'}, {'slug': 'will-bitcoin-hit-100k-in-2024', 'condition_id': '0x9c66114d2dfe2139325cc7a408a5cd5d2e73b55d919e2141b3a0ed83fc15895d', 'clob_token_id': '64903093311385616430821497488306433314807585397286521531639186532059591846310', 'end_date': '2024-12-30T12:00:00Z', 'category': 'crypto', 'role': 'dissimilar', 'triad_id': 'politics'}, {'slug': 'will-bitcoin-hit-100k-again-in-2024-dec-23', 'condition_id': '0x97587c58a3407fcc9a8df6396aaa8b66eff8b0c799fdf81880f258755b7d529c', 'clob_token_id': '2373171113481420188172421609695981388987238078561291090755743560262815731854', 'end_date': '2024-12-31T12:00:00Z', 'category': 'crypto', 'role': 'base', 'triad_id': 'crypto'}, {'slug': 'will-microstrategy-purchase-more-bitcoin-in-2024-dec-16', 'condition_id': '0xea2ff9d0ba315a4edc9755f46c00ec16bd916ffac0b0a6b571357d8f773dddaf', 'clob_token_id': '114336723244967523692771633488815816513121181465478099926587038213645095513800', 'end_date': '2024-12-31T00:00:00Z', 'category': 'crypto', 'role': 'similar', 'triad_id': 'crypto'}, {'slug': 'oscars-best-picture-will-anora-win-best-picture-at-the-2025-oscars', 'condition_id': '0x75a8981659dc4efa2a2bdb4b3d31c5a5827500a2feb19f6a4cb3889893f1485c', 'clob_token_id': '1018044885195187396947687732754809918865343613107923057688573272106246886316', 'end_date': '2025-03-02T12:00:00Z', 'category': 'entertainment', 'role': 'dissimilar', 'triad_id': 'crypto'}, {'slug': 'will-saquon-barkley-win-super-bowl-lix-mvp', 'condition_id': '0x1d8821183636eb50838d5fb9bd7ee7b5a5d7c37a43eca8402a09d3a248a73717', 'clob_token_id': '22535062482817123000285472211110590288002335348533150289803030312393762193632', 'end_date': '2025-02-09T12:00:00Z', 'category': 'sports', 'role': 'base', 'triad_id': 'sports'}, {'slug': 'will-shai-gilgeous-alexander-win-the-2025-nba-finals-mvp', 'condition_id': '0xa6de2e10501f82a91e012ab54223416fcd3644b0ec054bdb22a68e2062527e50', 'clob_token_id': '89925387190278774582641776158919820836874249308405980593525773746053389719723', 'end_date': '2025-06-24T12:00:00Z', 'category': 'sports', 'role': 'similar', 'triad_id': 'sports'}, {'slug': 'will-nikki-haley-win-the-2024-republican-iowa-caucus', 'condition_id': '0x50a21e51cc42148ee129af89fecf0b5f3218d340d687521ef99cae84dd6b4291', 'clob_token_id': '13523137249951112077326627170171849224232457014547264934113561394395820729837', 'end_date': '2024-01-15T00:00:00Z', 'category': 'politics', 'role': 'dissimilar', 'triad_id': 'sports'}]
