"""
Central configuration for the data pipeline.

API base URLs, triad definitions (Stage 1), and politics-focused market list (Stage 2) live here so every downstream module reads from one place.
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

# ── Stage 2 / RQ3: Resolved US political markets ────────────────────────────
# 18 resolved markets spanning presidential, nomination, VP, state-level,
# and non-election political questions. Used for RQ3 analysis of how
# signatures vary with liquidity, activity, and time-to-resolution.
POLITICS_MARKETS: list[dict] = [
    # --- Main presidential races (highest volume, long duration) ---
    {"slug": "will-donald-trump-win-the-2024-us-presidential-election",
     "condition_id": "0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917",
     "clob_token_id": "21742633143463906290569050155826241533067272736897614950488156847949938836455",
     "end_date": "2024-11-05T12:00:00Z", "category": "politics",
     "market_type": "presidential"},
    {"slug": "will-kamala-harris-win-the-2024-us-presidential-election",
     "condition_id": "0xc6485bb7ea46d7bb89beb9c91e7572ecfc72a6273789496f78bc5e989e4d1638",
     "clob_token_id": "69236923620077691027083946871148646972011131466059644796654161903044970987404",
     "end_date": "2024-11-04T12:00:00Z", "category": "politics",
     "market_type": "presidential"},
    # --- Popular vote (same deadline, distinct outcome) ---
    {"slug": "will-kamala-harris-win-the-popular-vote-in-the-2024-presidential-election",
     "condition_id": "0x265366ede72d73e137b2b9095a6cdc9be6149290caa295738a95e3d881ad0865",
     "clob_token_id": "21271000291843361249209065706097167029083067325856089903026951915683588703117",
     "end_date": "2024-11-05T12:00:00Z", "category": "politics",
     "market_type": "popular_vote"},
    {"slug": "will-donald-trump-win-the-popular-vote-in-the-2024-presidential-election",
     "condition_id": "0xcd1b6b71a1964f15e2c14809594cbfa0d576270e8ef94c8c24913121097e09e5",
     "clob_token_id": "42699080635179861375280720242213672850141860123562672932351602811041149946128",
     "end_date": "2024-11-05T12:00:00Z", "category": "politics",
     "market_type": "popular_vote"},
    # --- Other presidential candidates (varying price paths) ---
    {"slug": "will-joe-biden-win-the-2024-us-presidential-election",
     "condition_id": "0x14018049e265a2d88f284be9588e2e3542e3a3df08ccdb344d28355dd7fdd8ef",
     "clob_token_id": "88027839609243624193415614179328679602612916497045596227438675518749602824929",
     "end_date": "2024-11-05T12:00:00Z", "category": "politics",
     "market_type": "presidential"},
    {"slug": "will-nikki-haley-win-the-2024-us-presidential-election",
     "condition_id": "0xced9f9d90c94db9f1e1dbd7d9fba82fe4fa7431c0d4e91e28896c8ac2d6acadd",
     "clob_token_id": "19083349462791593334532840548890602187185739923311385087650426802477691161360",
     "end_date": "2024-11-05T00:00:00Z", "category": "politics",
     "market_type": "presidential"},
    {"slug": "will-robert-f-kennedy-jr-win-the-2024-us-presidential-election",
     "condition_id": "0x7da35195ac3c7bf167f88ab0c27067a99020e36de67d39968b71d9debcdd925e",
     "clob_token_id": "75551890681049796405776295654438099776333571510662809052054780589218524237663",
     "end_date": "2024-11-04T12:00:00Z", "category": "politics",
     "market_type": "presidential"},
    {"slug": "will-ron-desantis-win-the-2024-us-presidential-election",
     "condition_id": "0xad6d309aaa500d96855996e84da00dfb2379548a693ca684d0877cf94fec05d1",
     "clob_token_id": "54541905023211985194827443687227462634594584372996482268933020846517872533280",
     "end_date": "2024-11-05T12:00:00Z", "category": "politics",
     "market_type": "presidential"},
    # --- Nomination races (shorter duration, eventful) ---
    {"slug": "will-joe-biden-win-the-2024-democratic-presidential-nomination",
     "condition_id": "0x653483009043f1663360bc35ed8fb542d3f9453916082d20ff025adaa2fe7032",
     "clob_token_id": "29771515314065403331935508893946579645282141892225585852084723308418208825505",
     "end_date": "2024-08-18T12:00:00Z", "category": "politics",
     "market_type": "nomination"},
    {"slug": "will-kamala-harris-win-the-2024-democratic-presidential-nomination",
     "condition_id": "0x9e9071636d176562592a98dfede865385ba0cff7864cfa01f0479ca6e9e26e1b",
     "clob_token_id": "61307174745343826105381155328818841321097085669107501488931143161146416325701",
     "end_date": "2024-08-19T00:00:00Z", "category": "politics",
     "market_type": "nomination"},
    {"slug": "will-donald-j-trump-win-the-2024-republican-presidential-nomination",
     "condition_id": "0x911d086ef50a087b694d40ffa105b8a384a3275aa66ff4fa66e55da1ff91f6b3",
     "clob_token_id": "59148051098887930522073798732825622149012805909987461569002068477591645842343",
     "end_date": "2024-07-13T00:00:00Z", "category": "politics",
     "market_type": "nomination"},
    # --- VP selection (shorter duration) ---
    {"slug": "will-jd-vance-win-the-2024-republican-vp-nomination",
     "condition_id": "0x6a031c4888fcfb886861d3616cce760ea284dbf46e355b197400c2b12a4c37a6",
     "clob_token_id": "87406562398979962299468279520190492088871692967284999093076550605890113634903",
     "end_date": "2024-09-04T12:00:00Z", "category": "politics",
     "market_type": "vp_selection"},
    {"slug": "will-vivek-ramaswamy-win-the-2024-republican-vp-nomination",
     "condition_id": "0x99c0965b13b8374737bf087337afe310fe9fe22d0b2a35d7972cae10ddba2711",
     "clob_token_id": "77140697329479964870511208765806474371806608348448281625118401244163837435440",
     "end_date": "2024-09-05T12:00:00Z", "category": "politics",
     "market_type": "vp_selection"},
    # --- State-level (different granularity) ---
    {"slug": "will-a-republican-win-texas-in-the-2024-us-presidential-election",
     "condition_id": "0x883f6ac0468d3449161454db37acb951bad5b55a7752bc2dd007e40356cee353",
     "clob_token_id": "22954390469393761478435587094271162660864811978634802609696033004592606875337",
     "end_date": "2024-11-05T12:00:00Z", "category": "politics",
     "market_type": "state_level"},
    {"slug": "will-maine-be-the-tipping-point-state",
     "condition_id": "0x9216286e80bbcd2c6fbb7a976666a945c87f7072733fcb0d1ffcb5a25655a81b",
     "clob_token_id": "94053993457114475555712405561795508926703709810252365187386727919511761724556",
     "end_date": "2024-11-05T12:00:00Z", "category": "politics",
     "market_type": "state_level"},
    # --- Non-election political ---
    {"slug": "us-government-shutdown-by-october-1",
     "condition_id": "0x09b3e4c716df2579afa0ede1bf00edd84f8cbe1e335785f90336e7a909c52f8e",
     "clob_token_id": "98045093822431547938365423095786421804452886266888528428746910959667685407691",
     "end_date": "2025-10-01T00:00:00Z", "category": "politics",
     "market_type": "policy"},
    {"slug": "joe-biden-impeached-before-2024-election",
     "condition_id": "0x01a9eea306780839c5cf9a15a572a438c23af6c49b57f67e8a379f5e48e0e4f8",
     "clob_token_id": "75955614077910094300452138293023370533365022621611366627435700504860612163857",
     "end_date": "2024-11-05T00:00:00Z", "category": "politics",
     "market_type": "policy"},
    # --- Primary / caucus (short duration, early resolve) ---
    {"slug": "will-nikki-haley-win-the-2024-republican-iowa-caucus",
     "condition_id": "0x50a21e51cc42148ee129af89fecf0b5f3218d340d687521ef99cae84dd6b4291",
     "clob_token_id": "13523137249951112077326627170171849224232457014547264934113561394395820729837",
     "end_date": "2024-01-15T00:00:00Z", "category": "politics",
     "market_type": "primary"},
]
