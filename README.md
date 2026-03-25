# SYDE 532 Final Project: Prediction Markets as Complex Adaptive Systems
### Team: Het Patel, Aayush Soni, Mehak Dhaliwal, Aarya Penneru

## Quickstart

```bash
python -m venv venv && source venv/bin/activate 
pip install -r requirements.txt
```

### 1. Discover and select triad markets

Run `notebooks/01_market_selection.ipynb` to browse categories and pick the
base / similar / dissimilar triad. Selected IDs live in `src/config.py` under
`TRIADS` / `ALL_MARKETS`.

### 2. Collect data

```bash
python -m src.collect.run_pipeline          # full run
python -m src.collect.run_pipeline --force  # re-download everything
```

### 3. Transform

```bash
python -m src.transform
```

This builds per-market feature Parquet files (logit prices, returns, activity, order-book fields where available).

### 4. Stage 1 – Triad analysis

Open `notebooks/03_triad_analysis.ipynb` for return distributions, volatility clustering, reversal analysis, liquidity regressions, and cross-triad comparison.

### 5. Politics panel and RQ3 (market conditions)

Resolved US political markets are listed in `src/config.py` as `POLITICS_MARKETS`. 
The default collector (`python -m src.collect.run_pipeline`) and `python -m src.transform` iterate `ALL_MARKETS` (triad markets only). 
To rebuild politics features from APIs, collect hourly prices, activity, and order-book inputs for each politics slug (same artifacts as the triad pipeline), run `build_features` from `src/transform.py` per market, then concatenate the per-market `features_<slug>.parquet` files into `data/processed/features_politics.parquet`. 

Run `notebooks/04_rq3_market_conditions.ipynb` for how signatures vary with activity proxies and time-to-resolution. Figures are written under `plots/rq3_*.png`.

### 6. RQ4 – Agent-based model

`src/abm/` implements an hourly logit-space ABM (informed, noise, herding; latent anchor; reduced-form impact; optional endogenous participation and state-dependent depth). 
`notebooks/05_rq4_abm.ipynb` loads empirical targets from `features_politics.parquet`, runs calibration, compares simulated panels to the empirical pipeline, and saves `data/processed/sim_stage1.parquet`, `data/processed/sim_stage2.parquet`, and `data/processed/abm_calibration.json`.
Figures: `plots/rq4_*.png`.

## Project layout

```
src/
  config.py              – API URLs, triad IDs, POLITICS_MARKETS, settings
  client.py              – rate-limited HTTP helpers
  collect/
    discovery.py         – market discovery helpers
    prices.py            – historical price fetcher
    orderbook.py         – order-book snapshot collector
    trades.py            – paginated trade-history fetcher
    run_pipeline.py      – orchestrator script
  transform.py           – logit transform, returns, feature merge
  analyze/
    return_distribution.py
    volatility.py
    reversal.py
    liquidity_regression.py
    cross_triad.py
  abm/
    model.py             – simulator + feature-schema export
    calibration.py       – targets, distance, random-search calibration
notebooks/
  01_market_selection.ipynb
  02_eda.ipynb
  03_triad_analysis.ipynb
  04_rq3_market_conditions.ipynb
  05_rq4_abm.ipynb
data/
  raw/                   – archived JSON responses
  processed/             – Parquet features, trades, synthetic ABM panels
plots/                   – triad and RQ3/RQ4 figures
```