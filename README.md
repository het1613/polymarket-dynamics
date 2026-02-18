# SYDE 532 – Prediction Market Dynamics (Stage 1)

Stage 1 data collection, transformation, and empirical analysis pipeline for
Polymarket prediction-market data.

## Quickstart

```bash
pip install -r requirements.txt
```

### 1. Discover and select triad markets

Run `notebooks/01_market_selection.ipynb` to browse categories and pick the
base / similar / dissimilar triad. The selected IDs are saved into
`src/config.py`.

### 2. Collect data

```bash
python -m src.collect.run_pipeline          # full run
python -m src.collect.run_pipeline --force  # re-download everything
```

### 3. Transform

```bash
python -m src.transform
```

### 4. Analyse

Open `notebooks/03_triad_analysis.ipynb` for the full Stage 1 analysis
(return distributions, volatility clustering, reversal analysis, liquidity
regressions, and cross-triad comparison).

## Project layout

```
src/
  config.py              – API URLs, triad IDs, settings
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
notebooks/
  01_market_selection.ipynb
  02_eda.ipynb
  03_triad_analysis.ipynb
data/
  raw/                   – archived JSON responses
  processed/             – Parquet files for analysis
```
