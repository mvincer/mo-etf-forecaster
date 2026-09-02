# mo-etf-forecaster

Standalone ETF forecasting engine: for each ticker in a 40-name universe it forecasts the
probability of an up move and the return distribution over the next 1-5 trading days, keeps a
tamper-proof out-of-sample track record, and exposes everything to the Mo_Dash dashboard through
a small set of parquet artifacts.

## What it does

1. **Ingest** - full-history daily bars (yfinance), the CBOE volatility complex
   (VIX/VIX9D/VIX3M/VIX6M/VVIX/SKEW/...), 10 cross-asset series, fast FRED macro (yields, curve,
   credit, dollar, oil), slow point-in-time macro via ALFRED vintages (CPI, INDPRO, UNRATE,
   claims, M2, GDP, housing, sentiment), Shiller S&P 500 earnings/CAPE, per-ticker implied-vol
   history from IBKR, and a daily option-chain snapshot.
2. **Features** - ~400 columns per ticker, tagged into 12 blocks (PRICE always on + 11
   toggleable): TECH, PATTERN_D1, PATTERN_HTF, REGIME, VOLOPT, SKEW, MACRO_FAST, MACRO_SLOW,
   EARNINGS, CROSS, SEASON. Chart patterns (double tops, head-and-shoulders, wedges, flags,
   channels, trendlines) are detected on daily, weekly, and monthly bars with strict as-of
   alignment.
3. **Search** - a staged include/exclude ablation over the feature blocks (screen -> greedy ->
   exhaustive over the shortlist) followed by an Optuna model/hyperparameter search over
   logistic, elastic-net, LightGBM, XGBoost, CatBoost, RF/ET/HGB, LSTM/TCN, then stacking with
   isotonic calibration. All selection is nested inside the training window; trial counts and
   multiple-testing corrections are recorded.
4. **Validate** - purged, embargoed, expanding walk-forward. Every OOS prediction ever made is
   appended to `predictions_history.parquet` and its realized outcome backfilled - that file is
   the track record.
5. **Serve** - `data/outputs/` holds the five-artifact contract read by the Mo_Dash Forecast tab:
   `forecasts_latest.parquet`, `predictions_history.parquet`, `scorecard.parquet`,
   `ablation_results.parquet`, `champions.json`.

## Install

```powershell
# into the Mo_Dash venv (recommended)
C:\Dev\Mo_Dash\app\.venv\Scripts\python.exe -m pip install -e C:\Dev\mo_etf_forecaster
```

## Run

```powershell
$py = "C:\Dev\Mo_Dash\app\.venv\Scripts\python.exe"
& $py -m etf_forecaster.pipeline.ingest            # refresh all data sources
& $py -m etf_forecaster.pipeline.build_features    # assemble feature panels
& $py -m etf_forecaster.pipeline.train_search      # ablation + model search + walk-forward
& $py -m etf_forecaster.pipeline.daily_forecast    # today's forecasts + artifact refresh
& $py -m etf_forecaster.reporting.master_report    # regenerate the PDF report
```

`scripts/setup_daily_task.ps1` registers the Windows Task Scheduler jobs
(daily 17:45 ET inference, Saturday retrain, monthly full ablation).

## Configuration

Everything lives in `configs/`:

- `universe.yaml` - tickers, groups, sessions, optionability
- `blocks.yaml` - feature-block -> column-prefix mapping
- `search.yaml` - walk-forward geometry, ablation budgets, model roster
- `macro.yaml` - FRED fast series, ALFRED vintage series with publication lags

Secrets go in `.env` (see `.env.example`); the loader falls back to the Mo_Dash broker env files.
