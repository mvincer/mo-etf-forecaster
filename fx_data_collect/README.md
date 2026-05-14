# fx_data_collect — raw FX + macro repository

Collect **only raw** daily data you control:

1. **FX** — FXCM public yearly candledata (bid OHLC), then **Yahoo Finance** (`*=X`) fills any missing tail through **today**.
2. **Macro** — **FRED** rates / yields / indices from `2000-01-01` (daily panel, forward-filled).

Outputs under `data/` (or `FX_DATA_REPO_ROOT`):

| Path | Contents |
|------|----------|
| `raw/fx/*.parquet` | OHLCV per pair |
| `raw/fred/fred_panel.parquet` | All `fund_*` columns |
| `raw/manifest.json` | Row counts, date spans, sources |
| `raw/repository_snapshot.pkl` | Dict `fx` + `fred` for notebooks |
| `preview/collection_summary.csv` | One row per series |
| `preview/merged_preview_*_tail.csv` | EUR/USD (or `--primary-preview`) + sample FRED |
| `preview/collection_preview.png` | Quick matplotlib chart |

## Run

```powershell
cd "...\ETF Forecaster"
pip install -r fx_data_collect/requirements.txt
# copy fx_data_collect\.env.example to repo root .env or fx_data_collect\.env — add FRED_API_KEY
py -m fx_data_collect.run_collect
```

FX-only (no FRED key):

```powershell
py -m fx_data_collect.run_collect --skip-fred
```

## Model inputs (FRED + Yahoo macro + technicals)

After `run_collect`, build aligned **fundamentals** + **technicals** (same indicator library as **Currencies** `technical_indicators.py`, plus weekly `*_wk_*` columns):

```powershell
py -m fx_data_collect.run_model_inputs
```

Outputs under `data/model_inputs/`:

- `fundamentals_daily.parquet` — `fund_*` (FRED) + `mkt_*` (Yahoo OHLCV-derived columns)
- `technicals_daily.parquet` — daily + weekly technical columns per FX pair and each Yahoo macro symbol
- `manifest.json` — machine-readable report
- `MANIFEST_REPORT.md` — **which series succeeded, date ranges, frequency notes, and failures**

Configure extra series in `catalog.py` (`EXPANDED_FRED_SERIES`, `YAHOO_MARKETS`).

## Later: extend

- **Row-wise**: rerun `run_collect` — Yahoo tail reaches current calendar.
- **Column-wise**: add IDs to `catalog.EXPANDED_FRED_SERIES` / `YAHOO_MARKETS`.
