"""Print calendar coverage for the FX data repo (raw FX parquets + model_inputs).

Run from the ETF Forecaster repo root::

    py -m fx_data_collect.inspect_fx_dates

Repo root follows ``FX_DATA_REPO_ROOT`` or the default under ``fx_data_collect/data``.

Optional workbook path (first argument) prints the first row of the ``metadata`` sheet if present::

    py -m fx_data_collect.inspect_fx_dates "C:\\...\\daily_binary_fx_forecast.xlsx"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fx_data_collect.repository import resolve_repo_root


def _max_ts(idx: pd.Index) -> str:
    if idx is None or len(idx) == 0:
        return "(empty)"
    try:
        return str(pd.Timestamp(idx.max()))
    except Exception:
        return str(idx.max())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect FX raw/model_inputs date ranges.")
    p.add_argument(
        "xlsx",
        nargs="?",
        default=None,
        help="Optional path to daily_binary_fx_forecast.xlsx (metadata sheet).",
    )
    args = p.parse_args(argv)

    root = resolve_repo_root()
    print(f"FX data repo root: {root}")
    raw_fx = root / "raw" / "fx"
    if not raw_fx.is_dir():
        print(f"No directory {raw_fx}", file=sys.stderr)
        return 1

    paths = sorted(raw_fx.glob("*.parquet"))
    if not paths:
        print(f"No parquet files under {raw_fx}", file=sys.stderr)
        return 1

    print("\n--- raw/fx (last index per pair) ---")
    for path in paths:
        df = pd.read_parquet(path)
        print(f"  {path.name:24}  rows={len(df):6}  index_max={_max_ts(df.index)}")

    for name in ("fundamentals_daily.parquet", "technicals_daily.parquet"):
        mp = root / "model_inputs" / name
        print(f"\n--- model_inputs/{name} ---")
        if not mp.is_file():
            print(f"  (missing: {mp})")
            continue
        df = pd.read_parquet(mp)
        print(f"  rows={len(df):6}  index_max={_max_ts(df.index)}")

    if args.xlsx:
        xp = Path(args.xlsx).expanduser()
        print(f"\n--- workbook metadata: {xp} ---")
        if not xp.is_file():
            print(f"  (missing)", file=sys.stderr)
            return 1
        try:
            meta = pd.read_excel(xp, sheet_name="metadata")
        except Exception as e:
            print(f"  (no metadata sheet: {e})", file=sys.stderr)
            return 1
        if meta.empty:
            print("  (empty metadata sheet)")
        else:
            row = meta.iloc[0]
            for col in meta.columns:
                v = row[col]
                if pd.notna(v) and str(v).strip() != "":
                    print(f"  {col}: {v}")
        try:
            sig = pd.read_excel(xp, sheet_name="daily_signals")
        except Exception:
            sig = None
        if sig is not None and not sig.empty:
            for col in ("target_forecast_date", "latest_feature_date", "input_feature_date"):
                if col not in sig.columns:
                    continue
                s = pd.to_datetime(sig[col], errors="coerce")
                print(f"  daily_signals[{col}] min={s.min()} max={s.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
