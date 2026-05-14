"""
Collect raw FX (FXCM candledata + Yahoo tail) and FRED macro panel.

Usage (from repo root):

  cd \"...\\ETF Forecaster\"
  py -m fx_data_collect.run_collect

Env:
  FRED_API_KEY   — required unless --skip-fred
  FX_DATA_REPO_ROOT — optional; default ./fx_data_collect/data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load .env from project roots (parent = ETF Forecaster, sibling dotfiles OK)
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE.parent / ".env")
load_dotenv(_HERE.parent / "project" / ".env")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Collect raw FX + FRED into local repository")
    p.add_argument("--skip-fred", action="store_true", help="FX only (no FRED_API_KEY needed)")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override FX_DATA_REPO_ROOT / default package data folder",
    )
    p.add_argument("--primary-preview", default="EUR/USD", help="Pair for merged CSV/chart")
    p.add_argument("--no-chart", action="store_true")
    args = p.parse_args(argv)

    from fx_data_collect.config import FX_PAIRS, START_DATE, YAHOO_FX_MAP
    from fx_data_collect.preview import merged_preview_sample, summary_table, write_preview_chart
    from fx_data_collect.repository import RawRepository, collect_all_fx, collect_fred_optional

    repo = RawRepository(args.repo_root.resolve()) if args.repo_root else RawRepository.from_env()

    logging.info("Repository root: %s", repo.root)
    logging.info("Collecting FX from %s through Yahoo tail merge …", START_DATE)

    fx_frames, fx_meta = collect_all_fx(FX_PAIRS, YAHOO_FX_MAP, repo=repo, min_date=START_DATE)

    try:
        fred, fred_meta = collect_fred_optional(repo, skip=args.skip_fred)
    except RuntimeError as e:
        logging.error("%s", e)
        logging.error("Use --skip-fred for FX-only, or set FRED_API_KEY.")
        fred, fred_meta = pd.DataFrame(), {"skipped": True, "error": str(e)}

    manifest = {
        "start_date": START_DATE,
        "pairs": FX_PAIRS,
        "fx": fx_meta,
        "fred": fred_meta,
    }
    repo.write_manifest(manifest)

    if len(fred.columns):
        repo.pickle_snapshot(fx_frames, fred)
    else:
        repo.pickle_snapshot(fx_frames, pd.DataFrame())

    summ = summary_table(fx_meta, fred_meta if isinstance(fred_meta, dict) else {})
    summ_path = repo.preview / "collection_summary.csv"
    summ.to_csv(summ_path, index=False)
    logging.info("Wrote %s", summ_path)

    primary = args.primary_preview if args.primary_preview in fx_frames else FX_PAIRS[0]
    merged = merged_preview_sample(fx_frames, fred, primary_pair=primary, last_n=3000)
    merged_path = repo.preview / f"merged_preview_{primary.replace('/', '_')}_tail.csv"
    merged.to_csv(merged_path)
    logging.info("Wrote %s", merged_path)

    if not args.no_chart:
        try:
            png = repo.preview / "collection_preview.png"
            write_preview_chart(
                merged,
                png,
                title=f"Raw preview (tail): {primary} Close + sample FRED — ends {merged.index.max().date()}",
            )
            logging.info("Wrote %s", png)
        except RuntimeError as e:
            logging.warning("%s", e)

    # Echo newest dates
    for pair in FX_PAIRS[:3]:
        end = fx_meta[pair].get("end")
        logging.info("%s last bar: %s", pair, end)

    logging.info("Done. Snapshot: %s", repo.root / "raw" / "repository_snapshot.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
