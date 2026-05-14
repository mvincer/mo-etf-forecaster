"""
Build merged **fundamentals** + **technicals** model input tables.

  py -m fx_data_collect.run_model_inputs

Requires: prior ``py -m fx_data_collect.run_collect`` (FX parquets) and ``FRED_API_KEY`` in environment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "fx_data_collect" / ".env")
load_dotenv(_ROOT / "project" / ".env")
# Same key as Currencies app (common layout: sibling folder)
load_dotenv(_ROOT.parent / "Currencies" / ".env")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build fundamentals + technicals from FRED, Yahoo, cached FX")
    p.add_argument("--repo-root", type=Path, default=None, help="Override FX_DATA_REPO_ROOT")
    p.add_argument("--start", default="2000-01-01", help="FRED observation start + Yahoo history start")
    args = p.parse_args(argv)

    from fx_data_collect.build_model_inputs import build_model_inputs

    build_model_inputs(repo_root=args.repo_root, observation_start=str(args.start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
