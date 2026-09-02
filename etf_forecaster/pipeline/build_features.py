"""Assemble feature panels for every ticker.

Run:  python -m etf_forecaster.pipeline.build_features [--tickers SPY,QQQ] [--no-markov] [--no-garch]
"""

from __future__ import annotations

import argparse
import logging
import time

from etf_forecaster.config import all_tickers
from etf_forecaster.features import assemble

log = logging.getLogger(__name__)


def run(tickers: list[str] | None = None, *, markov: bool = True, garch: bool = True,
        jobs: int = 8) -> None:
    from joblib import Parallel, delayed

    shared = assemble.load_shared(slow_macro=True)
    tickers = [t for t in (tickers or all_tickers()) if t in shared.bars]

    def _one(t: str) -> str:
        t0 = time.time()
        try:
            panel = assemble.assemble_ticker(t, shared, markov=markov, garch=garch)
            return f"{t}: {panel.shape} in {time.time() - t0:.0f}s"
        except Exception as exc:  # noqa: BLE001
            return f"{t}: FAILED {exc}"

    results = Parallel(n_jobs=jobs, verbose=5)(delayed(_one)(t) for t in tickers)
    for r in results:
        log.info(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="")
    ap.add_argument("--no-markov", action="store_true")
    ap.add_argument("--no-garch", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run([t.strip() for t in args.tickers.split(",") if t.strip()] or None,
        markov=not args.no_markov, garch=not args.no_garch, jobs=args.jobs)


if __name__ == "__main__":
    main()
