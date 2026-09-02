"""Daily data refresh: prices, vol indices, cross series, FRED fast, ALFRED vintages
(weekly), Shiller (weekly), IBKR IV, option-chain snapshot, skew history, calendar.

Run:  python -m etf_forecaster.pipeline.ingest [--full] [--skip-chains] [--skip-ibkr]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from etf_forecaster.config import chain_tickers, ensure_dirs

log = logging.getLogger(__name__)


def run(*, full: bool = False, skip_chains: bool = False, skip_ibkr: bool = False,
        weekly: bool | None = None) -> None:
    ensure_dirs()
    weekly = weekly if weekly is not None else (date.today().weekday() == 5 or full)

    from etf_forecaster.data import yahoo
    rep = yahoo.refresh_bars(incremental=not full)
    failed = rep[rep["rows"] == 0].index.tolist()
    if failed:
        log.warning("price ingest failures: %s", failed)

    from etf_forecaster.data import fred
    try:
        fred.refresh_fast_panel()
    except Exception as exc:  # noqa: BLE001
        log.warning("FRED fast refresh failed: %s", exc)

    if weekly:
        from etf_forecaster.data import alfred, shiller
        try:
            alfred.refresh_vintages()
        except Exception as exc:  # noqa: BLE001
            log.warning("ALFRED refresh failed: %s", exc)
        try:
            shiller.refresh_shiller()
        except Exception as exc:  # noqa: BLE001
            log.warning("Shiller refresh failed: %s", exc)

    from etf_forecaster.data.calendar import build_calendar
    try:
        bars = yahoo.load_bars("SPY")
        if bars is not None:
            build_calendar(bars.index)
    except Exception as exc:  # noqa: BLE001
        log.warning("calendar rebuild failed: %s", exc)

    if not skip_ibkr:
        from etf_forecaster.data import ibkr_vol
        try:
            ibkr_vol.refresh_ibkr_vol()
        except Exception as exc:  # noqa: BLE001
            log.warning("IBKR vol refresh failed: %s", exc)

    if not skip_chains:
        from etf_forecaster.data import chains
        from etf_forecaster.data.fred import load_fast_panel
        from etf_forecaster.options.surface import build_skew_history
        try:
            chains.snapshot_all()
        except Exception as exc:  # noqa: BLE001
            log.warning("chain snapshot failed: %s", exc)
        panel = load_fast_panel()
        r = panel["us_3m"] if panel is not None and "us_3m" in panel else None
        for t in chain_tickers():
            try:
                build_skew_history(t, rate_series=r)
            except Exception as exc:  # noqa: BLE001
                log.warning("skew history failed for %s: %s", t, exc)
    log.info("ingest complete")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--skip-chains", action="store_true")
    ap.add_argument("--skip-ibkr", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(full=args.full, skip_chains=args.skip_chains, skip_ibkr=args.skip_ibkr)


if __name__ == "__main__":
    main()
