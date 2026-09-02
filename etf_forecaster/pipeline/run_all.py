"""End-to-end daily pipeline: ingest -> features -> daily forecast -> PDF -> email.

Run:  python -m etf_forecaster.pipeline.run_all [--skip-ingest] [--retrain]
"""

from __future__ import annotations

import argparse
import logging

log = logging.getLogger(__name__)


def run(*, skip_ingest: bool = False, retrain: bool = False, full_search: bool = False,
        email: bool = True, jobs: int = 8) -> None:
    if not skip_ingest:
        from etf_forecaster.pipeline import ingest
        ingest.run()

    from etf_forecaster.pipeline import build_features
    build_features.run(jobs=jobs)

    if retrain:
        from etf_forecaster.pipeline import train_search
        train_search.run(fast=not full_search, jobs=jobs)

    from etf_forecaster.pipeline import daily_forecast
    latest = daily_forecast.run(jobs=jobs)

    try:
        from etf_forecaster.reporting import master_report
        master_report.generate()
    except Exception as exc:  # noqa: BLE001
        log.warning("PDF report failed: %s", exc)

    if email:
        try:
            _send_summary(latest)
        except Exception as exc:  # noqa: BLE001
            log.warning("email summary failed: %s", exc)


def _send_summary(latest) -> None:
    import os
    import smtplib
    from email.mime.text import MIMEText

    from etf_forecaster.config import load_env
    load_env()
    user = os.environ.get("GMAIL_USER", "")
    pwd = os.environ.get("GMAIL_APP_PASSWORD", "")
    to = os.environ.get("EMAIL_TO", user)
    if not user or not pwd:
        log.info("no GMAIL_USER/GMAIL_APP_PASSWORD; skipping email")
        return
    top = latest[latest["horizon"] == 1].sort_values("p_up")
    lines = ["ETF Forecaster - daily summary", ""]
    lines.append("Most bearish (h=1): " + ", ".join(
        f"{r.ticker} {r.p_up:.0%}" for r in top.head(5).itertuples()))
    lines.append("Most bullish (h=1): " + ", ".join(
        f"{r.ticker} {r.p_up:.0%}" for r in top.tail(5).itertuples()))
    msg = MIMEText("\n".join(lines))
    msg["Subject"] = "ETF Forecaster daily"
    msg["From"] = user
    msg["To"] = to
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pwd)
        s.send_message(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--full-search", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(skip_ingest=args.skip_ingest, retrain=args.retrain, full_search=args.full_search,
        email=not args.no_email, jobs=args.jobs)


if __name__ == "__main__":
    main()
