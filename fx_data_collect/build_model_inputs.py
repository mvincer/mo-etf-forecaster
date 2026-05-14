"""
Build **fundamentals_daily** (FRED + Yahoo macro OHLC) and **technicals_daily**
(same indicator set as Currencies + weekly ``*_wk_*`` columns), aligned to a **calendar-daily** index.

Writes manifest JSON + ``MANIFEST_REPORT.md``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fx_data_collect.catalog import EXPANDED_FRED_SERIES, YAHOO_MARKETS
from fx_data_collect.config import FX_PAIRS, START_DATE
from fx_data_collect.repository import RawRepository, resolve_repo_root
from fx_data_collect.sources.fxcm_http import parquet_symbol
from fx_data_collect.sources.fred_pull import download_fred_panel_reporting
from fx_data_collect.sources.yahoo_macro import download_all_macro
from fx_data_collect.technicals import compute_technical_features, weekly_technicals_ffill_to_daily

logger = logging.getLogger(__name__)


def _merge_fred_dict() -> dict[str, str]:
    """Merge config + expanded catalog; **drop duplicate FRED IDs** (keep first label per series_id)."""
    from fx_data_collect.config import FRED_SERIES

    merged_order = {**EXPANDED_FRED_SERIES, **FRED_SERIES}
    seen_ids: set[str] = set()
    out: dict[str, str] = {}
    for lab, sid in merged_order.items():
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        out[lab] = sid
    return out


def _calendar_index(
    fx_frames: dict[str, pd.DataFrame],
    macro_frames: dict[str, pd.DataFrame],
    fred_index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for df in fx_frames.values():
        if len(df):
            mins.append(pd.Timestamp(df.index.min()))
            maxs.append(pd.Timestamp(df.index.max()))
    for df in macro_frames.values():
        if len(df):
            mins.append(pd.Timestamp(df.index.min()))
            maxs.append(pd.Timestamp(df.index.max()))
    if len(fred_index):
        mins.append(pd.Timestamp(fred_index.min()))
        maxs.append(pd.Timestamp(fred_index.max()))
    if not mins:
        return pd.DatetimeIndex([])
    lo = min(mins).normalize()
    hi = max(maxs).normalize()
    return pd.date_range(lo, hi, freq="D")


def _fill_missing_fred_from_cache(
    fred_panel: pd.DataFrame,
    fred_reports: list[dict[str, Any]],
    *,
    fred_map: dict[str, str],
    cached_path: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Reuse ``fund_*`` columns from a previous fundamentals parquet for any series that the FRED API
    couldn't return (no key, network error, or empty response). The cached values are merged into
    ``fred_panel`` and will be forward-filled to the new calendar index by the caller — i.e., the user's
    requested cascade ``FRED → cached → ffill`` for FRED features.
    """
    expected = {f"fund_{lab}" for lab in fred_map}
    present = set(map(str, fred_panel.columns)) if not fred_panel.empty else set()
    missing = expected - present
    if not missing:
        return fred_panel, fred_reports

    if not cached_path.exists():
        for r in fred_reports:
            if r.get("column") in missing and r.get("status") in ("no_api_key", "error", "empty"):
                r["error"] = (r.get("error") or "") + " | no cached fundamentals available"
        return fred_panel, fred_reports

    try:
        cached = pd.read_parquet(cached_path)
    except Exception as e:
        logger.warning("Could not read cached fundamentals %s: %s", cached_path, e)
        return fred_panel, fred_reports

    fill_cols = [c for c in missing if c in cached.columns]
    if not fill_cols:
        return fred_panel, fred_reports

    cached_block = cached[fill_cols].copy()
    cached_block.index = pd.DatetimeIndex(pd.to_datetime(cached_block.index))
    if fred_panel.empty:
        fred_panel = cached_block
    else:
        fred_panel = pd.concat([fred_panel, cached_block], axis=1).sort_index()
        fred_panel = fred_panel.loc[:, ~fred_panel.columns.duplicated(keep="first")]

    cache_end = str(cached_block.dropna(how="all").index.max()) if len(cached_block) else None
    for r in fred_reports:
        col = r.get("column")
        if col in fill_cols and r.get("status") in ("no_api_key", "error", "empty", "pending"):
            r["status"] = "cached_fallback"
            r["frequency"] = "cached_ffill"
            r["end"] = cache_end
            r["error"] = (
                "FRED unavailable; reused last cached value for this series and "
                "forward-filled to current calendar."
            )
    logger.warning(
        "FRED fallback: reused %d cached fund_* column(s) from %s (last cached date %s).",
        len(fill_cols),
        cached_path,
        cache_end,
    )
    return fred_panel, fred_reports


def _load_fx_parquets(raw_fx_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for pair in FX_PAIRS:
        path = raw_fx_dir / f"{parquet_symbol(pair)}.parquet"
        if not path.exists():
            logger.warning("Missing FX parquet (run run_collect first): %s", path)
            continue
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).tz_localize(None).normalize()
        out[pair] = df.sort_index()
    return out


def build_model_inputs(
    *,
    repo_root: Path | None = None,
    observation_start: str = START_DATE,
) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root else resolve_repo_root()
    repo = RawRepository(root)
    out_dir = repo.root / "model_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fred_map = _merge_fred_dict()
    fred_panel, fred_reports = download_fred_panel_reporting(
        fred_map,
        observation_start=observation_start,
    )

    cached_fund_path = out_dir / "fundamentals_daily.parquet"
    fred_panel, fred_reports = _fill_missing_fred_from_cache(
        fred_panel,
        fred_reports,
        fred_map=fred_map,
        cached_path=cached_fund_path,
    )

    macro_frames, yahoo_reports = download_all_macro(YAHOO_MARKETS, start=observation_start)

    fx_frames = _load_fx_parquets(repo.raw_fx)

    idx = _calendar_index(fx_frames, macro_frames, fred_panel.index if len(fred_panel) else pd.DatetimeIndex([]))
    if len(idx) == 0:
        raise RuntimeError("No data to align — run fx_data_collect.run_collect first for FX parquets.")

    # --- Fundamentals ---
    fund_parts: list[pd.DataFrame] = []
    if len(fred_panel.columns):
        fp = fred_panel.reindex(idx).ffill()
        fund_parts.append(fp)

    for lab, df in macro_frames.items():
        block = pd.DataFrame(index=idx)
        for c in df.columns:
            block[f"mkt_{lab}_{str(c).lower()}"] = df[c].reindex(idx).ffill()
        fund_parts.append(block)

    fundamentals = pd.concat(fund_parts, axis=1) if fund_parts else pd.DataFrame(index=idx)
    fundamentals = fundamentals.loc[:, ~fundamentals.columns.duplicated()]

    # --- Technicals (daily + weekly ffill) ---
    tech_blocks: list[pd.DataFrame] = []

    for pair, ohlc in fx_frames.items():
        pid = parquet_symbol(pair)
        if ohlc.empty or len(ohlc) < 20:
            continue
        try:
            td = compute_technical_features(ohlc, prefix=f"{pid}_").reindex(idx)
            tw = weekly_technicals_ffill_to_daily(ohlc, prefix=f"{pid}_wk_", daily_index=idx)
            tech_blocks.extend([td, tw])
        except Exception as e:
            logger.warning("Technicals failed %s: %s", pair, e)

    for lab, ohlc in macro_frames.items():
        if ohlc.empty or len(ohlc) < 20:
            continue
        try:
            td = compute_technical_features(ohlc, prefix=f"{lab}_").reindex(idx)
            tw = weekly_technicals_ffill_to_daily(ohlc, prefix=f"{lab}_wk_", daily_index=idx)
            tech_blocks.extend([td, tw])
        except Exception as e:
            logger.warning("Technicals failed market %s: %s", lab, e)

    technicals = pd.concat(tech_blocks, axis=1) if tech_blocks else pd.DataFrame(index=idx)
    technicals = technicals.loc[:, ~technicals.columns.duplicated()]

    f_path = out_dir / "fundamentals_daily.parquet"
    t_path = out_dir / "technicals_daily.parquet"
    fundamentals.to_parquet(f_path)
    technicals.to_parquet(t_path)

    # --- Manifest ---
    summary: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "observation_start": observation_start,
        "calendar_index": {
            "start": str(idx.min().date()),
            "end": str(idx.max().date()),
            "n_days": int(len(idx)),
            "frequency": "calendar_daily",
        },
        "outputs": {
            "fundamentals_daily": str(f_path),
            "technicals_daily": str(t_path),
        },
        "fred_series_requested": len(fred_map),
        "fred_series_reports": fred_reports,
        "yahoo_symbols_requested": len(YAHOO_MARKETS),
        "yahoo_reports": yahoo_reports,
        "fx_pairs_loaded": list(fx_frames.keys()),
        "technicals_columns": int(technicals.shape[1]),
        "fundamentals_columns": int(fundamentals.shape[1]),
    }

    json_path = out_dir / "manifest.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = out_dir / "MANIFEST_REPORT.md"
    md_path.write_text(_render_markdown(summary), encoding="utf-8")

    logger.info("Wrote %s, %s, %s, %s", f_path, t_path, json_path, md_path)
    return summary


def _render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Model inputs manifest\n")
    lines.append(f"- Generated (UTC): **{summary.get('generated_utc', '')}**\n")
    lines.append(f"- Observation start: **{summary.get('observation_start', '')}**\n")
    cal = summary.get("calendar_index") or {}
    lines.append(
        f"- Calendar index: **{cal.get('start')}** → **{cal.get('end')}** "
        f"({cal.get('n_days')} days, **{cal.get('frequency')}**)\n",
    )
    lines.append(f"- Fundamentals columns: **{summary.get('fundamentals_columns')}**  \n")
    lines.append(f"- Technicals columns: **{summary.get('technicals_columns')}**  \n")
    lines.append("\n## FRED series\n")
    lines.append("| Label | FRED ID | Status | Start | End | Obs | Notes |\n")
    lines.append("|-------|---------|--------|-------|-----|-----|-------|\n")
    for r in summary.get("fred_series_reports") or []:
        err = (r.get("error") or "").replace("|", "/")
        lines.append(
            f"| {r.get('label')} | {r.get('fred_id')} | {r.get('status')} | "
            f"{r.get('start') or ''} | {r.get('end') or ''} | {r.get('n_obs')} | {r.get('frequency', '')} {err} |\n",
        )

    lines.append("\n## Yahoo Finance (daily OHLC)\n")
    lines.append("| Label | Ticker | Status | Start | End | Rows | Error |\n")
    lines.append("|-------|--------|--------|-------|-----|------|-------|\n")
    for r in summary.get("yahoo_reports") or []:
        err = (r.get("error") or "").replace("|", "/")
        lines.append(
            f"| {r.get('label')} | {r.get('yahoo_ticker')} | {r.get('status')} | "
            f"{r.get('start') or ''} | {r.get('end') or ''} | {r.get('rows')} | {err} |\n",
        )

    lines.append("\n## FX parquets used\n")
    for p in summary.get("fx_pairs_loaded") or []:
        lines.append(f"- {p}\n")

    lines.append("\n## Could not retrieve or empty\n")
    bad_fred = [r for r in (summary.get("fred_series_reports") or []) if r.get("status") != "ok"]
    bad_y = [r for r in (summary.get("yahoo_reports") or []) if r.get("status") != "ok"]
    for r in bad_fred:
        lines.append(
            f"- **FRED** `{r.get('label')}` ({r.get('fred_id')}): **{r.get('status')}** — {r.get('error') or 'see table'}\n",
        )
    for r in bad_y:
        lines.append(
            f"- **Yahoo** `{r.get('label')}` ({r.get('yahoo_ticker')}): **{r.get('status')}** — {r.get('error') or 'see table'}\n",
        )
    if not bad_fred and not bad_y:
        lines.append("- (none — all returned data)\n")

    lines.append("\n## Frequency notes\n")
    lines.append(
        "- **FRED**: native publication varies (daily / weekly / monthly). "
        "Panel stores **calendar daily** rows with **forward-fill** after merge (same approach as Currencies).\n"
    )
    lines.append(
        "- **Yahoo**: **daily** bars (exchange calendar; rows aligned to **calendar daily** index with ffill).\n",
    )
    lines.append(
        "- **Technicals**: **daily** from raw OHLC; **weekly** resample `W-FRI` then ffill to calendar (`*_wk_*`).\n",
    )
    return "".join(lines)
