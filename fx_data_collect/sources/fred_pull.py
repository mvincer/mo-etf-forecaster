"""FRED observations merged to a daily panel (forward-filled)."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from fx_data_collect.config import FRED_SERIES

logger = logging.getLogger(__name__)


def _fetch_series(series_id: str, api_key: str, observation_start: str) -> pd.Series:
    params = urllib.parse.urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    with urllib.request.urlopen(url, timeout=90) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    obs = payload.get("observations") or []
    rows: list[tuple[pd.Timestamp, float]] = []
    for o in obs:
        d = o.get("date")
        v = o.get("value")
        if v in (".", "", None):
            continue
        try:
            rows.append((pd.Timestamp(d), float(v)))
        except (TypeError, ValueError):
            continue
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    s.index = pd.to_datetime(s.index)
    s.name = series_id
    return s.astype(float)


def download_fred_panel(*, api_key: str | None = None, observation_start: str = "2000-01-01") -> pd.DataFrame:
    key = api_key or os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY missing. Free key: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    frames: list[pd.Series] = []
    for label, sid in FRED_SERIES.items():
        try:
            time.sleep(0.16)
            s = _fetch_series(sid, key, observation_start)
            if s.empty:
                logger.warning("FRED empty %s (%s)", label, sid)
                continue
            s.name = f"fund_{label}"
            frames.append(s)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError) as e:
            logger.warning("FRED skip %s %s: %s", label, sid, e)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=1).sort_index()
    if panel.columns.duplicated().any():
        panel = panel.loc[:, ~panel.columns.duplicated(keep="first")]
    panel = panel.resample("D").last().sort_index().ffill()
    return panel.apply(pd.to_numeric, errors="coerce").astype(float)


def download_fred_panel_reporting(
    series_map: dict[str, str],
    *,
    api_key: str | None = None,
    observation_start: str = "2000-01-01",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Same merged daily panel as :func:`download_fred_panel`, plus per-series rows for manifest reporting.

    Each report dict: label, fred_id, status, start, end, n_obs, native_frequency_note, error
    """
    key = api_key or os.environ.get("FRED_API_KEY", "").strip()
    reports: list[dict[str, Any]] = []
    frames: list[pd.Series] = []

    if not key:
        for label, sid in series_map.items():
            reports.append(
                {
                    "label": label,
                    "fred_id": sid,
                    "column": f"fund_{label}",
                    "source": "fred",
                    "status": "no_api_key",
                    "start": None,
                    "end": None,
                    "n_obs": 0,
                    "frequency": "unknown",
                    "error": "FRED_API_KEY missing",
                },
            )
        return pd.DataFrame(), reports

    for label, sid in series_map.items():
        rep: dict[str, Any] = {
            "label": label,
            "fred_id": sid,
            "column": f"fund_{label}",
            "source": "fred",
            "status": "pending",
            "start": None,
            "end": None,
            "n_obs": 0,
            "frequency": "daily_ffill",
            "error": None,
        }
        try:
            time.sleep(0.16)
            s = _fetch_series(sid, key, observation_start)
            if s.empty:
                rep["status"] = "empty"
                reports.append(rep)
                continue
            rep["status"] = "ok"
            rep["start"] = str(s.index.min())
            rep["end"] = str(s.index.max())
            rep["n_obs"] = int(len(s))
            rep["frequency"] = "native_mixed_ffilled_daily_in_panel"
            s.name = f"fund_{label}"
            frames.append(s)
        except Exception as e:
            rep["status"] = "error"
            rep["error"] = str(e)[:500]
        reports.append(rep)

    if not frames:
        return pd.DataFrame(), reports

    panel = pd.concat(frames, axis=1).sort_index()
    if panel.columns.duplicated().any():
        panel = panel.loc[:, ~panel.columns.duplicated(keep="first")]
    panel = panel.resample("D").last().sort_index().ffill()
    panel = panel.apply(pd.to_numeric, errors="coerce").astype(float)
    return panel, reports
