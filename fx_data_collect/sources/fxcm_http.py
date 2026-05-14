"""Yearly FXCM public candledata (gzip CSV) — same mirror as Currencies ``fxcm_loader``."""

from __future__ import annotations

import gzip
import io
import logging
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

_CANDLEDATA_TMPL = "https://candledata.fxcorporate.com/D1/{sym}/{year}.csv.gz"
_HTTP_HEADERS = {"User-Agent": "ETF-Forecaster-fx_data_collect/1.0 (pandas; research)"}


def candledata_symbol(instrument: str) -> str:
    return instrument.replace("/", "").replace(" ", "").strip()


def _csv_rows_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip(): c for c in df.columns}
    time_col = cols.get("DateTime") or cols.get("Time") or cols.get("Date")
    if time_col is None:
        raise ValueError(f"No time column in candledata CSV: {list(df.columns)}")
    idx = pd.to_datetime(df[time_col], errors="coerce")
    bo = cols.get("BidOpen") or cols.get("Open")
    bh = cols.get("BidHigh") or cols.get("High")
    bl = cols.get("BidLow") or cols.get("Low")
    bc = cols.get("BidClose") or cols.get("Close")
    if bc is None:
        raise ValueError("No bid/close column")
    o = pd.to_numeric(df[bo], errors="coerce") if bo else pd.to_numeric(df[bc], errors="coerce")
    h = pd.to_numeric(df[bh], errors="coerce") if bh else pd.to_numeric(df[bc], errors="coerce")
    l = pd.to_numeric(df[bl], errors="coerce") if bl else pd.to_numeric(df[bc], errors="coerce")
    c = pd.to_numeric(df[bc], errors="coerce")
    out = pd.DataFrame(
        {"Open": o.to_numpy(), "High": h.to_numpy(), "Low": l.to_numpy(), "Close": c.to_numpy()},
        index=idx,
    )
    out.index.name = None
    vol_col = cols.get("tickqty") or cols.get("TickQty") or cols.get("Volume")
    if vol_col:
        out["Volume"] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0).to_numpy()
    else:
        out["Volume"] = 0.0
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index().dropna(subset=["Close"])


def download_instrument_history_http(
    instrument: str,
    *,
    min_year: int = 2000,
    max_year: int | None = None,
) -> pd.DataFrame:
    sym = candledata_symbol(instrument)
    end_year = max_year if max_year is not None else datetime.now(timezone.utc).year
    parts: list[pd.DataFrame] = []

    for year in range(min_year, end_year + 1):
        url = _CANDLEDATA_TMPL.format(sym=sym, year=year)
        req = urllib.request.Request(url, headers=_HTTP_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = gzip.decompress(resp.read())
            chunk = pd.read_csv(io.BytesIO(raw))
            parts.append(_csv_rows_to_ohlcv(chunk))
        except urllib.error.HTTPError as e:
            if e.code != 404:
                logger.warning("candledata HTTP %s %s: %s", sym, year, e)
        except Exception as e:
            logger.warning("candledata read fail %s %s: %s", sym, year, e)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True)).tz_localize(None).normalize()
    return out


def parquet_symbol(instrument: str) -> str:
    return re.sub(r"[^\w]+", "_", instrument.strip()).strip("_")
