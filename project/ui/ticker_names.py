"""Display names for tickers — avoid slow yfinance `.info` (can hang the Charts tab)."""

from __future__ import annotations

import yfinance as yf


def get_display_name(ticker: str) -> str:
    t = ticker.strip().upper()
    try:
        tk = yf.Ticker(t)
        fi = getattr(tk, "fast_info", None)
        if fi is not None:
            if isinstance(fi, dict):
                for key in ("shortName", "longName", "name"):
                    v = fi.get(key)
                    if v and isinstance(v, str):
                        return v.strip()
            else:
                for key in ("shortName", "longName", "name"):
                    v = getattr(fi, key, None)
                    if v and isinstance(v, str):
                        return v.strip()
    except Exception:
        pass
    return t
