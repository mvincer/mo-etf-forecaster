"""User-selectable bar intervals and lookbacks for ~N bars of history."""

from __future__ import annotations

from datetime import date, timedelta

# Shown in UI (value -> yfinance / internal)
BAR_INTERVAL_CHOICES: list[str] = ["1m", "5m", "15m", "1h", "1d"]

YAHOO_INTERVAL: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "1d": "1d",
}

# Polygon multiplier + timespan string
POLYGON_AGG: dict[str, tuple[int, str]] = {
    "1m": (1, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "1h": (60, "minute"),
    "1d": (1, "day"),
}


def yahoo_period_for_n_bars(interval: str, n_bars: int = 1000) -> str:
    """Conservative Yahoo `period=` string to (usually) obtain at least n_bars."""
    n = max(n_bars, 100)
    if interval == "1m":
        return "7d"
    if interval == "5m":
        return "60d"
    if interval == "15m":
        return "60d"
    if interval == "1h":
        return "730d"
    if interval == "1d":
        return "max"
    return "60d"


def calendar_lookback_days(interval: str, n_bars: int = 1000) -> int:
    """Calendar days back from today for Polygon range requests."""
    n = max(n_bars, 50)
    if interval == "1d":
        return min(int(n * 1.5) + 10, 365 * 30)
    if interval == "1h":
        return min(int(n / 6.5) + 30, 365 * 5)
    if interval == "15m":
        return min(int((n * 15) / (390 * 0.65)) + 10, 120)
    if interval == "5m":
        return min(int((n * 5) / (390 * 0.65)) + 10, 70)
    if interval == "1m":
        return min(max(5, int(n / (390 * 0.65)) + 3), 10)
    return 60


def start_date_for_polygon(interval: str, n_bars: int = 1000) -> date:
    return date.today() - timedelta(days=calendar_lookback_days(interval, n_bars))


def bar_duration_hours(interval: str) -> float:
    """Hours per bar for half-life → day conversion."""
    m = {"1m": 1 / 60, "5m": 5 / 60, "15m": 15 / 60, "1h": 1.0, "1d": 24.0}
    return m.get(interval, 5 / 60)


def trim_last_n_bars(df, n: int = 1000):
    if df is None or df.empty:
        return df
    if len(df) <= n:
        return df
    return df.iloc[-n:].copy()
