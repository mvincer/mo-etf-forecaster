"""Per-currency long/short counts from pair-level signals (base = first, quote = second)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from utils.fx_signal_side import classify_signal_side

# Pairs used to find a tradable instrument for a currency bias (same universe as forecasts).
_PAIR_UNIVERSE: tuple[str, ...] = (
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "NZD/USD",
    "USD/CAD",
    "EUR/JPY",
    "EUR/GBP",
    "GBP/JPY",
)

# When a currency is **base** in several pairs, prefer the most liquid for execution.
_PREFERRED_PAIR_AS_BASE: dict[str, str] = {
    "EUR": "EUR/USD",
    "GBP": "GBP/USD",
    "AUD": "AUD/USD",
    "NZD": "NZD/USD",
    "USD": "USD/JPY",
}


def parse_fx_pair(primary: str) -> tuple[str, str] | None:
    """Return ``(base_ccy, quote_ccy)`` for ``EUR/USD`` or ``EURUSD``."""
    s = primary.strip().upper().replace(" ", "")
    if "/" in s:
        a, b = s.split("/", 1)
        if len(a) == 3 and len(b) == 3 and a.isalpha() and b.isalpha():
            return a, b
    s = s.replace("_", "")
    if len(s) == 6 and s.isalpha():
        return s[:3], s[3:]
    return None


def currency_leg_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each signal row is a **pair** side: buy = long base + short quote; sell = short base + long quote.

    Returns columns: ``currency``, ``n_long``, ``n_short``, ``n_total``, ``score``, ``dominant``.
    ``score`` = (n_long - n_short) / n_total, or NA if n_total == 0.
    ``dominant`` is ``long`` / ``short`` / ``tie`` / ``none``.
    """
    if df.empty or "primary" not in df.columns:
        return pd.DataFrame(
            columns=["currency", "n_long", "n_short", "n_total", "score", "dominant"],
        )
    sides = classify_signal_side(df)
    long_c: dict[str, int] = defaultdict(int)
    short_c: dict[str, int] = defaultdict(int)
    for i in df.index:
        pr = str(df.loc[i, "primary"])
        pq = parse_fx_pair(pr)
        if pq is None:
            continue
        base, quote = pq
        side = sides.get(i, pd.NA)
        if side == "buy":
            long_c[base] += 1
            short_c[quote] += 1
        elif side == "sell":
            short_c[base] += 1
            long_c[quote] += 1
    ccys = sorted(set(long_c) | set(short_c))
    rows = []
    for c in ccys:
        nl = int(long_c[c])
        ns = int(short_c[c])
        nt = nl + ns
        if nt == 0:
            score = float("nan")
            dom = "none"
        else:
            score = (nl - ns) / nt
            if nl > ns:
                dom = "long"
            elif ns > nl:
                dom = "short"
            else:
                dom = "tie"
        rows.append({"currency": c, "n_long": nl, "n_short": ns, "n_total": nt, "score": score, "dominant": dom})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["n_total", "currency"], ascending=[False, True], kind="stable").reset_index(drop=True)


def express_currency_bias_as_pair(*, currency: str, want_long_ccy: bool) -> tuple[str, bool] | None:
    """
    Map **long this currency** / **short this currency** to a **pair** and **buy/sell that pair**.

    - If currency is **base** of some pair in the universe: long ccy → buy pair; short ccy → sell pair.
    - If currency is **only** a quote: long ccy → **sell** the pair; short ccy → **buy** the pair.
    """
    ccy = currency.strip().upper()
    if len(ccy) != 3:
        return None
    as_base = [p for p in _PAIR_UNIVERSE if parse_fx_pair(p) and parse_fx_pair(p)[0] == ccy]
    if as_base:
        pref = _PREFERRED_PAIR_AS_BASE.get(ccy)
        if pref and pref in as_base:
            p = pref
        else:
            p = sorted(as_base)[0]
        return (p, want_long_ccy)
    as_quote = [p for p in _PAIR_UNIVERSE if parse_fx_pair(p) and parse_fx_pair(p)[1] == ccy]
    if as_quote:
        p = sorted(as_quote)[0]
        return (p, not want_long_ccy)
    return None


@dataclass(frozen=True)
class PlannedOrder:
    primary: str
    is_buy: bool
    currency: str
    dominant: str


def plan_equal_bias_orders(currency_table: pd.DataFrame) -> list[PlannedOrder]:
    """One order per currency with a non-tie dominant bias."""
    out: list[PlannedOrder] = []
    if currency_table.empty:
        return out
    for _, row in currency_table.iterrows():
        dom = str(row.get("dominant", ""))
        ccy = str(row.get("currency", "")).strip().upper()
        if dom not in ("long", "short") or len(ccy) != 3:
            continue
        want_long = dom == "long"
        plan = express_currency_bias_as_pair(currency=ccy, want_long_ccy=want_long)
        if plan is None:
            continue
        primary, is_buy = plan
        out.append(PlannedOrder(primary=primary, is_buy=is_buy, currency=ccy, dominant=dom))
    return out
