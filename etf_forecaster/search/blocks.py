"""Feature-column selection by block set and lag-window size."""

from __future__ import annotations

import re

from etf_forecaster.config import block_of_column

_LAG_RE = re.compile(r"^px_lag_(?:ret|rng)_(\d+)$")


def select_columns(all_cols: list[str], blocks: set[str] | list[str], *,
                   window: int = 20) -> list[str]:
    """Columns whose block is PRICE (always on) or in `blocks`; the raw candle-lag
    columns are truncated to the requested window X."""
    want = set(blocks) | {"PRICE"}
    out: list[str] = []
    for c in all_cols:
        b = block_of_column(c)
        if b is None or b not in want:
            continue
        m = _LAG_RE.match(c)
        if m and int(m.group(1)) > window:
            continue
        out.append(c)
    return out
