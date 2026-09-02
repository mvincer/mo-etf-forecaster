"""Shared signal dataclass for chart patterns."""

# Vendored from Mo_Dash strategies/chart_patterns/pattern_signal.py - keep in sync manually.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class PatternSignal:
    """A confirmed chart pattern ready for backtesting."""

    pattern: str
    direction: str          # "buy" | "sell"
    pair: str
    timeframe: str
    confirm_idx: int        # bar index where pattern confirmed (neckline break)
    confirm_ts: pd.Timestamp
    entry_price: float
    sl: float
    tp: float
    neckline: float
    pattern_high: float
    pattern_low: float
    swing_indices: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.sl)

    @property
    def signal_id(self) -> str:
        ts = self.confirm_ts.strftime("%Y%m%d%H%M")
        return f"{self.pair.replace('/', '')}_{self.timeframe}_{self.pattern}_{ts}"
