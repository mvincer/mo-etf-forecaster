"""Causal rolling-window pivots (neurotrader888/TechnicalAnalysisAutomation).

A swing at bar k is only known at confirmation bar k+order.
Never use centered scipy.argrelextrema for backtests.
"""

# Vendored from Mo_Dash strategies/chart_patterns/pivots.py - keep in sync manually.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Pivot:
    confirm_i: int   # bar when pivot becomes known
    idx: int         # bar of the extreme
    price: float
    kind: int        # +1 top, -1 bottom


def rw_top(data: np.ndarray, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            return False
    return True


def rw_bottom(data: np.ndarray, curr_index: int, order: int) -> bool:
    if curr_index < order * 2 + 1:
        return False
    k = curr_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            return False
    return True


def extract_pivots(data: np.ndarray, order: int) -> List[Pivot]:
    """Scan forward; each pivot is stamped at its confirmation bar."""
    out: List[Pivot] = []
    for i in range(len(data)):
        if rw_top(data, i, order):
            out.append(Pivot(confirm_i=i, idx=i - order, price=float(data[i - order]), kind=1))
        if rw_bottom(data, i, order):
            out.append(Pivot(confirm_i=i, idx=i - order, price=float(data[i - order]), kind=-1))
    out.sort(key=lambda p: (p.confirm_i, p.idx))
    return out


def alternating_pivots(pivots: List[Pivot]) -> List[Pivot]:
    """Keep alternating top/bottom sequence (drop consecutive same-kind)."""
    if not pivots:
        return []
    alt = [pivots[0]]
    for p in pivots[1:]:
        if p.kind == alt[-1].kind:
            # keep the more extreme of the two
            if p.kind == 1 and p.price >= alt[-1].price:
                alt[-1] = p
            elif p.kind == -1 and p.price <= alt[-1].price:
                alt[-1] = p
        else:
            alt.append(p)
    return alt
