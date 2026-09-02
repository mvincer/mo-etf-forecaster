"""Swing high/low detection and ATR helpers."""

# Vendored from Mo_Dash strategies/chart_patterns/swing_points.py - keep in sync manually.

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def find_swing_highs(high: np.ndarray, order: int) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    n = len(high)
    for i in range(order, n - order):
        window = high[i - order : i + order + 1]
        if high[i] >= window.max() and high[i] > high[i - 1] and high[i] > high[i + 1]:
            out.append((i, float(high[i])))
    return out


def find_swing_lows(low: np.ndarray, order: int) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    n = len(low)
    for i in range(order, n - order):
        window = low[i - order : i + order + 1]
        if low[i] <= window.min() and low[i] < low[i - 1] and low[i] < low[i + 1]:
            out.append((i, float(low[i])))
    return out


def price_near(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol
