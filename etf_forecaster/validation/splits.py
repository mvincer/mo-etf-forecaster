"""Purged, embargoed, expanding walk-forward split geometry.

Rows are positions in a time-ordered panel. For a refit at position s forecasting
positions [s, s+step), training may use rows [0, s - h) only: row s-h is the last row
whose h-bar forward label completes strictly before the first forecast, which both
purges overlapping labels and embargoes h bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Fold:
    train_end: int      # exclusive
    test_start: int
    test_end: int       # exclusive
    calib_start: int    # inside [0, train_end)


def walkforward_folds(n: int, *, min_train: int, refit_every: int, horizon: int,
                      calib_frac: float = 0.15, start: int | None = None,
                      end: int | None = None) -> Iterator[Fold]:
    lo = max(min_train, start or min_train)
    hi = min(n, end or n)
    for s in range(lo, hi, refit_every):
        train_end = s - horizon
        if train_end < min_train // 2:
            continue
        calib_start = max(0, int(train_end * (1 - calib_frac)))
        yield Fold(train_end=train_end, test_start=s,
                   test_end=min(s + refit_every, hi), calib_start=calib_start)
