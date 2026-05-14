"""Interactive Brokers — historical bars via ``ib_insync`` + Gateway/TWS (todo).

Live/paper **orders** for FX summaries use ``utils.ibkr_trade``. ETF loader paths still call
``assert_ibkr_ready()`` until IBKR historical data is implemented alongside Yahoo/Polygon.
"""

from __future__ import annotations

NOT_IMPLEMENTED_MSG = (
    "IBKR historical data for this loader is not implemented yet. "
    "Use Yahoo Finance or Master (Polygon), or wire ib.reqHistoricalData via utils.ibkr_trade patterns."
)


def assert_ibkr_ready() -> None:
    raise RuntimeError(NOT_IMPLEMENTED_MSG)
