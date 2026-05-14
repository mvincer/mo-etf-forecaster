"""Interactive Brokers spot FX execution via **IB Gateway** or **TWS** + ``ib_insync``.

Paper trading typically listens on **127.0.0.1:7497**; live is often **7496**. Enable API access in
Gateway/TWS (Configure → Settings → API → Enable ActiveX / Socket Clients).

Credentials are **not** stored in code: no IB username/password here — you log in through Gateway/TWS,
and this module connects to the already-running socket.

Sizing uses **placeholder** math → ``MarketOrder`` quantity (base units). Confirm min size per pair on demo.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

try:
    from ib_insync import Forex, IB, MarketOrder
except ImportError:  # pragma: no cover
    Forex = IB = MarketOrder = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class IbkrSocketSettings:
    host: str
    port: int
    client_id: int
    account: str | None = None
    assumed_net_liquidation: float | None = None
    max_order_qty: int | None = None
    orderqty_divisor: float | None = None


def _ov_get(key: str, overrides: dict[str, Any] | None) -> str:
    if overrides and key in overrides and overrides[key] not in (None, ""):
        return str(overrides[key]).strip()
    return os.environ.get(key, "").strip()


def ibkr_settings_from_env(overrides: dict[str, Any] | None = None) -> IbkrSocketSettings:
    host = _ov_get("IBKR_HOST", overrides) or "127.0.0.1"
    port_s = _ov_get("IBKR_PORT", overrides) or "7497"
    port = int(port_s)
    cid_s = _ov_get("IBKR_CLIENT_ID", overrides) or "1"
    client_id = int(cid_s)
    acct = _ov_get("IBKR_ACCOUNT", overrides) or None
    anl_s = _ov_get("IBKR_ASSUMED_NETLIQUIDATION", overrides)
    assumed = float(anl_s) if anl_s else None
    mo_s = _ov_get("IBKR_MAX_ORDER_QTY", overrides)
    max_q = int(mo_s) if mo_s else None
    div_s = _ov_get("IBKR_ORDERQTY_DIVISOR", overrides)
    div = float(div_s) if div_s else None
    return IbkrSocketSettings(
        host=host,
        port=port,
        client_id=client_id,
        account=acct,
        assumed_net_liquidation=assumed,
        max_order_qty=max_q,
        orderqty_divisor=div,
    )


_FOREX_RE = re.compile(r"^[A-Za-z]{6}$")


def parse_forex_contract(primary: str) -> Any:
    """Build ``Forex`` from ``EURUSD``, ``EUR_USD``, or ``EUR/USD`` style primaries."""
    if Forex is None:
        raise RuntimeError('Install ib_insync: pip install ib_insync')
    raw = primary.strip().upper().replace(" ", "").replace("_", "")
    if "/" in primary:
        parts = primary.strip().upper().split("/")
        if len(parts) != 2 or len(parts[0]) != 3 or len(parts[1]) != 3:
            raise ValueError(f"Bad FX pair format: {primary!r}")
        base, quote = parts[0], parts[1]
    elif _FOREX_RE.match(raw):
        base, quote = raw[:3], raw[3:]
    else:
        raise ValueError(
            f"Unsupported primary {primary!r} for IBKR Forex auto-route — use six-letter symbols "
            f"(e.g. EURUSD) or BASE/QUOTE."
        )
    return Forex(base, quote)


def ib_pair_display(primary: str) -> str:
    try:
        c = parse_forex_contract(primary)
        return f"{c.symbol}.{c.currency}"
    except Exception:
        return primary.strip()


def _ensure_ib() -> IB:
    if IB is None:
        raise RuntimeError(
            "Missing dependency: pip install ib_insync\n"
            "Also run IB Gateway or TWS with API socket enabled."
        )
    return IB()


def _net_liquidation(ib: IB, account: str | None, deadline_sec: float = 5.0) -> float | None:
    """Best-effort NetLiquidation from account summary."""
    acct = account or ""
    deadline = time.monotonic() + deadline_sec
    while time.monotonic() < deadline:
        ib.sleep(0.3)
        last: float | None = None
        rows = ib.accountSummary(acct) if acct else ib.accountSummary("")
        if not rows:
            continue
        for av in rows:
            if av.tag == "NetLiquidation":
                try:
                    last = float(str(av.value).replace(",", ""))
                except ValueError:
                    continue
                break
        if last is not None:
            return last
    return None


def placeholder_order_qty(
    *,
    net_liq: float,
    pct_equity: float,
    leverage: float,
    max_qty: int,
    divisor: float,
) -> float:
    if net_liq <= 0 or pct_equity <= 0:
        raise ValueError("Net liquidation and % of equity must be positive.")
    lev = max(leverage, 1e-6)
    notional = net_liq * (pct_equity / 100.0) * lev
    if divisor <= 0:
        divisor = 100.0
    raw = float(max(1, round(notional / divisor)))
    cap = float(max(1, max_qty))
    return min(raw, cap)


def test_ib_connection(settings: IbkrSocketSettings) -> tuple[float | None, str]:
    """Return (net_liquidation_or_none, status_line)."""
    ib = _ensure_ib()
    try:
        try:
            ib.connect(settings.host, settings.port, clientId=settings.client_id, readonly=True)
        except TypeError:
            ib.connect(settings.host, settings.port, clientId=settings.client_id)
        nl = _net_liquidation(ib, settings.account)
        if nl is None:
            nl = settings.assumed_net_liquidation
            msg = f"Connected to **{settings.host}:{settings.port}** (readonly). NetLiquidation not read — using fallback **{nl}**." if nl else (
                f"Connected to **{settings.host}:{settings.port}** but could not read NetLiquidation. "
                "Set **IBKR_ASSUMED_NETLIQUIDATION** or specify **IBKR_ACCOUNT**."
            )
        else:
            msg = f"Connected to **{settings.host}:{settings.port}**. NetLiquidation ≈ **{nl:,.2f}**."
        return nl, msg
    finally:
        ib.disconnect()


def execute_ib_forex_market(
    *,
    primary: str,
    is_buy: bool,
    pct_equity: float,
    leverage: float,
    settings: IbkrSocketSettings,
) -> str:
    """Qualify Forex contract, place market order, wait for fill/reject."""
    contract = parse_forex_contract(primary)
    ib = _ensure_ib()
    max_q = settings.max_order_qty or max(1, int(os.environ.get("IBKR_MAX_ORDER_QTY", "100000")))
    div = settings.orderqty_divisor or float(os.environ.get("IBKR_ORDERQTY_DIVISOR", "100"))

    try:
        try:
            ib.connect(settings.host, settings.port, clientId=settings.client_id, readonly=False)
        except TypeError:
            ib.connect(settings.host, settings.port, clientId=settings.client_id)
        ib.qualifyContracts(contract)

        nl = _net_liquidation(ib, settings.account)
        if nl is None or nl <= 0:
            nl = settings.assumed_net_liquidation
        if nl is None or nl <= 0:
            raise RuntimeError(
                "Could not read NetLiquidation. Set **IBKR_ASSUMED_NETLIQUIDATION** for placeholder sizing."
            )

        qty = placeholder_order_qty(
            net_liq=nl,
            pct_equity=pct_equity,
            leverage=leverage,
            max_qty=max_q,
            divisor=div,
        )
        action = "BUY" if is_buy else "SELL"
        order = MarketOrder(action, float(qty))
        if settings.account:
            order.account = settings.account

        trade = ib.placeOrder(contract, order)
        t_end = time.monotonic() + 30.0
        terminal = {"Filled", "Cancelled", "Inactive", "ApiCancelled"}
        while trade.orderStatus.status not in terminal:
            ib.sleep(0.25)
            if time.monotonic() > t_end:
                break

        st = trade.orderStatus.status
        filled = trade.orderStatus.filled
        avg = trade.orderStatus.avgFillPrice
        sym = f"{contract.symbol}.{contract.currency}"
        extra = f" filled={filled}" if filled else ""
        extra += f" @ {avg}" if avg and str(avg) != "nan" else ""
        return f"{action} **{sym}** qty≈**{qty}** — status **{st}**.{extra} (placeholder sizing)."
    finally:
        ib.disconnect()
