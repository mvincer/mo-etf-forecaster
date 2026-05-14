"""FXCM **ForexConnect** Python trading (Gehtsoft ``forexconnect`` package).

Credentials and server settings come from environment variables or Streamlit secrets — never from code.

Order size (default): **% of account equity × leverage** (equity from ForexConnect). Target notional is in
**account currency**, converted to the traded pair's **base currency** using subscribed **OFFERS** mids, then
``lots = notional_base / base_unit_size``. Override account currency with **``FXCM_FC_ACCOUNT_CURRENCY``** if needed.
Optional **``lots``** bypasses that path. ``AMOUNT`` = ``base_unit_size × lots`` (capped by ``FXCM_FC_MAX_LOTS``).
Each **true market** entry requests an attached **stop** via ``RATE_STOP`` (default **1%** from the live bid/ask:
below ask for buys, above bid for sells). If the broker rejects entry+stop, the order is sent **without** an attached
stop and the return message tells you the theoretical stop level.

**Windows / PyPI:** ``forexconnect`` wheels exist only for **CPython 3.5–3.7** (``win_amd64``). The main app can run
on **3.11+**; orders are delegated to a sidecar **``.venv-fc``** (Python 3.7) via ``scripts/fxcm_fc_worker.py``.
Run **``setup_fxcm_venv.bat``** to create ``.venv`` + ``.venv-fc``. Override the interpreter with **``FOREXCONNECT_PYTHON``**.

Reference: https://github.com/gehtsoft/forex-connect — sample ``OpenPosition.py``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep
from typing import Any

from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

_FC_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _forexconnect_importable() -> bool:
    try:
        import forexconnect  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _fc_bridge_python_exe() -> Path | None:
    env = (os.environ.get("FOREXCONNECT_PYTHON") or "").strip()
    if env:
        p = Path(env)
        return p if p.is_file() else None
    candidate = _FC_PROJECT_ROOT / ".venv-fc" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else None


def _run_fc_worker(payload: dict[str, Any], *, timeout: float) -> str:
    bridge = _fc_bridge_python_exe()
    if bridge is None:
        raise RuntimeError(
            "No ``forexconnect`` in this interpreter and no **.venv-fc** bridge found.\n\n"
            "Run **setup_fxcm_venv.bat** in the `project` folder (installs Python 3.7 sidecar + ``forexconnect``), "
            "or set **FOREXCONNECT_PYTHON** to a Python 3.7 executable that has ``forexconnect`` installed."
        )
    worker = _FC_PROJECT_ROOT / "scripts" / "fxcm_fc_worker.py"
    if not worker.is_file():
        raise RuntimeError(f"Missing FXCM worker script: {worker}")

    fd, raw_path = tempfile.mkstemp(suffix=".json", text=True)
    os.close(fd)
    path = Path(raw_path)
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.run(
            [str(bridge), str(worker), str(path)],
            cwd=str(_FC_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(err or f"FXCM worker exited with code {proc.returncode}")
        return (proc.stdout or "").strip()
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _load_forexconnect() -> tuple[Any, Any, Any]:
    try:
        from forexconnect import fxcorepy, ForexConnect, Common

        return fxcorepy, ForexConnect, Common
    except ImportError as e:
        raise RuntimeError(
            "The ``forexconnect`` package is not installed in this interpreter.\n\n"
            "On **Windows**, install the **3.7 sidecar** (``setup_fxcm_venv.bat`` creates ``.venv-fc``) or set "
            "**FOREXCONNECT_PYTHON** to a Python that has ``forexconnect``.\n\n"
            f"Original error: {e}"
        ) from e


@dataclass(frozen=True)
class FxcmForexConnectSettings:
    user: str
    password: str
    url: str
    connection: str
    session: str = ""
    pin: str = ""
    account: str | None = None
    assumed_equity: float | None = None
    max_lots: int = 100
    lots_divisor: float = 100_000.0


def _env(key: str, overrides: dict[str, Any] | None) -> str:
    if overrides and key in overrides and overrides[key] not in (None, ""):
        return str(overrides[key]).strip()
    return os.environ.get(key, "").strip()


def fc_settings_from_env(overrides: dict[str, Any] | None = None) -> FxcmForexConnectSettings:
    user = _env("FXCM_FC_USER", overrides)
    pwd = _env("FXCM_FC_PASSWORD", overrides)
    url = _env("FXCM_FC_URL", overrides) or "https://www.fxcorporate.com/Hosts.jsp"
    conn = _env("FXCM_FC_CONNECTION", overrides) or "Demo"
    sess = _env("FXCM_FC_SESSION", overrides) or ""
    pin = _env("FXCM_FC_PIN", overrides) or ""
    acct = _env("FXCM_FC_ACCOUNT", overrides) or None
    ae = _env("FXCM_FC_ASSUMED_EQUITY", overrides)
    assumed = float(ae) if ae else None
    try:
        max_lots = max(1, int(_env("FXCM_FC_MAX_LOTS", overrides) or "100"))
    except ValueError:
        max_lots = 100
    try:
        div = float(_env("FXCM_FC_LOTS_DIVISOR", overrides) or "100000")
    except ValueError:
        div = 100_000.0
    return FxcmForexConnectSettings(
        user=user,
        password=pwd,
        url=url,
        connection=conn,
        session=sess,
        pin=pin,
        account=acct,
        assumed_equity=assumed,
        max_lots=max_lots,
        lots_divisor=div,
    )


def validate_fc_settings(s: FxcmForexConnectSettings) -> None:
    miss = []
    if not s.user:
        miss.append("FXCM_FC_USER")
    if not s.password:
        miss.append("FXCM_FC_PASSWORD")
    if miss:
        raise RuntimeError("Missing ForexConnect settings: " + ", ".join(miss))


_FOREX_RE = re.compile(r"^[A-Za-z]{6}$")


def primary_to_fc_instrument(primary: str) -> str:
    """Map workbook ``primary`` to a canonical **BASE/QUOTE** id (FXCM often also exposes **BASEQUOTE**)."""
    s = primary.strip().upper().replace(" ", "")
    if "/" in s:
        a, b = s.split("/", 1)
        a, b = a.strip(), b.strip()
        if len(a) == 3 and len(b) == 3:
            return f"{a}/{b}"
    s = s.replace("_", "")
    if _FOREX_RE.match(s):
        return f"{s[:3]}/{s[3:]}"
    return primary.strip()


def _fc_pair_key(primary: str) -> str:
    """Six letters BASEQUOTE for matching rows in the OFFERS table."""
    return re.sub(r"[^A-Z]", "", primary_to_fc_instrument(primary).upper())


def _fc_instrument_env_override(primary: str) -> str | None:
    """Optional JSON map ``FXCM_FC_INSTRUMENT_MAP`` e.g. ``{"AUD/USD": "AUDUSD"}`` for odd account symbols."""
    raw = (os.environ.get("FXCM_FC_INSTRUMENT_MAP") or "").strip()
    if not raw:
        return None
    try:
        m = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("FXCM_FC_INSTRUMENT_MAP is not valid JSON; ignoring.")
        return None
    if not isinstance(m, dict):
        return None
    for k in (primary.strip(), primary.strip().upper(), primary_to_fc_instrument(primary)):
        if not k:
            continue
        v = m.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _fc_instrument_candidates(primary: str) -> list[str]:
    """Symbols to try with ``Common.get_offer`` (exact table ``instrument`` strings differ by venue)."""
    seen: set[str] = set()
    out: list[str] = []

    def add(x: str) -> None:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    ov = _fc_instrument_env_override(primary)
    if ov:
        add(ov)
    canon = primary_to_fc_instrument(primary)
    add(canon)
    key = _fc_pair_key(primary)
    if len(key) == 6:
        add(f"{key[:3]}/{key[3:]}")
        add(key)
    add(primary.strip().upper().replace(" ", "").replace("_", ""))
    return out


def _iter_fc_offer_rows(fx: Any, ForexConnect: Any, fxcorepy: Any) -> list[Any]:
    from forexconnect.errors import TableManagerError  # noqa: PLC0415

    try:
        return list(fx.get_table(ForexConnect.OFFERS))
    except TableManagerError:
        return list(fx.get_table_reader(fxcorepy.O2GTableType.OFFERS))


def _resolve_fc_offer(fx: Any, Common: Any, ForexConnect: Any, fxcorepy: Any, primary: str) -> Any:
    """Resolve an OFFERS row for ``primary``; FXCM tables may use ``EUR/USD`` or ``EURUSD`` (and require subscription **T**)."""
    for cand in _fc_instrument_candidates(primary):
        offer = Common.get_offer(fx, cand)
        if offer:
            return offer

    key6 = _fc_pair_key(primary)
    if len(key6) != 6:
        raise RuntimeError(
            f"Instrument **{primary_to_fc_instrument(primary)}** not found for this login "
            f"(could not derive a 6-letter pair key from {primary!r})."
        )

    rows = _iter_fc_offer_rows(fx, ForexConnect, fxcorepy)
    subscribed: Any | None = None
    fallback: Any | None = None
    for offer in rows:
        ins = str(getattr(offer, "instrument", "") or "")
        if _fc_pair_key(ins) != key6:
            continue
        st = str(getattr(offer, "subscription_status", "") or "")
        if st == "T":
            subscribed = offer
            break
        if fallback is None:
            fallback = offer

    chosen = subscribed or fallback
    if chosen is None:
        raise RuntimeError(
            f"Instrument **{primary_to_fc_instrument(primary)}** not found for this login "
            f"(no OFFERS row for **{key6}**). Check FXCM Trading Station → instrument list for this account, "
            f"or set **FXCM_FC_INSTRUMENT_MAP** in `.env` to map the pair to the exact symbol string."
        )
    st = str(getattr(chosen, "subscription_status", "") or "")
    if st != "T":
        raise RuntimeError(
            f"Found **{chosen.instrument}** but it is not price-subscribed (subscription_status={st!r}, need **T**). "
            "Open the pair in FXCM Trading Station / Marketscope once so the session subscribes to its quotes, then retry."
        )
    return chosen


def _session_status_changed(session: Any, status: Any) -> None:
    logger.info("ForexConnect session status: %s", status)


def _account_equity(account_row: Any, settings: FxcmForexConnectSettings) -> float:
    for attr in ("equity", "Equity", "balance", "Balance", "gross_pl", "usable_margin"):
        v = getattr(account_row, attr, None)
        if v is not None:
            try:
                f = float(str(v).replace(",", ""))
                if f > 0:
                    return f
            except ValueError:
                continue
    if settings.assumed_equity and settings.assumed_equity > 0:
        return float(settings.assumed_equity)
    raise RuntimeError(
        "Could not read account equity/balance from ForexConnect. "
        "Set **FXCM_FC_ASSUMED_EQUITY** for login diagnostics only."
    )


def _account_currency(account_row: Any) -> str:
    """Three-letter account currency when the API exposes it (else **USD** with a log line)."""
    env = (os.environ.get("FXCM_FC_ACCOUNT_CURRENCY") or "").strip().upper()
    if len(env) == 3 and env.isalpha():
        return env
    for attr in (
        "currency",
        "Currency",
        "account_currency",
        "AccountCurrency",
        "accountCurrency",
        "instrument_currency",
        "InstrumentCurrency",
    ):
        v = getattr(account_row, attr, None)
        if v is None:
            continue
        s = str(v).strip().upper()
        if len(s) == 3 and s.isalpha():
            return s
    logger.warning(
        "Could not read account currency from ForexConnect; assuming **USD** for cross sizing. "
        "Set **FXCM_FC_ACCOUNT_CURRENCY** (e.g. JPY) if your equity is not in USD."
    )
    return "USD"


def _instrument_base_quote(instrument: str) -> tuple[str, str] | None:
    """Return ``(BASE, QUOTE)`` for a 6-letter FX symbol such as ``EUR/USD`` or ``EURUSD``."""
    s = instrument.strip().upper().replace(" ", "")
    if "/" in s:
        a, b = s.split("/", 1)
        a, b = a.strip(), b.strip()
        if len(a) == 3 and len(b) == 3 and a.isalpha() and b.isalpha():
            return a, b
    s = re.sub(r"[^A-Z]", "", s)
    if len(s) == 6 and s.isalpha():
        return s[:3], s[3:]
    return None


def _build_fx_conversion_graph(fx: Any, ForexConnect: Any, fxcorepy: Any) -> dict[str, list[tuple[str, float]]]:
    """Adjacency: ``ccy -> [(neighbor, mult)]`` where ``amount_neighbor = amount_ccy * mult``."""
    adj: dict[str, list[tuple[str, float]]] = {}
    rows = _iter_fc_offer_rows(fx, ForexConnect, fxcorepy)
    for offer in rows:
        st = str(getattr(offer, "subscription_status", "") or "")
        if st != "T":
            continue
        ins = str(getattr(offer, "instrument", "") or "")
        pq = _instrument_base_quote(ins)
        if pq is None:
            continue
        base_ccy, quote_ccy = pq
        try:
            bid, ask, _ = _fc_offer_bid_ask_digits(offer)
            mid = 0.5 * (bid + ask)
        except Exception:
            continue
        if mid <= 0 or not math.isfinite(mid):
            continue
        # 1 BASE = mid QUOTE  →  amount_quote = amount_base * mid
        adj.setdefault(base_ccy, []).append((quote_ccy, mid))
        adj.setdefault(quote_ccy, []).append((base_ccy, 1.0 / mid))
    return adj


def _convert_currency_amount(
    adj: dict[str, list[tuple[str, float]]],
    amount: float,
    from_ccy: str,
    to_ccy: str,
) -> float:
    """Convert ``amount`` of ``from_ccy`` into equivalent units of ``to_ccy`` using mid rates on ``adj``."""
    a0 = from_ccy.strip().upper()
    b0 = to_ccy.strip().upper()
    if a0 == b0:
        return float(amount)
    if float(amount) <= 0 or not math.isfinite(float(amount)):
        raise ValueError("Conversion amount must be positive and finite.")
    q: deque[tuple[str, float]] = deque([(a0, float(amount))])
    seen: set[str] = {a0}
    while q:
        ccy, amt = q.popleft()
        if ccy == b0:
            return amt
        for nb, mult in adj.get(ccy, []):
            if nb in seen:
                continue
            seen.add(nb)
            q.append((nb, amt * mult))
    raise RuntimeError(
        f"No subscribed FX path from **{a0}** to **{b0}** in OFFERS (need mids for cross conversion). "
        "Subscribe majors in Trading Station (e.g. USD, EUR, JPY legs) or set **FXCM_FC_ACCOUNT_CURRENCY** "
        "and ensure crosses exist."
    )


def _lots_from_pct_leverage_notional(
    fx: Any,
    Common: Any,
    ForexConnect: Any,
    fxcorepy: Any,
    tsp: Any,
    *,
    account: Any,
    settings: FxcmForexConnectSettings,
    primary: str,
    pct_equity: float,
    leverage: float,
) -> float:
    """Target notional = equity × (pct/100) × leverage in **account currency**, then convert to **pair base** → lots."""
    eq = _account_equity(account, settings)
    acct_ccy = _account_currency(account)
    pct = max(float(pct_equity), 0.0)
    lev = max(float(leverage), 1e-9)
    notional_acct = eq * (pct / 100.0) * lev
    if notional_acct <= 0 or not math.isfinite(notional_acct):
        raise ValueError("Computed target notional is invalid; check equity, %, and leverage.")

    offer = _resolve_fc_offer(fx, Common, ForexConnect, fxcorepy, primary)
    sym = str(offer.instrument)
    pq = _instrument_base_quote(sym)
    if pq is None:
        raise RuntimeError(f"Could not parse base/quote from instrument **{sym}**.")
    pair_base, _pair_quote = pq

    adj = _build_fx_conversion_graph(fx, ForexConnect, fxcorepy)
    target_base_units = _convert_currency_amount(adj, notional_acct, acct_ccy, pair_base)
    if target_base_units <= 0 or not math.isfinite(target_base_units):
        raise RuntimeError("Cross conversion produced a non-positive base-currency size.")

    base_unit_size = float(tsp.get_base_unit_size(sym, account))
    if base_unit_size <= 0:
        raise RuntimeError(f"Invalid base_unit_size for **{sym}**: {base_unit_size}")
    lots_raw = float(target_base_units) / base_unit_size
    if not math.isfinite(lots_raw) or lots_raw <= 0:
        raise RuntimeError("Computed lot count is invalid after cross conversion.")
    return float(lots_raw)


def _effective_lots(lots: float, settings: FxcmForexConnectSettings) -> float:
    if float(lots) <= 0:
        raise ValueError("Lots must be positive.")
    cap = float(settings.max_lots) if int(settings.max_lots) > 0 else float("inf")
    return min(float(lots), cap)


def _resolve_order_lots(
    *,
    fx: Any,
    Common: Any,
    ForexConnect: Any,
    fxcorepy: Any,
    tsp: Any,
    primary: str,
    lots: float | None,
    pct_equity: float | None,
    leverage: float | None,
    account: Any,
    settings: FxcmForexConnectSettings,
) -> float:
    """Prefer explicit ``lots``; otherwise **% of equity × leverage** → notional in account CCY → pair base → lots."""
    if lots is not None:
        return float(lots)
    if pct_equity is not None and leverage is not None:
        return _lots_from_pct_leverage_notional(
            fx,
            Common,
            ForexConnect,
            fxcorepy,
            tsp,
            account=account,
            settings=settings,
            primary=primary,
            pct_equity=float(pct_equity),
            leverage=float(leverage),
        )
    raise RuntimeError(
        "Order size: pass **`lots`**, or both **`pct_equity`** and **`leverage`** for equity-based sizing "
        "(notional in account currency, converted to the pair's base currency via live OFFERS mids). "
        "Restart Streamlit after pulling updates if you still see this."
    )


def _lots_to_amount(base_unit_size: float, lots: float, settings: FxcmForexConnectSettings) -> tuple[int, float]:
    """Return ``(AMOUNT, effective_lots)`` for ``TRUE_MARKET_OPEN`` (``AMOUNT`` = base units × lots)."""
    eff = _effective_lots(lots, settings)
    raw = float(base_unit_size) * eff
    amount = int(max(1, round(raw)))
    return amount, eff


def _fc_offer_bid_ask_digits(offer: Any) -> tuple[float, float, int]:
    bid_v = getattr(offer, "bid", None)
    ask_v = getattr(offer, "ask", None)
    if bid_v is None or ask_v is None:
        try:
            bid_v = float(offer["Bid"])
            ask_v = float(offer["Ask"])
        except Exception as e:
            raise RuntimeError("Could not read Bid/Ask from the offer row (needed for stop placement).") from e
    digits_raw = getattr(offer, "digits", None)
    try:
        digits = int(digits_raw) if digits_raw is not None else 5
    except (TypeError, ValueError):
        digits = 5
    digits = max(0, min(digits, 10))
    bid = float(bid_v)
    ask = float(ask_v)
    if bid <= 0 or ask <= 0 or ask < bid:
        raise RuntimeError(f"Invalid bid/ask snapshot (bid={bid}, ask={ask}) for stop calculation.")
    return bid, ask, digits


def _fc_stop_rate(*, is_buy: bool, bid: float, ask: float, stop_loss_pct: float, digits: int) -> float:
    """Initial stop: below ask for buys, above bid for sells (``stop_loss_pct`` percent adverse move)."""
    p = max(float(stop_loss_pct), 1e-9) / 100.0
    if is_buy:
        ref = ask
        stop = ref * (1.0 - p)
    else:
        ref = bid
        stop = ref * (1.0 + p)
    d = max(int(digits), 1)
    return float(round(stop, d))


def _create_market_entry_request(
    fx: Any,
    fxcorepy: Any,
    *,
    str_account: str,
    offer: Any,
    is_buy: bool,
    amount: int,
    stop_loss_pct: float,
) -> tuple[Any, str]:
    """Build ``TRUE_MARKET_OPEN``; attach ``RATE_STOP`` when accepted, else plain market."""
    buy_sell = "B" if is_buy else "S"
    market_open = fxcorepy.Constants.Orders.TRUE_MARKET_OPEN
    sym = str(offer.instrument)
    base_kw: dict[str, Any] = dict(
        ACCOUNT_ID=str_account,
        BUY_SELL=buy_sell,
        AMOUNT=int(amount),
        SYMBOL=sym,
    )
    bid, ask, digits = _fc_offer_bid_ask_digits(offer)
    stop_rate = _fc_stop_rate(is_buy=is_buy, bid=bid, ask=ask, stop_loss_pct=stop_loss_pct, digits=digits)
    req = fx.create_order_request(order_type=market_open, RATE_STOP=stop_rate, **base_kw)
    if req is not None:
        return req, f" · initial SL **{stop_rate:.{max(digits, 1)}f}** ({stop_loss_pct:g}% from quote)"
    req_plain = fx.create_order_request(order_type=market_open, **base_kw)
    if req_plain is None:
        raise RuntimeError("ForexConnect refused to build market order request (with or without stop).")
    logger.warning(
        "Market+RATE_STOP rejected for %s; sending market without attached stop (stop would have been %.10f).",
        sym,
        stop_rate,
    )
    return (
        req_plain,
        f" · **no attached SL** (broker rejected entry+stop); place stop manually near **{stop_rate:.{max(digits, 1)}f}** "
        f"({stop_loss_pct:g}% from quote).",
    )


class _TradesMonitor:
    def __init__(self) -> None:
        self._open_order_id: str | None = None
        self._trades: dict[str, Any] = {}
        self._event = threading.Event()

    def on_added_trade(self, _a: Any, _b: Any, trade_row: Any) -> None:
        oid = trade_row.open_order_id
        self._trades[oid] = trade_row
        if self._open_order_id == oid:
            self._event.set()

    def wait(self, timeout: float, open_order_id: str) -> Any | None:
        self._open_order_id = open_order_id
        if open_order_id in self._trades:
            return self._trades[open_order_id]
        self._event.wait(timeout)
        return self._trades.get(open_order_id)


class _OrdersMonitor:
    def __init__(self) -> None:
        self._order_id: str | None = None
        self._added: dict[str, Any] = {}
        self._changed: dict[str, Any] = {}
        self._deleted: dict[str, Any] = {}
        self._add_ev = threading.Event()
        self._chg_ev = threading.Event()
        self._del_ev = threading.Event()

    def on_added_order(self, _a: Any, _b: Any, order_row: Any) -> None:
        oid = order_row.order_id
        self._added[oid] = order_row
        if self._order_id == oid:
            self._add_ev.set()

    def on_changed_order(self, _a: Any, _b: Any, order_row: Any) -> None:
        oid = order_row.order_id
        self._changed[oid] = order_row
        if self._order_id == oid:
            self._chg_ev.set()

    def on_deleted_order(self, _a: Any, _b: Any, order_row: Any) -> None:
        oid = order_row.order_id
        self._deleted[oid] = order_row
        if self._order_id == oid:
            self._del_ev.set()

    def wait_lifecycle(self, timeout: float, order_id: str) -> bool:
        self._order_id = order_id
        if order_id not in self._added:
            self._add_ev.wait(timeout)
        if order_id not in self._changed:
            self._chg_ev.wait(timeout)
        if order_id not in self._deleted:
            self._del_ev.wait(timeout)
        return order_id in self._added and order_id in self._changed and order_id in self._deleted


def _test_fc_login_once(settings: FxcmForexConnectSettings) -> str:
    """Single attempt: log in, resolve account, return status line (no order)."""
    validate_fc_settings(settings)
    if os.environ.get("_FC_WORKER_PROCESS") != "1":
        b = _fc_bridge_python_exe()
        if b is not None:
            return _run_fc_worker({"op": "login_test", "settings": asdict(settings)}, timeout=90.0)
    if not _forexconnect_importable():
        raise RuntimeError(
            "No **.venv-fc** bridge and ``forexconnect`` is not importable in this interpreter. "
            "Run **setup_fxcm_venv.bat** in `project` or set **FOREXCONNECT_PYTHON**."
        )
    _fxcorepy, ForexConnect, Common = _load_forexconnect()
    with ForexConnect() as fx:
        try:
            fx.login(
                settings.user,
                settings.password,
                settings.url,
                settings.connection,
                settings.session,
                settings.pin,
                _session_status_changed,
            )
            account = Common.get_account(fx, settings.account)
            if not account:
                raise RuntimeError("No valid account — set **FXCM_FC_ACCOUNT** or check login.")
            eq = _account_equity(account, settings)
            acy = _account_currency(account)
            return (
                f"Logged in (**{settings.connection}**). Account **{account.account_id}** · "
                f"equity/balance snapshot ≈ **{eq:,.2f}** · sizing currency **{acy}**."
            )
        finally:
            try:
                fx.logout()
            except Exception:
                pass


@retry(stop=stop_after_attempt(2), wait=wait_fixed(2), reraise=True)
def test_fc_login(settings: FxcmForexConnectSettings) -> str:
    """Log in test with one retry (transient network / price server blips)."""
    return _test_fc_login_once(settings)


def execute_fc_market_order(
    *,
    primary: str,
    is_buy: bool,
    settings: FxcmForexConnectSettings,
    lots: float | None = None,
    pct_equity: float | None = None,
    leverage: float | None = None,
    stop_loss_pct: float = 1.0,
    order_wait_sec: float = 25.0,
) -> str:
    validate_fc_settings(settings)
    if lots is None and (pct_equity is None or leverage is None):
        raise RuntimeError(
            "Pass **`lots`** for fixed lot size, or both **`pct_equity`** and **`leverage`** for equity-based sizing "
            "(notional in account currency, converted to pair base via live quotes). Restart Streamlit after updates."
        )
    if os.environ.get("_FC_WORKER_PROCESS") != "1":
        b = _fc_bridge_python_exe()
        if b is not None:
            pl: dict[str, Any] = {
                "op": "single",
                "settings": asdict(settings),
                "primary": primary,
                "is_buy": is_buy,
                "stop_loss_pct": float(stop_loss_pct),
                "order_wait_sec": order_wait_sec,
            }
            if lots is not None:
                pl["lots"] = float(lots)
            if pct_equity is not None:
                pl["pct_equity"] = float(pct_equity)
            if leverage is not None:
                pl["leverage"] = float(leverage)
            return _run_fc_worker(pl, timeout=180.0)
    if not _forexconnect_importable():
        raise RuntimeError(
            "No **.venv-fc** bridge and ``forexconnect`` is not importable here — cannot place orders in-process."
        )
    fxcorepy, ForexConnect, Common = _load_forexconnect()
    display_symbol = primary_to_fc_instrument(primary)
    eff_lots = 0.0
    amount = 0
    order_id = "?"
    ok = False
    trade_row: Any | None = None
    sl_note = ""

    with ForexConnect() as fx:
        trades_listener = orders_listener = None
        try:
            fx.login(
                settings.user,
                settings.password,
                settings.url,
                settings.connection,
                settings.session,
                settings.pin,
                _session_status_changed,
            )
            account = Common.get_account(fx, settings.account)
            if not account:
                raise RuntimeError("No valid account — set **FXCM_FC_ACCOUNT** or check login.")
            str_account = account.account_id

            offer = _resolve_fc_offer(fx, Common, ForexConnect, fxcorepy, primary)
            display_symbol = str(offer.instrument)

            tsp = fx.login_rules.trading_settings_provider
            base_unit_size = tsp.get_base_unit_size(display_symbol, account)
            resolved_lots = _resolve_order_lots(
                fx=fx,
                Common=Common,
                ForexConnect=ForexConnect,
                fxcorepy=fxcorepy,
                tsp=tsp,
                primary=primary,
                lots=lots,
                pct_equity=pct_equity,
                leverage=leverage,
                account=account,
                settings=settings,
            )
            amount, eff_lots = _lots_to_amount(base_unit_size, resolved_lots, settings)

            request, sl_note = _create_market_entry_request(
                fx,
                fxcorepy,
                str_account=str_account,
                offer=offer,
                is_buy=is_buy,
                amount=amount,
                stop_loss_pct=stop_loss_pct,
            )

            orders_monitor = _OrdersMonitor()
            trades_monitor = _TradesMonitor()
            trades_table = fx.get_table(ForexConnect.TRADES)
            orders_table = fx.get_table(ForexConnect.ORDERS)
            trades_listener = Common.subscribe_table_updates(
                trades_table, on_add_callback=trades_monitor.on_added_trade
            )
            orders_listener = Common.subscribe_table_updates(
                orders_table,
                on_add_callback=orders_monitor.on_added_order,
                on_delete_callback=orders_monitor.on_deleted_order,
                on_change_callback=orders_monitor.on_changed_order,
            )
            resp = fx.send_request(request)
            order_id = resp.order_id
            ok = orders_monitor.wait_lifecycle(order_wait_sec, order_id)
            trade_row = trades_monitor.wait(order_wait_sec, order_id) if ok else None
        finally:
            if trades_listener is not None:
                try:
                    trades_listener.unsubscribe()
                except Exception:
                    pass
            if orders_listener is not None:
                try:
                    orders_listener.unsubscribe()
                except Exception:
                    pass
            sleep(0.5)
            try:
                fx.logout()
            except Exception:
                pass

    side = "BUY" if is_buy else "SELL"
    if trade_row is not None:
        return (
            f"{side} **{display_symbol}** · lots **{eff_lots:g}** · amount **{amount}**{sl_note} · "
            f"OrderID **{order_id}** · Trade **{trade_row.trade_id}** @ **{trade_row.open_rate:.5f}**."
        )
    if ok:
        return (
            f"{side} **{display_symbol}** · lots **{eff_lots:g}** · amount **{amount}**{sl_note} · "
            f"OrderID **{order_id}** (trade row timeout)."
        )
    return (
        f"{side} **{display_symbol}** · lots **{eff_lots:g}** · amount **{amount}**{sl_note} · "
        f"OrderID **{order_id}** (order lifecycle timeout)."
    )


def execute_fc_market_orders_sequence(
    *,
    legs: list[tuple[str, bool]],
    settings: FxcmForexConnectSettings,
    lots_per_leg: float | None = None,
    pct_equity_each: float | None = None,
    leverage: float | None = None,
    stop_loss_pct: float = 1.0,
    order_wait_sec: float = 25.0,
) -> str:
    """Execute several true-market opens in **one** ForexConnect session (equal sizing per leg)."""
    if not legs:
        return "No orders requested."
    if lots_per_leg is None and (pct_equity_each is None or leverage is None):
        raise RuntimeError(
            "Pass **`lots_per_leg`**, or **`pct_equity_each`** and **`leverage`** for per-leg equity sizing "
            "(each leg converts from account currency to that pair's base)."
        )
    validate_fc_settings(settings)
    if os.environ.get("_FC_WORKER_PROCESS") != "1":
        b = _fc_bridge_python_exe()
        if b is not None:
            pl2: dict[str, Any] = {
                "op": "sequence",
                "settings": asdict(settings),
                "legs": [[a, b] for a, b in legs],
                "stop_loss_pct": float(stop_loss_pct),
                "order_wait_sec": order_wait_sec,
            }
            if lots_per_leg is not None:
                pl2["lots_per_leg"] = float(lots_per_leg)
            if pct_equity_each is not None:
                pl2["pct_equity_each"] = float(pct_equity_each)
            if leverage is not None:
                pl2["leverage"] = float(leverage)
            return _run_fc_worker(pl2, timeout=max(300.0, 60.0 * len(legs)))
    if not _forexconnect_importable():
        raise RuntimeError(
            "No **.venv-fc** bridge and ``forexconnect`` is not importable here — cannot place basket in-process."
        )
    fxcorepy, ForexConnect, Common = _load_forexconnect()
    lines: list[str] = []

    with ForexConnect() as fx:
        trades_listener = orders_listener = None
        try:
            fx.login(
                settings.user,
                settings.password,
                settings.url,
                settings.connection,
                settings.session,
                settings.pin,
                _session_status_changed,
            )
            account = Common.get_account(fx, settings.account)
            if not account:
                raise RuntimeError("No valid account — set **FXCM_FC_ACCOUNT** or check login.")
            str_account = account.account_id
            tsp = fx.login_rules.trading_settings_provider

            for primary, is_buy in legs:
                try:
                    offer = _resolve_fc_offer(fx, Common, ForexConnect, fxcorepy, primary)
                except RuntimeError as err:
                    lines.append(str(err))
                    continue
                sym = str(offer.instrument)
                resolved_per = _resolve_order_lots(
                    fx=fx,
                    Common=Common,
                    ForexConnect=ForexConnect,
                    fxcorepy=fxcorepy,
                    tsp=tsp,
                    primary=primary,
                    lots=lots_per_leg,
                    pct_equity=pct_equity_each,
                    leverage=leverage,
                    account=account,
                    settings=settings,
                )
                base_unit_size = tsp.get_base_unit_size(sym, account)
                amount, eff_lots = _lots_to_amount(base_unit_size, resolved_per, settings)

                request, sl_note = _create_market_entry_request(
                    fx,
                    fxcorepy,
                    str_account=str_account,
                    offer=offer,
                    is_buy=is_buy,
                    amount=amount,
                    stop_loss_pct=stop_loss_pct,
                )

                orders_monitor = _OrdersMonitor()
                trades_monitor = _TradesMonitor()
                trades_table = fx.get_table(ForexConnect.TRADES)
                orders_table = fx.get_table(ForexConnect.ORDERS)
                trades_listener = Common.subscribe_table_updates(
                    trades_table, on_add_callback=trades_monitor.on_added_trade
                )
                orders_listener = Common.subscribe_table_updates(
                    orders_table,
                    on_add_callback=orders_monitor.on_added_order,
                    on_delete_callback=orders_monitor.on_deleted_order,
                    on_change_callback=orders_monitor.on_changed_order,
                )
                try:
                    resp = fx.send_request(request)
                    order_id = resp.order_id
                    ok = orders_monitor.wait_lifecycle(order_wait_sec, order_id)
                    trade_row = trades_monitor.wait(order_wait_sec, order_id) if ok else None
                    side = "BUY" if is_buy else "SELL"
                    if trade_row is not None:
                        lines.append(
                            f"{side} **{sym}** · lots **{eff_lots:g}** · amount **{amount}**{sl_note} · "
                            f"OrderID **{order_id}** · Trade **{trade_row.trade_id}** @ **{trade_row.open_rate:.5f}**."
                        )
                    elif ok:
                        lines.append(
                            f"{side} **{sym}** · lots **{eff_lots:g}** · amount **{amount}**{sl_note} · "
                            f"OrderID **{order_id}** (trade row timeout)."
                        )
                    else:
                        lines.append(
                            f"{side} **{sym}** · lots **{eff_lots:g}** · amount **{amount}**{sl_note} · "
                            f"OrderID **{order_id}** (order lifecycle timeout)."
                        )
                finally:
                    if trades_listener is not None:
                        try:
                            trades_listener.unsubscribe()
                        except Exception:
                            pass
                        trades_listener = None
                    if orders_listener is not None:
                        try:
                            orders_listener.unsubscribe()
                        except Exception:
                            pass
                        orders_listener = None
                sleep(0.25)
        finally:
            sleep(0.5)
            try:
                fx.logout()
            except Exception:
                pass

    return "\n".join(lines)


def fc_pair_display(primary: str) -> str:
    return primary_to_fc_instrument(primary)
