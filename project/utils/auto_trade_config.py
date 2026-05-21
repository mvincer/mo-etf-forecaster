"""Persistent settings for the daily FX auto-trade check.

The config lives in ``Mo_Dash/Dashboard/auto_trade.config.json`` so it is shared between
the dashboard (read/write via the auto-trade panel) and the 5:20 PM scheduled job
(``project/scripts/daily_trade_check.py``).

Schema (with safe defaults):

``enabled`` (bool, **default false**)
    Master toggle. When **false**, the daily check only **analyzes and reports** —
    it never closes or opens a position. The dashboard always renders the latest
    report regardless of this flag.

``top_n`` (int, default 2)
    Size of the target book each day. Selection walks the ranking (by
    ``|pair_score|`` descending) and picks pairs **greedily** while requiring the
    new pair's BASE and QUOTE to be disjoint from already-selected pairs (no
    currency shared across selected trades). A position already open in this
    selected book is **kept** (no churn, no double commission); anything not in
    the selected book is closed; missing slots are opened (when enabled).

``pct_equity_per_trade`` (float, default 5.0)
    Notional percentage of account equity assigned to each auto-opened trade.

``leverage`` (float, default 10.0)
    Leverage multiplier used to convert ``pct_equity`` into a lot size, matching
    the manual dashboard trade form's behaviour.

``stop_pct`` (float, default 1.0)
    Stop-loss distance in % of mid-rate, applied to each auto-opened trade.

``max_open`` (int, default 2)
    Hard cap on simultaneously open auto-trades, regardless of ``top_n``.

``skip_unsubscribed`` (bool, default true)
    If a top-ranked pair has FXCM ``subscription_status != 'T'`` (e.g. AUD/USD
    today), log + report it and skip rather than failing the whole pass.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


DEFAULT_CONFIG_FILENAME = "auto_trade.config.json"


@dataclass
class AutoTradeConfig:
    enabled: bool = False
    top_n: int = 2
    pct_equity_per_trade: float = 5.0
    leverage: float = 10.0
    stop_pct: float = 1.0
    max_open: int = 2
    skip_unsubscribed: bool = True
    # Read-only metadata (filled in by ``load_config``).
    _path: str = field(default="", repr=False)


def default_config_path() -> Path:
    """``Mo_Dash/Dashboard/auto_trade.config.json`` — discovered from this file's location.

    This file lives at ``Mo_Dash/ETF/ETF Forecaster/project/utils/auto_trade_config.py``.
    Walk up four levels to reach ``Mo_Dash/``, then into ``Dashboard/``.
    """
    env = (os.environ.get("FX_AUTO_TRADE_CONFIG") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    mo_dash_root = here.parents[4]  # …/Mo_Dash
    return (mo_dash_root / "Dashboard" / DEFAULT_CONFIG_FILENAME).resolve()


def load_config(path: Path | None = None) -> AutoTradeConfig:
    """Load config from JSON, falling back to defaults when missing/corrupt."""
    p = (path or default_config_path()).resolve()
    cfg = AutoTradeConfig(_path=str(p))
    if not p.is_file():
        return cfg
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cfg
    if not isinstance(raw, dict):
        return cfg
    valid = {f for f in AutoTradeConfig.__dataclass_fields__.keys() if not f.startswith("_")}
    for k, v in raw.items():
        if k not in valid:
            continue
        try:
            current = getattr(cfg, k)
            setattr(cfg, k, type(current)(v))
        except (TypeError, ValueError):
            continue
    return cfg


def save_config(cfg: AutoTradeConfig, path: Path | None = None) -> Path:
    """Persist config to JSON (creates parent dir as needed)."""
    p = (path or default_config_path()).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    cfg._path = str(p)
    return p
