"""Central configuration: paths, YAML configs, env, universe helpers."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_ROOT = Path(os.environ.get("ETF_FORECASTER_DATA", REPO_ROOT / "data"))
OUTPUTS_DIR = DATA_ROOT / "outputs"
RUNS_DIR = REPO_ROOT / "runs"
LOGS_DIR = REPO_ROOT / "logs"


def mo_dash_root() -> Path:
    return Path(os.environ.get("MO_DASH_ROOT", r"C:\Dev\Mo_Dash"))


def load_env() -> None:
    """Load .env from this repo, falling back to the Mo_Dash broker env files."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(REPO_ROOT / ".env")
    md = mo_dash_root()
    for p in (md / "brokers" / "fred" / ".env", md / ".env", md / "app" / ".env"):
        if p.is_file():
            load_dotenv(p)


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def universe_cfg() -> dict[str, Any]:
    return _load_yaml("universe.yaml")


def blocks_cfg() -> dict[str, Any]:
    return _load_yaml("blocks.yaml")["blocks"]


def search_cfg() -> dict[str, Any]:
    return _load_yaml("search.yaml")


def macro_cfg() -> dict[str, Any]:
    return _load_yaml("macro.yaml")


def models_cfg() -> dict[str, Any]:
    return _load_yaml("models.yaml")


# ---------------------------------------------------------------- universe helpers

def all_tickers() -> list[str]:
    return list(universe_cfg()["tickers"].keys())


def ticker_meta(ticker: str) -> dict[str, Any]:
    return universe_cfg()["tickers"][ticker]


def ticker_group(ticker: str) -> str:
    for group, members in universe_cfg()["groups"].items():
        if ticker in members:
            return group
    return "other"


def close_order(ticker: str) -> int:
    cfg = universe_cfg()
    session = cfg["tickers"][ticker]["session"]
    return int(cfg["sessions"][session]["close_order"])


def chain_tickers() -> list[str]:
    return [t for t, m in universe_cfg()["tickers"].items() if m.get("chain")]


def ibkr_iv_tickers() -> list[str]:
    return [t for t, m in universe_cfg()["tickers"].items() if m.get("ibkr_iv")]


def vol_index_tickers() -> list[str]:
    return list(universe_cfg()["vol_indices"])


def cross_series_tickers() -> list[str]:
    return list(universe_cfg()["cross_series"])


def sym_key(ticker: str) -> str:
    """Filesystem-safe key for a Yahoo ticker: ^VIX -> VIX, 000660.KS -> 000660_KS, CL=F -> CL_F."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", ticker.replace("^", "")).strip("_")


def block_prefix(block: str) -> str:
    return blocks_cfg()[block]["prefix"]


def toggleable_blocks() -> list[str]:
    return [b for b, meta in blocks_cfg().items() if meta.get("toggleable")]


def block_of_column(col: str) -> str | None:
    """Map a feature column to its block by prefix (longest prefix wins)."""
    best, best_len = None, 0
    for block, meta in blocks_cfg().items():
        p = meta["prefix"]
        if col.startswith(p) and len(p) > best_len:
            best, best_len = block, len(p)
    return best


def ensure_dirs() -> None:
    for d in (DATA_ROOT, OUTPUTS_DIR, RUNS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
