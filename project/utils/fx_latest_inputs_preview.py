"""Load FX pipeline inputs from ``fx_data_collect`` for dashboard inspection."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_fx_collect_path(etf_root: Path | None) -> bool:
    if etf_root is None or not (etf_root / "fx_data_collect").is_dir():
        return False
    root_pkg = str(etf_root)
    if root_pkg not in sys.path:
        sys.path.insert(0, root_pkg)
    return True


def load_fx_input_preview(
    etf_root: Path | None,
    *,
    raw_tail: int = 30,
    panel_tail: int = 15,
) -> dict[str, Any]:
    """Return dataframes / messages for everything merged into the FX repo panel."""
    out: dict[str, Any] = {"errors": []}
    if not _ensure_fx_collect_path(etf_root):
        out["errors"].append("ETF project root or fx_data_collect not found.")
        return out
    try:
        from fx_data_collect.config import FX_PAIRS  # noqa: PLC0415
        from fx_data_collect.repository import resolve_repo_root  # noqa: PLC0415
    except Exception as e:
        out["errors"].append(f"Import fx_data_collect: {e}")
        return out
    try:
        repo = resolve_repo_root()
    except Exception as e:
        out["errors"].append(f"resolve_repo_root: {e}")
        return out

    mi = repo / "model_inputs"
    f_path = mi / "fundamentals_daily.parquet"
    t_path = mi / "technicals_daily.parquet"
    man_path = mi / "manifest.json"

    for label, path in (
        ("fundamentals_daily", f_path),
        ("technicals_daily", t_path),
    ):
        if not path.is_file():
            out["errors"].append(f"Missing {path}")
            out[label] = None
            continue
        try:
            df = pd.read_parquet(path)
            out[label] = df.tail(int(panel_tail))
            out[f"{label}_shape"] = df.shape
            out[f"{label}_index_min"] = str(df.index.min()) if len(df) else ""
            out[f"{label}_index_max"] = str(df.index.max()) if len(df) else ""
        except Exception as e:
            out["errors"].append(f"{label}: {e}")
            out[label] = None

    if man_path.is_file():
        try:
            out["manifest"] = json.loads(man_path.read_text(encoding="utf-8"))
        except Exception as e:
            out["errors"].append(f"manifest.json: {e}")
            out["manifest"] = None
    else:
        out["manifest"] = None
        out["errors"].append(f"Missing {man_path}")

    raw_frames: dict[str, pd.DataFrame] = {}
    for inst in FX_PAIRS:
        pid = inst.replace("/", "_").replace(" ", "_")
        p = repo / "raw" / "fx" / f"{pid}.parquet"
        if not p.is_file():
            raw_frames[inst] = pd.DataFrame()
            continue
        try:
            raw_frames[inst] = pd.read_parquet(p).tail(int(raw_tail))
        except Exception as e:
            out["errors"].append(f"{inst}: {e}")
            raw_frames[inst] = pd.DataFrame()
    out["raw_fx"] = raw_frames
    out["repo_root"] = str(repo)
    return out
