"""On-disk raw repository: parquet per series, manifest, optional pickle snapshot."""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fx_data_collect.config import START_DATE, default_repo_root
from fx_data_collect.sources.fxcm_http import download_instrument_history_http, parquet_symbol
from fx_data_collect.sources.fred_pull import download_fred_panel
from fx_data_collect.sources.yahoo_fx import (
    _naive_normalized_index,
    download_yahoo_fx_daily,
    stitch_fxcm_with_yahoo_tail,
)


def resolve_repo_root() -> Path:
    env = (os.environ.get("FX_DATA_REPO_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return default_repo_root().resolve()


@dataclass
class RawRepository:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.raw_fx = self.root / "raw" / "fx"
        self.raw_fred = self.root / "raw" / "fred"
        self.preview = self.root / "preview"
        for p in (self.raw_fx, self.raw_fred, self.preview):
            p.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> RawRepository:
        return cls(resolve_repo_root())

    def manifest_path(self) -> Path:
        return self.root / "raw" / "manifest.json"

    def save_fx_parquet(self, instrument: str, df: pd.DataFrame) -> Path:
        path = self.raw_fx / f"{parquet_symbol(instrument)}.parquet"
        df.to_parquet(path)
        return path

    def save_fred_parquet(self, df: pd.DataFrame) -> Path:
        path = self.raw_fred / "fred_panel.parquet"
        df.to_parquet(path)
        return path

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        p = self.manifest_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p

    def pickle_snapshot(self, fx_frames: dict[str, pd.DataFrame], fred: pd.DataFrame) -> Path:
        snap = {
            "fx": {k: v for k, v in fx_frames.items()},
            "fred": fred,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "start_date": START_DATE,
        }
        path = self.root / "raw" / "repository_snapshot.pkl"
        with open(path, "wb") as f:
            pickle.dump(snap, f, protocol=5)
        return path


def collect_all_fx(
    pairs: list[str],
    yahoo_map: dict[str, str],
    *,
    repo: RawRepository,
    min_date: str = START_DATE,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    min_ts = pd.Timestamp(min_date)
    frames: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}

    for pair in pairs:
        fxcm = download_instrument_history_http(pair)
        y_sym = yahoo_map.get(pair)
        if not y_sym:
            raise KeyError(f"No Yahoo ticker mapping for {pair}")

        stitched, m = stitch_fxcm_with_yahoo_tail(fxcm, y_sym, min_date=min_ts)
        stitched = _naive_normalized_index(stitched.sort_index())
        stitched = stitched[~stitched.index.duplicated(keep="last")]
        # FXCM yearly candledata may start mid-history (~2012); prepend Yahoo back to START_DATE.
        if len(stitched) and stitched.index.min() > min_ts:
            head_end = stitched.index.min() - pd.Timedelta(days=1)
            if head_end >= min_ts:
                y_head = download_yahoo_fx_daily(y_sym, min_ts, head_end)
                if len(y_head):
                    m["yahoo_prepend_rows"] = int(len(y_head))
                    stitched = _naive_normalized_index(pd.concat([y_head, stitched]))
                    stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index()
        stitched = stitched.loc[stitched.index >= min_ts]
        frames[pair] = stitched
        meta[pair] = {
            **m,
            "rows": len(stitched),
            "start": str(stitched.index.min()) if len(stitched) else None,
            "end": str(stitched.index.max()) if len(stitched) else None,
            "path": str(repo.save_fx_parquet(pair, stitched)),
        }

    return frames, meta


def collect_fred_optional(repo: RawRepository, *, skip: bool) -> tuple[pd.DataFrame, dict]:
    if skip:
        return pd.DataFrame(), {"skipped": True}
    df = download_fred_panel(observation_start=START_DATE)
    path = repo.save_fred_parquet(df)
    return df, {
        "skipped": False,
        "path": str(path),
        "rows": len(df),
        "start": str(df.index.min()) if len(df) else None,
        "end": str(df.index.max()) if len(df) else None,
        "cols": list(df.columns),
    }
