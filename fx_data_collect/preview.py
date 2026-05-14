"""CSV summaries + matplotlib PNG for quick human inspection."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def parquet_slug(pair: str) -> str:
    return pair.replace("/", "_")


def summary_table(fx_meta: dict[str, dict], fred_meta: dict) -> pd.DataFrame:
    rows = []
    for pair, m in fx_meta.items():
        rows.append(
            {
                "series": pair,
                "kind": "fx",
                "rows": m.get("rows"),
                "start": m.get("start"),
                "end": m.get("end"),
                "fxcm_rows": m.get("fxcm_rows"),
                "yahoo_rows": m.get("yahoo_rows"),
                "yahoo_only": m.get("yahoo_only"),
                "path": m.get("path"),
            }
        )
    rows.append(
        {
            "series": "fred_panel",
            "kind": "fred",
            "rows": fred_meta.get("rows"),
            "start": fred_meta.get("start"),
            "end": fred_meta.get("end"),
            "fxcm_rows": None,
            "yahoo_rows": None,
            "yahoo_only": None,
            "path": fred_meta.get("path"),
        }
    )
    return pd.DataFrame(rows)


def merged_preview_sample(
    fx_frames: dict[str, pd.DataFrame],
    fred: pd.DataFrame,
    *,
    primary_pair: str,
    last_n: int = 2500,
) -> pd.DataFrame:
    """Left calendar join: primary FX Close + key FRED columns (readable CSV)."""
    if primary_pair not in fx_frames:
        raise KeyError(primary_pair)
    fx = fx_frames[primary_pair][["Close"]].rename(columns={"Close": f"fx_close_{parquet_slug(primary_pair)}"})
    fx.index = pd.DatetimeIndex(pd.to_datetime(fx.index)).normalize()

    if len(fred.columns):
        want = ("us_10y", "us_2y", "dxy", "eu_", "jp_", "idx_us_sp500")
        subfred_cols = [c for c in fred.columns if any(x in c for x in want)]
        fr = fred[subfred_cols]
    else:
        fr = pd.DataFrame()

    merged = fx.join(fr, how="left").sort_index()
    if last_n and len(merged) > last_n:
        merged = merged.tail(int(last_n))
    return merged


def write_preview_chart(
    merged: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> Path:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("Install matplotlib for PNG preview: pip install matplotlib") from e

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    fx_cols = [c for c in merged.columns if c.startswith("fx_close")]
    if fx_cols:
        axes[0].plot(merged.index, merged[fx_cols[0]].values, color="#1f77b4", lw=1.2, label=fx_cols[0])
        axes[0].set_ylabel("FX close")
        axes[0].legend(loc="upper left")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)

    macro_cols = [c for c in merged.columns if c.startswith("fund_")]
    if macro_cols:
        for c in macro_cols[:3]:
            axes[1].plot(merged.index, merged[c].values, lw=1.0, label=c.replace("fund_", ""))
        axes[1].set_ylabel("FRED (levels)")
        axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.25)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
