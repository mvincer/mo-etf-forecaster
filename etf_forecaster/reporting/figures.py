"""Matplotlib figures for the PDF report (returned as PNG bytes)."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def ablation_heatmap(ablation: pd.DataFrame, horizon: int) -> bytes | None:
    s1 = ablation[(ablation["stage"] == "s1_screen") & (ablation["horizon"] == horizon)]
    if not len(s1):
        return None
    base = s1[s1["blocks"] == "PRICE_only"].set_index("group")["log_loss"]
    rest = s1[s1["blocks"] != "PRICE_only"].copy()
    rest["gain"] = rest.apply(lambda r: base.get(r["group"], np.nan) - r["log_loss"], axis=1)
    pivot = rest.pivot_table(index="blocks", columns="group", values="gain", aggfunc="mean")
    if pivot.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(pivot) + 1.5))
    vmax = np.nanmax(np.abs(pivot.values)) or 1e-4
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.4f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Stage-1 marginal block value (Δ log-loss vs PRICE alone), h={horizon}")
    fig.colorbar(im, ax=ax, shrink=0.7)
    return _png(fig)


def calibration_curve(hist: pd.DataFrame, horizon: int) -> bytes | None:
    g = hist[(hist["model"] == "champion") & (hist["horizon"] == horizon)].dropna(subset=["y"])
    if len(g) < 200:
        return None
    bins = np.clip((g["p_up"] * 10).astype(int), 0, 9)
    rel = g.groupby(bins).agg(p=("p_up", "mean"), y=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(rel["p"], rel["y"], "o-", color="#3355bb")
    for _, r in rel.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["p"], r["y"]), fontsize=6,
                    textcoords="offset points", xytext=(4, -8))
    ax.set_xlabel("forecast P(up)")
    ax.set_ylabel("observed up-rate")
    ax.set_title(f"Reliability, champion series, h={horizon} ({len(g)} OOS preds)")
    return _png(fig)


def rolling_hit_rate(hist: pd.DataFrame, horizon: int) -> bytes | None:
    g = hist[(hist["model"] == "champion") & (hist["horizon"] == horizon)].dropna(subset=["y"])
    if len(g) < 300:
        return None
    g = g.sort_values("date")
    g["date"] = pd.to_datetime(g["date"])
    daily = g.groupby("date").apply(
        lambda x: float(((x["p_up"] > 0.5) == (x["y"] > 0.5)).mean()), include_groups=False)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(daily.index, daily.rolling(126, min_periods=40).mean(), color="#3355bb", lw=1.2)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_title(f"126-day rolling OOS hit rate pooled across tickers, h={horizon}")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    return _png(fig)


def hit_by_ticker(scorecard: pd.DataFrame, horizon: int) -> bytes | None:
    sc = scorecard[(scorecard["model"] == "champion") & (scorecard["window"] == "all")
                   & (scorecard["horizon"] == horizon)]
    if not len(sc):
        return None
    sc = sc.sort_values("hit_rate")
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(sc) + 1.2))
    colors = ["#33aa55" if f else "#aabbcc" for f in sc.get("fdr_pass", [False] * len(sc))]
    ax.barh(sc["ticker"], sc["hit_rate"] - 0.5, left=0.5, color=colors)
    ax.axvline(0.5, color="gray", lw=1)
    ax.set_xlim(0.40, max(0.66, sc["hit_rate"].max() + 0.02))
    ax.set_title(f"OOS hit rate by ticker, h={horizon} "
                 "(green = survives BH-FDR at 10%)")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.tick_params(labelsize=7)
    return _png(fig)
