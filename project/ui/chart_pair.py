"""Plotly charts for cointegrated ETF pairs."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from analysis.cointegration import estimate_half_life_hours, ols_log_spread


class PairLike(Protocol):
    etf_y: str
    etf_x: str
from data.yahoo import last_n_calendar_days


def add_weekend_shading(
    fig: go.Figure,
    tmin: pd.Timestamp,
    tmax: pd.Timestamp,
    rows: tuple[int, ...] = (1, 2),
    *,
    max_weekend_bands: int = 160,
) -> None:
    """
    Pink Sat–Sun bands. Uses a stride so long histories do not add thousands of vrects
    (that freezes Plotly / Streamlit).
    """
    ts = pd.Timestamp(tmin).normalize()
    te = pd.Timestamp(tmax).normalize()
    if te <= ts:
        return
    first_sat = ts + pd.Timedelta(days=(5 - ts.weekday()) % 7)
    n_sat = max(1, (te - first_sat).days // 7 + 1)
    step_weeks = max(1, (n_sat + max_weekend_bands - 1) // max_weekend_bands)
    week_step = pd.Timedelta(days=7 * step_weeks)
    d = first_sat
    n = 0
    while d <= te + pd.Timedelta(days=1) and n < max_weekend_bands:
        x0 = d
        x1 = d + pd.Timedelta(days=2)
        for row in rows:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="rgba(255, 182, 193, 0.22)",
                line_width=0,
                layer="below",
                row=row,
                col=1,
            )
        d += week_step
        n += 1


def aligned_log_closes(
    etf_top: dict[str, pd.DataFrame],
    y_sym: str,
    x_sym: str,
    lookback_days: int | None = None,
) -> tuple[pd.Series, pd.Series] | None:
    """
    Align log closes on inner join.
    `lookback_days=None` uses the full series available in `etf_top`.
    """
    if y_sym not in etf_top or x_sym not in etf_top:
        return None
    dy = etf_top[y_sym]
    dx = etf_top[x_sym]
    if lookback_days is not None:
        dy = last_n_calendar_days(dy, lookback_days)
        dx = last_n_calendar_days(dx, lookback_days)
    if dy.empty or dx.empty or "close" not in dy.columns or "close" not in dx.columns:
        return None
    y = dy["close"].astype(float)
    x = dx["close"].astype(float)
    j = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    if len(j) < 50:
        return None
    return np.log(j["y"]), np.log(j["x"])


def pair_key(pr: PairLike) -> str:
    return f"{pr.etf_y}|{pr.etf_x}"


def find_pair(pairs: list[Any], key: str) -> Any | None:
    for p in pairs:
        if pair_key(p) == key:
            return p
    return None


def make_pair_figure(
    pr: PairLike,
    ly: pd.Series,
    lx: pd.Series,
    *,
    xaxis_label: str = "Time (daily, max history)",
    shade_weekends: bool = True,
) -> tuple[go.Figure, dict[str, Any]]:
    """
    Refit OLS spread on the provided `ly`/`lx` window.
    Percentile bands use this same spread sample.
    """
    spread, alpha_c, beta_c = ols_log_spread(ly, lx)
    spread = spread.dropna()
    vals = spread.values.astype(float)
    p10, p30, p50, p70, p90 = (float(x) for x in np.percentile(vals, [10, 30, 50, 70, 90]))

    ly_n = ly - float(ly.iloc[0])
    lx_n = lx - float(lx.iloc[0])

    t0, t1 = ly.index.min(), ly.index.max()
    span_days = max(0, (t1 - t0).days)
    cur = float(spread.iloc[-1])
    pct_full = float(stats.percentileofscore(vals, cur, kind="rank"))
    hl_chart = estimate_half_life_hours(spread)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.52, 0.48],
        subplot_titles=("Spread (residual)", "Log prices (rebased to first bar)"),
    )

    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            mode="lines",
            name="Spread",
            line=dict(color="#1a535c", width=1),
        ),
        row=1,
        col=1,
    )
    for yv, dash in [
        (p90, "dot"),
        (p70, "dash"),
        (p50, "dash"),
        (p30, "dash"),
        (p10, "dot"),
    ]:
        fig.add_hline(
            y=yv,
            line_dash=dash,
            line_color="rgba(100,100,120,0.55)",
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=ly_n.index,
            y=ly_n.values,
            mode="lines",
            name=f"log {pr.etf_y}",
            line=dict(color="#e76f51", width=1.2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lx_n.index,
            y=lx_n.values,
            mode="lines",
            name=f"log {pr.etf_x}",
            line=dict(color="#264653", width=1.2),
        ),
        row=2,
        col=1,
    )

    if shade_weekends:
        add_weekend_shading(fig, pd.Timestamp(t0), pd.Timestamp(t1))

    fig.update_layout(
        height=720,
        margin=dict(l=48, r=28, t=56, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_xaxes(title_text=xaxis_label, row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=1, col=1)
    fig.update_yaxes(title_text="Δ log price", row=2, col=1)

    info = {
        "beta_chart": beta_c,
        "alpha_chart": alpha_c,
        "n_bars": len(spread),
        "span_days": span_days,
        "half_life_hours_chart": hl_chart,
        "spread_pct_full_window": pct_full,
        "t_start": t0,
        "t_end": t1,
    }
    return fig, info
