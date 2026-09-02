"""Generate ETF_Forecaster_Methodology_and_Results.pdf.

A living document: methodology (design of record) + data dictionary + feature blocks +
search design + ablation results + per-ticker performance + track-record state.
Written to C:\\Dev\\Mo_Dash\\docs\\etf_forecaster\\ with a mirror copy in this repo.

Run:  python -m etf_forecaster.reporting.master_report
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from etf_forecaster import store
from etf_forecaster.config import (
    REPO_ROOT,
    blocks_cfg,
    macro_cfg,
    mo_dash_root,
    search_cfg,
    universe_cfg,
)
from etf_forecaster.data import lake
from etf_forecaster.reporting import figures

log = logging.getLogger(__name__)

_STYLES = getSampleStyleSheet()
H1 = ParagraphStyle("h1x", parent=_STYLES["Heading1"], spaceBefore=18)
H2 = ParagraphStyle("h2x", parent=_STYLES["Heading2"], spaceBefore=12)
BODY = ParagraphStyle("bodyx", parent=_STYLES["BodyText"], fontSize=9.2, leading=12.5)
SMALL = ParagraphStyle("smallx", parent=_STYLES["BodyText"], fontSize=7.6, leading=9.5)

_TABLE_STYLE = TableStyle([
    ("FONTSIZE", (0, 0), (-1, -1), 7),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4f0")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a0a8b8")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fa")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
])


def _p(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def _table(df: pd.DataFrame, col_widths=None, max_rows: int = 45) -> Table:
    df = df.head(max_rows)
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(_TABLE_STYLE)
    return t


def _img(png: bytes | None, width: float = 6.8 * inch):
    if png is None:
        return None
    import io
    from PIL import Image as PILImage
    pil = PILImage.open(io.BytesIO(png))
    w, h = pil.size
    return Image(io.BytesIO(png), width=width, height=width * h / w)


# ------------------------------------------------------------------ sections

_METHODOLOGY = """
<b>Objective.</b> For each of 40 ETFs/indices, forecast the probability that the close in
h trading days (h = 1..5) is above today's close, plus a conformal quantile distribution of
the h-day log return, and maintain an audited out-of-sample track record.
<br/><br/>
<b>Data.</b> Full-history daily bars (yfinance) for the universe plus the CBOE volatility
complex (VIX, VIX9D, VIX3M, VIX6M, VVIX, SKEW, VXN, VXD, OVX, GVZ) and 10 cross-asset
series; 14 fast FRED series (yields, curve, credit OAS, dollar, oil, NFCI, Fed balance
sheet, SOFR); 24 slow macro series pulled as point-in-time ALFRED vintages (CPI, PCE,
INDPRO, capacity utilization, payrolls, claims, M2, GDP, housing, sentiment, NBER dates);
Shiller S&amp;P 500 earnings/CAPE; per-ticker 30-day implied and historical volatility from
IBKR; and a daily option-chain snapshot out to 120 DTE from which 25-delta risk reversal,
butterfly, ATM term structure and a dealer-gamma proxy are computed (this block accrues
history forward from first snapshot).
<br/><br/>
<b>Point-in-time discipline.</b> Slow macro features are computed from the series exactly
as known on each historical date via ALFRED vintages (fallback: conservative publication
lags). Weekly FRED series are lagged one week. Korean closes are same-day usable for US
targets but US closes are lagged one day for Korean targets. Weekly/monthly chart patterns
only become visible the first daily bar after their (completed) higher-timeframe bar ends.
Shiller earnings are lagged three months.
<br/><br/>
<b>Features.</b> ~690 columns per ticker in 12 blocks: PRICE (always on; returns, gaps,
range-vol estimators, last-60-candle block), TECH (20+ indicators), PATTERN_D1 and
PATTERN_HTF (16 chart-pattern detectors - double/triple tops and bottoms, head and
shoulders, channels, S/R, wedges, flags, trendlines and breaks - run on daily, weekly and
monthly bars, converted to per-bar features with direction decay, trigger flags, ATR
distances and cross-timeframe agreement), REGIME (ADX state, HH/HL market structure,
variance ratio, MA posture, vol terciles, leak-free Markov-switching probability), VOLOPT
(VIX complex, term slopes, IBKR IV, VRP, HAR-RV, EWMA/GARCH), SKEW (chain-derived),
MACRO_FAST, MACRO_SLOW (vintage-aligned), EARNINGS, CROSS (session-aligned lead-lag,
relative strength, beta, group momentum rank), SEASON (calendar and event days).
<br/><br/>
<b>Search.</b> A staged include/exclude ablation over the 11 toggleable blocks per
(ticker-group, horizon): stage 1 screens each block against a PRICE-only floor; stage 2
greedy forward/backward selection; stage 3 exhaustive 2^k over the shortlist; stage 4
model/hyperparameter search (logistic, elastic-net, LightGBM, XGBoost, CatBoost, RF/ET/HGB,
LSTM/TCN over the candle tensor; Optuna TPE in full mode) including the lag-window X in
{10, 20, 40, 60}; stage 5 stacking of the top configurations with isotonic calibration.
All selection is nested in the development slice (first 60% of history). Every evaluated
cell is persisted with its running trial count.
<br/><br/>
<b>Validation.</b> Purged, embargoed, expanding walk-forward: training data for a refit
forecasting date t ends h bars before t (both purge and embargo for overlapping h-day
labels). The out-of-sample track record consists exclusively of predictions dated after
the development slice; champion switching along the way uses only trailing OOS data
available at that moment (sticky rule). Magnitude quantiles are conformalized (CQR) on a
held-out calibration slice targeting 80% coverage for the q10-q90 band. Reported results
carry Benjamini-Hochberg FDR flags and binomial p-values; hit rates should be read net of
the recorded trial counts.
<br/><br/>
<b>Honest expectations.</b> Daily equity-index direction is a low-signal problem: a
realistic sustained OOS hit rate is 52-56%, not 70%. VIX-family tickers score much higher
hit rates for structural reasons (persistent downward drift of VIXY, mean reversion of
VIX), which the probability calibration reflects. The chain-derived SKEW block has
essentially no history yet and will only earn a place in the ablation as snapshots accrue.
"""


def _data_dictionary() -> list:
    parts = [_p("2. Data dictionary and coverage", H1)]
    cov = lake.read_df(lake.DATA_ROOT / "coverage_report.parquet")
    if cov is not None and len(cov):
        cov = cov.reset_index()[["ticker", "rows", "start", "end"]]
        parts.append(_p("Daily bars (yfinance), full available history:", BODY))
        parts.append(_table(cov, max_rows=70))
    man = lake.manifest()
    rows = [{"artifact": k, "rows": v.get("rows"), "start": v.get("start"),
             "end": v.get("end")} for k, v in sorted(man.items())
            if not k.startswith("bars/") and not k.startswith("features/")]
    if rows:
        parts.append(Spacer(1, 8))
        parts.append(_p("Other artifacts (vol indices, macro, vintages, IBKR IV, skew):", BODY))
        parts.append(_table(pd.DataFrame(rows), max_rows=80))
    mc = macro_cfg()
    parts.append(Spacer(1, 8))
    parts.append(_p(
        f"Fast FRED series: {', '.join(mc['fred_fast'].keys())}.<br/>"
        f"ALFRED vintage series: {', '.join(mc['alfred_slow'].keys())}.", SMALL))
    return parts


def _blocks_section() -> list:
    parts = [_p("3. Feature blocks", H1)]
    rows = []
    counts = {}
    feats = lake.read_df(lake.features_path("SPY"))
    if feats is not None:
        from etf_forecaster.config import block_of_column
        for c in feats.columns:
            b = block_of_column(c)
            counts[b] = counts.get(b, 0) + 1
    for name, meta in blocks_cfg().items():
        rows.append({"block": name, "prefix": meta["prefix"],
                     "toggleable": meta["toggleable"], "columns (SPY)": counts.get(name, "-"),
                     "description": meta["desc"]})
    parts.append(_table(pd.DataFrame(rows), col_widths=[70, 40, 50, 55, 260]))
    return parts


def _search_section() -> list:
    cfg = search_cfg()
    parts = [_p("4. Grid-search design and budgets", H1)]
    parts.append(_p(
        f"Horizons: {cfg['horizons']}. Walk-forward: min_train "
        f"{cfg['walkforward']['min_train_bars']} bars, refit every "
        f"{cfg['walkforward']['refit_every']} (full) / {cfg['fast']['outer_refit_every']} "
        f"(fast) bars, embargo = horizon, calibration slice "
        f"{cfg['walkforward']['calibration_frac']:.0%}. Ablation candidates: "
        f"{', '.join(cfg['ablation']['candidate_blocks'])}. Greedy min gain "
        f"{cfg['ablation']['greedy_min_gain']}, exhaustive cap 2^"
        f"{cfg['ablation']['exhaustive_max_blocks']}. Optuna trials per cell (full): "
        f"{cfg['model_search']['n_trials']}. Window grid: "
        f"{cfg['model_search']['window_grid']}. Champion metric {cfg['selection']['metric']}"
        f" with sticky margin {cfg['selection']['sticky_margin']}; BH-FDR at "
        f"{cfg['multiple_testing']['fdr_alpha']:.0%}.", BODY))
    abl = store.read(store.ABLATION)
    if abl is not None and len(abl):
        parts.append(_p(
            f"Ablation cells evaluated to date: <b>{len(abl)}</b> across "
            f"{abl['group'].nunique()} groups x {abl['horizon'].nunique()} horizons "
            f"({abl['stage'].value_counts().to_dict()}).", BODY))
    return parts


def _ablation_section() -> list:
    parts = [_p("5. Ablation results - which blocks earn their keep", H1)]
    abl = store.read(store.ABLATION)
    if abl is None or not len(abl):
        parts.append(_p("No ablation results yet - run train_search.", BODY))
        return parts
    for h in sorted(abl["horizon"].unique()):
        img = _img(figures.ablation_heatmap(abl, int(h)))
        if img is not None:
            parts.append(img)
            parts.append(Spacer(1, 6))
    s3 = abl[abl["stage"] == "s3_exhaustive"]
    if len(s3):
        best = s3.loc[s3.groupby(["group", "horizon"])["log_loss"].idxmin()]
        best = best[["group", "horizon", "blocks", "log_loss", "hit_rate", "n",
                     "n_trials_so_far"]].round(4).sort_values(["group", "horizon"])
        parts.append(_p("Winning block sets (stage-3 exhaustive):", BODY))
        parts.append(_table(best, max_rows=40))
    return parts


def _performance_section() -> list:
    parts = [_p("6. Out-of-sample performance", H1)]
    hist = store.read(store.HISTORY)
    sc = store.read(store.SCORECARD)
    if hist is None or not len(hist):
        parts.append(_p("No OOS history yet.", BODY))
        return parts
    champ = hist[hist["model"] == "champion"].dropna(subset=["y"])
    parts.append(_p(
        f"Track record: <b>{len(champ):,}</b> matured champion-series OOS predictions "
        f"across {champ['ticker'].nunique()} tickers, from "
        f"{pd.to_datetime(champ['date']).min().date()} to "
        f"{pd.to_datetime(champ['date']).max().date()}. Every prediction ever made is "
        "stored in predictions_history.parquet with its realized outcome.", BODY))
    for h in (1, 5):
        for fn in (figures.rolling_hit_rate, figures.calibration_curve, figures.hit_by_ticker):
            arg = hist if fn is not figures.hit_by_ticker else sc
            if arg is None:
                continue
            img = _img(fn(arg, h))
            if img is not None:
                parts.append(img)
                parts.append(Spacer(1, 6))
    if sc is not None and len(sc):
        pooled = sc[(sc["model"] == "champion") & (sc["window"] == "all")]
        summary = pooled.groupby("horizon").agg(
            tickers=("ticker", "nunique"), n=("n", "sum"), hit=("hit_rate", "mean"),
            log_loss=("log_loss", "mean"), brier=("brier", "mean"), auc=("auc", "mean"),
            fdr_pass=("fdr_pass", "sum")).round(4).reset_index()
        parts.append(_p("Champion-series summary by horizon (mean across tickers):", BODY))
        parts.append(_table(summary))
    return parts


def _limitations() -> list:
    return [
        _p("7. Known limitations", H1),
        _p(
            "(1) The chain-derived SKEW block starts with a single day of history; its value "
            "is unmeasurable for months. (2) IBKR implied vol covers ~15 years for large ETFs "
            "and less for younger ones; earlier history has no per-ticker IV, only the index "
            "complex. (3) ALFRED vintages typically begin 1996-1998; before that, publication-"
            "lag shifts are used. (4) The fast-mode backfill uses reduced budgets (annual "
            "refits in the nested search, quarterly in the outer walk-forward); the scheduled "
            "monthly full ablation re-runs everything at full budget and results supersede. "
            "(5) Sequence models (LSTM/TCN) participate only in the full-budget roster. "
            "(6) A ~52-55% pooled hit rate on daily horizons is expected; treat any ticker "
            "cell that has not survived BH-FDR as noise. (7) ^GSPTSE has no options and the "
            ".KS/.TO names have thinner coverage; their SKEW/VOLOPT blocks degrade to the "
            "index-level complex.", BODY),
    ]


def generate(out_name: str = "ETF_Forecaster_Methodology_and_Results.pdf") -> Path:
    dest_dir = mo_dash_root() / "docs" / "etf_forecaster"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / out_name

    doc = SimpleDocTemplate(str(dest), pagesize=letter, topMargin=0.7 * inch,
                            bottomMargin=0.7 * inch, leftMargin=0.8 * inch,
                            rightMargin=0.8 * inch, title="ETF Forecaster - Methodology and Results")
    story: list = [
        _p("ETF Forecaster - Methodology and Results", _STYLES["Title"]),
        _p(f"Generated {datetime.now():%Y-%m-%d %H:%M} - mo-etf-forecaster - "
           f"universe of {len(universe_cfg()['tickers'])} tickers, horizons 1-5 days", BODY),
        Spacer(1, 10),
        _p("1. Methodology (design of record)", H1),
        _p(_METHODOLOGY, BODY),
        PageBreak(),
    ]
    story += _data_dictionary()
    story.append(PageBreak())
    story += _blocks_section()
    story += _search_section()
    story.append(PageBreak())
    story += _ablation_section()
    story.append(PageBreak())
    story += _performance_section()
    story += _limitations()
    doc.build(story)

    mirror = REPO_ROOT / "docs"
    mirror.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(dest, mirror / out_name)
    log.info("PDF written to %s (mirror in %s)", dest, mirror)
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate()


if __name__ == "__main__":
    main()
