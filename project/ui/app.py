"""Forecasting dashboard with lazy-loaded ETF and FX tabs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Streamlit sets sys.path[0] to this file's directory; add project root for `data` / `analysis`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Try common locations (many users put .env in the repo root instead of project/).
for _env_path in (_ROOT / ".env", _ROOT.parent / ".env", Path.cwd() / ".env"):
    load_dotenv(_env_path, override=False)
load_dotenv(override=False)

from analysis.lead_lag_screen import (
    HALF_LIFE_DAYS_MAX,
    HALF_LIFE_DAYS_MIN,
    SCREEN_BARS,
    run_preset_screening,
    screened_to_records,
)
from data.loader import N_BARS_DOWNLOAD
from data.timeframes import BAR_INTERVAL_CHOICES
from data.yahoo import MACRO_SYMBOLS, risk_reversal_status
from ui.chart_pair import aligned_log_closes, find_pair, make_pair_figure, pair_key
from ui.ticker_names import get_display_name
from ui.email_report import send_html_email
from ui.fx_forecast_tab import render_fx_forecast_tab

DEFAULT_EMAIL = "mr.mh.rahmani@gmail.com"

DASHBOARD_TABS = ["Dashboard", "ETF Forecaster", "FX Forecasting", "Analytics"]
ETF_PAGES = ["Market data", "Lead-lag screening", "Charts"]


def _df_to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, float_format=lambda x: f"{x:.4g}" if pd.notna(x) else "")


def _trade_hint(pct: float) -> str:
    if pd.isna(pct):
        return ""
    if pct >= 90:
        return "Spread stretched high vs window — mean-reversion bias toward lower spread (check hedge sign)."
    if pct <= 10:
        return "Spread stretched low vs window — mean-reversion bias toward higher spread (check hedge sign)."
    if pct >= 70:
        return "Elevated vs window."
    if pct <= 30:
        return "Depressed vs window."
    return "Mid-range vs window distribution."


@st.cache_data(ttl=3600, show_spinner=True)
def _load_bundle(source: str, bar_interval: str):
    from data.loader import load_etf_universe

    return load_etf_universe(source, bar_interval, n_bars=N_BARS_DOWNLOAD)


def _init_session() -> None:
    if "pair_results" not in st.session_state:
        st.session_state.pair_results = None
    if "dashboard_nav" not in st.session_state:
        st.session_state.dashboard_nav = DASHBOARD_TABS[0]
    if "section_nav" not in st.session_state:
        st.session_state.section_nav = ETF_PAGES[0]
    # `pending_nav` lets buttons rendered AFTER the section radio request a page
    # change for the next rerun (you can't write to a widget-bound key after the
    # widget has been instantiated in the same run).
    if "pending_nav" in st.session_state:
        target = st.session_state.pop("pending_nav")
        if target in ETF_PAGES:
            st.session_state.section_nav = target
        if target in DASHBOARD_TABS:
            st.session_state.dashboard_nav = target
    if "chart_pair_key" not in st.session_state:
        st.session_state.chart_pair_key: str | None = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = "yfinance"
    if "bar_interval" not in st.session_state:
        st.session_state.bar_interval = "5m"


def _polygon_key_configured() -> bool:
    return bool(
        os.environ.get("POLYGON_API_KEY")
        or os.environ.get("POLYGON_MASSIVE_API_KEY")
        or (st.session_state.get("polygon_api_key_ui") or "").strip()
    )


def _apply_polygon_credentials() -> None:
    """Streamlit secrets → env; sidebar password field → env (so data/polygon.py sees the key)."""
    try:
        sec = getattr(st, "secrets", None)
        if sec is not None and "POLYGON_API_KEY" in sec and not os.environ.get("POLYGON_API_KEY"):
            os.environ["POLYGON_API_KEY"] = str(sec["POLYGON_API_KEY"]).strip()
    except Exception:
        pass
    ui = (st.session_state.get("polygon_api_key_ui") or "").strip()
    if ui:
        os.environ["POLYGON_API_KEY"] = ui


def main() -> None:
    st.set_page_config(page_title="Forecasting Dashboard", layout="wide")
    _init_session()
    _apply_polygon_credentials()

    st.title("Forecasting Dashboard")
    st.radio(
        "Workspace",
        DASHBOARD_TABS,
        horizontal=True,
        label_visibility="collapsed",
        key="dashboard_nav",
    )
    workspace = st.session_state.dashboard_nav

    if workspace == "Dashboard":
        st.subheader("All forecasting and analytics in one place")
        st.write(
            "Use tabs above to open each workflow. Initial load stays fast by deferring heavy data work until you open a tab."
        )
        c1, c2, c3 = st.columns(3)
        c1.info("ETF Forecaster: market data, lead-lag screening, pair charts.")
        c2.info("FX Forecasting: next-day directional model signals and refresh controls.")
        c3.info("Analytics: reserved for future cross-asset and research modules.")
        return

    if workspace == "FX Forecasting":
        render_fx_forecast_tab(default_email=DEFAULT_EMAIL)
        return

    if workspace == "Analytics":
        st.subheader("Analytics")
        st.info("Reserved for future analytical modules.")
        return

    with st.sidebar:
        st.header("Data")
        opts = ["yfinance", "master_polygon", "ibkr"]
        idx = opts.index(st.session_state.data_source) if st.session_state.data_source in opts else 0
        ds = st.selectbox(
            "Data source",
            opts,
            index=idx,
            format_func=lambda x: {
                "yfinance": "Yahoo Finance (yfinance)",
                "master_polygon": "Master (Polygon.io)",
                "ibkr": "Interactive Brokers (IBKR)",
            }[x],
        )
        if ds != st.session_state.data_source:
            st.session_state.data_source = ds
            st.cache_data.clear()
            st.session_state.pair_results = None
            st.session_state.chart_pair_key = None
            st.rerun()

        bi_idx = (
            BAR_INTERVAL_CHOICES.index(st.session_state.bar_interval)
            if st.session_state.bar_interval in BAR_INTERVAL_CHOICES
            else 1
        )
        bi = st.selectbox(
            "Bar interval (each download = last 1000 bars)",
            BAR_INTERVAL_CHOICES,
            index=bi_idx,
            help="1m ≈ ~16.7 h of RTH minutes; 1d ≈ 1000 trading days requested (Yahoo: up to data limits).",
        )
        if bi != st.session_state.bar_interval:
            st.session_state.bar_interval = bi
            st.cache_data.clear()
            st.session_state.pair_results = None
            st.session_state.chart_pair_key = None
            st.rerun()

        if st.session_state.data_source == "master_polygon":
            st.text_input(
                "Polygon / Master API key",
                type="password",
                key="polygon_api_key_ui",
                help="Paste the key from your Polygon (Massive) dashboard. Stored in this session only unless you use .env.",
            )
            st.caption(
                f"Or add one line to **`{_ROOT / '.env'}`** (or repo-root `.env`): `POLYGON_API_KEY=your_key` — then restart the app."
            )
        if st.session_state.data_source == "ibkr":
            st.warning(
                "IBKR historical bars are not wired in this loader yet (pick Yahoo or Polygon). "
                "FXCM demo/live orders use **FX Forecasting → Trade from summary** (ForexConnect `.env` keys)."
            )

        if st.button("Download / refresh data", type="primary"):
            st.cache_data.clear()
            st.session_state.pair_results = None
            st.session_state.chart_pair_key = None
            st.rerun()
        st.markdown(
            f"**Phase 1:** Rank seed ETFs by **30-day average daily volume**; keep **top 100** with distinct underlying exposures. "
            f"Fetch **{N_BARS_DOWNLOAD}** bars at the selected interval for those names plus **preset pair** tickers. "
            "Macro series use **Yahoo** symbols. **Charts** use the same **1000-bar** history as the download."
        )

    _apply_polygon_credentials()

    st.subheader("ETF Forecaster")

    if st.session_state.data_source == "master_polygon" and not _polygon_key_configured():
        st.error(
            "**Master (Polygon)** needs an API key.\n\n"
            "1. Paste your key in the **sidebar** field *Polygon / Master API key*, **or**\n"
            "2. Create **`project/.env`** with: `POLYGON_API_KEY=paste_your_key_here`\n\n"
            "Get a key at [polygon.io/dashboard/keys](https://polygon.io/dashboard/keys). "
            "After saving `.env`, restart Streamlit (or use the sidebar key and press **Download / refresh data**)."
        )
        st.stop()

    try:
        bundle = _load_bundle(st.session_state.data_source, st.session_state.bar_interval)
    except Exception as e:
        st.error(f"Data load failed: {e}")
        st.stop()

    etf_top = bundle["etf_top"]
    etf_raw = bundle["etf_raw"]
    top100 = bundle["top100"]
    macro = bundle["macro"]
    macro_resolved = bundle["macro_resolved"]
    bar_interval = bundle["bar_interval"]
    bar_period = bundle["bar_period"]
    n_bars = bundle["n_bars"]

    st.caption(
        f"Bars: **{bar_interval}**, lookback period **{bar_period}**, **{n_bars}** bars/symbol (source **{st.session_state.data_source}**). "
        f"**Phase 2** uses **{SCREEN_BARS}** overlapping bars: Engle–Granger p<0.05, Granger (leader→follower, lags 1–5), "
        f"half-life **{HALF_LIFE_DAYS_MIN}–{HALF_LIFE_DAYS_MAX}** days (OU-style on the spread). "
        f"**Phase 3:** Z-score on the **full {n_bars}-bar** spread; entry |Z|>2, exit toward |Z|≤0.5."
    )

    pairs = st.session_state.pair_results

    with st.sidebar:
        if pairs:
            passed = [p for p in pairs if p.all_phase2_pass]
            if passed:
                st.divider()
                st.subheader("Chart shortcuts")
                st.caption("Pairs that **passed Phase 2** — opens **Charts**.")
                with st.expander("Passed pairs", expanded=len(passed) <= 12):
                    for i, pr in enumerate(passed):
                        label = f"{pr.etf_y}  ↔  {pr.etf_x}"
                        if st.button(label, key=f"sb_chart_{i}", use_container_width=True):
                            st.session_state.section_nav = "Charts"
                            st.session_state.chart_pair_key = pair_key(pr)
                            st.rerun()

    st.radio(
        "Section",
        ETF_PAGES,
        horizontal=True,
        label_visibility="collapsed",
        key="section_nav",
    )
    page = st.session_state.section_nav

    if page == "Market data":
        ac1, ac2, _ = st.columns([1, 1, 4])
        ac1.markdown("**Asset class dashboards**")
        if ac2.button("Open FX direction forecasts →", use_container_width=True):
            st.session_state.pending_nav = "FX Forecasting"
            st.rerun()

        st.subheader("Top 100 ETFs — 30-day average daily volume (distinct underlying)")
        st.write(
            f"{len(top100)} symbols; **{len(etf_top)}** with usable **{bar_interval}** bars in this run. "
            "Preset pairs use the **top-100 ticker** for each leg’s underlying (e.g. SPY or VOO for large-cap US), not necessarily the preset symbol name."
        )
        st.dataframe(pd.DataFrame({"ticker": top100}), hide_index=True, use_container_width=True)

        st.subheader(f"Macro / volatility ({bar_interval}, last {n_bars} bars)")
        macro_rows = []
        for label in MACRO_SYMBOLS:
            df = macro.get(label, pd.DataFrame())
            sym = macro_resolved.get(label, "")
            macro_rows.append(
                {
                    "name": label,
                    "symbol": sym,
                    "bars": len(df),
                    "last_close": float(df["close"].iloc[-1]) if not df.empty and "close" in df.columns else None,
                }
            )
        st.dataframe(pd.DataFrame(macro_rows), hide_index=True, use_container_width=True)
        st.caption(
            "Treasury indices (^TNX, ^IRX) can be sparse vs futures; **us_2y_yield** falls back to **ZT=F** "
            "(2-year T-note futures price) when 2YY=F has too few intraday points — not the same units as a yield %."
        )

        st.subheader("S&P 500 risk reversals (delta buckets)")
        st.info(
            "Yahoo Finance does not offer historical **risk reversal** series for SPX by delta. "
            "Use CBOE, a prime broker, or an options data vendor; the table below is a placeholder."
        )
        st.dataframe(risk_reversal_status(), hide_index=True, use_container_width=True)

        st.info(
            "**Category 9 (sentiment vs ETF)** has no bundled price series here — use a vendor sentiment feed if you want that test."
        )

    elif page == "Lead-lag screening":
        st.subheader("Preset pairs — Phase 2 & 3")
        st.caption(
            "Structural pairs (size, supply chain, factors, rates, geography, etc.). "
            "Each pair runs on **150** aligned bars at your selected interval, then Z-score on the **full downloaded** window. "
            "If the preset says **SPY** but your top 100 kept **VOO** (same S&P 500 bucket), the screen uses **VOO** for data."
        )
        if st.button("Run lead-lag screening"):
            with st.spinner("Cointegration, Granger, half-life, Z-score…"):
                st.session_state.pair_results = run_preset_screening(
                    etf_raw,
                    top100,
                    bar_interval,
                )
            st.session_state.chart_pair_key = None
            n_pass = sum(1 for p in (st.session_state.pair_results or []) if p.all_phase2_pass)
            st.success(
                f"Evaluated {len(st.session_state.pair_results or [])} preset pair(s) with both legs in top 100; "
                f"**{n_pass}** passed Phase 2."
            )
            st.rerun()

        pairs = st.session_state.pair_results
        if pairs:
            recs = screened_to_records(pairs)
            df = pd.DataFrame(recs)
            show = df[
                [
                    "category",
                    "preset_sym_a",
                    "preset_sym_b",
                    "etf_y",
                    "etf_x",
                    "leader",
                    "follower",
                    "cointegration_pvalue",
                    "coint_pass",
                    "granger_min_p",
                    "granger_pass",
                    "half_life_days",
                    "half_life_pass",
                    "phase2_all_pass",
                    "hedge_beta",
                    "zscore_1000bar",
                    "phase3_signal",
                ]
            ]
            st.dataframe(show, hide_index=True, use_container_width=True)

            st.markdown(
                "**Phase 2:** Engle–Granger **p < 0.05**; Granger **leader → follower** returns with min p < 0.05 over lags 1–5; "
                f"half-life in **[{HALF_LIFE_DAYS_MIN}, {HALF_LIFE_DAYS_MAX}]** days (from AR(1) on the spread, scaled by bar size). "
                "**Phase 3:** Z = (spread − mean) / std on the **full** bar window; entries |Z| > 2; exit toward mean (|Z| ≤ 0.5 or 0)."
            )

            email_to = os.environ.get("EMAIL_TO", DEFAULT_EMAIL)
            st.caption(f"Email recipient: **{email_to}** (override with `EMAIL_TO` in `.env`).")

            if st.button("Email results to me"):
                try:
                    html = "<h3>Lead-lag screening</h3>" + _df_to_html_table(show)
                    text = show.to_string(index=False)
                    send_html_email(
                        to_addr=email_to,
                        subject="ETF Forecaster — lead-lag screening",
                        html_body=html,
                        text_body=text,
                    )
                    st.success("Sent.")
                except Exception as e:
                    st.error(str(e))
        elif st.session_state.pair_results is not None:
            st.warning(
                "No preset pairs produced a result: missing data, overlapping underlyings, or not enough "
                "aligned bars. (Preset tickers are mapped to your **top-100** fund per underlying — e.g. **SPY** → **VOO** if VOO ranked higher.)"
            )
        else:
            st.info("Press **Run lead-lag screening** after data load.")

    else:
        st.subheader("Pair charts (1000-bar window)")
        pairs = st.session_state.pair_results
        if not pairs:
            st.info("Run **Run lead-lag screening** on the Lead-lag tab first.")
        else:
            options = [pair_key(p) for p in pairs]
            if st.session_state.chart_pair_key not in options:
                st.session_state.chart_pair_key = options[0]

            def _fmt_pair(k: str) -> str:
                a, b = k.split("|", 1)
                pr = find_pair(pairs, k)
                if pr is None:
                    return f"{a} ↔ {b}"
                cat = (pr.category[:37] + "…") if len(pr.category) > 40 else pr.category
                return f"{a} ↔ {b} — {cat}"

            choice = st.selectbox(
                "Pair",
                options,
                index=options.index(st.session_state.chart_pair_key),
                format_func=_fmt_pair,
            )
            st.session_state.chart_pair_key = choice

            pr = find_pair(pairs, choice)
            if pr is None:
                st.error("Pair not found.")
            else:
                ny = get_display_name(pr.etf_y)
                nx = get_display_name(pr.etf_x)
                st.markdown(f"### {ny} (`{pr.etf_y}`)  ↔  {nx} (`{pr.etf_x}`)")
                st.caption(pr.category)
                st.caption(
                    f"Using **{n_bars}** bars at **{bar_interval}** (same as download). "
                    "Pink bands: weekends (sampled so the chart stays responsive)."
                )

                al = aligned_log_closes(etf_raw, pr.etf_y, pr.etf_x, lookback_days=None)
                if al is None:
                    st.warning("Could not align prices for this pair (missing bars?).")
                else:
                    ly, lx = al
                    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                    c1.metric("Coint p", f"{pr.coint_pvalue:.4f}")
                    c2.metric("Granger min p", f"{pr.granger_min_pvalue:.4f}" if pr.granger_min_pvalue is not None else "—")
                    c3.metric("Half-life d", f"{pr.half_life_days:.2f}" if pr.half_life_days is not None else "—")
                    c4.metric("Phase 2 pass", "Yes" if pr.all_phase2_pass else "No")
                    c5.metric("β (log)", f"{pr.hedge_beta:.4f}")
                    c6.metric("Z (window)", f"{pr.zscore:.2f}" if pd.notna(pr.zscore) else "—")
                    c7.metric("Coint OK", "Yes" if pr.coint_pass else "No")

                    st.markdown(pr.phase3_signal)

                    try:
                        fig, cinfo = make_pair_figure(
                            pr,
                            ly,
                            lx,
                            xaxis_label=f"Time ({bar_interval}, {len(ly)} bars)",
                            shade_weekends=True,
                        )
                        try:
                            st.plotly_chart(fig, width="stretch")
                        except TypeError:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Chart failed to render: {e}")

                    st.markdown(
                        f"**Chart window** — {cinfo['n_bars']:,} bars, ~{cinfo['span_days']} calendar days, "
                        f"{cinfo['t_start']} → {cinfo['t_end']}"
                    )
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("β (chart)", f"{cinfo['beta_chart']:.4f}")
                    d2.metric("Half-life h (AR1)", f"{cinfo['half_life_hours_chart']:.1f}" if cinfo["half_life_hours_chart"] is not None else "—")
                    d3.metric("Spread %ile", f"{cinfo['spread_pct_full_window']:.1f}")
                    d4.metric("α (log)", f"{cinfo['alpha_chart']:.4g}")
                    st.caption(_trade_hint(cinfo["spread_pct_full_window"]))


def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_inside_streamlit():
        main()
    else:
        import subprocess

        app_path = Path(__file__).resolve()
        project_root = Path(__file__).resolve().parents[1]
        print("Starting Streamlit (run from project root so imports work).")
        print("Tip: py -m streamlit run ui/app.py\n")
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", str(app_path)],
                cwd=str(project_root),
            )
        )
