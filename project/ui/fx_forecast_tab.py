"""FX direction forecasts tab for the ETF Forecaster dashboard.

Displays the precomputed Excel produced by the FX project's daily report. The tab can
also kick off a fresh refresh (``--inject-live-quote --align-to-next-bar``) in a
background subprocess so each open of the dashboard sees a forecast for tomorrow's
bar against the latest live FX spot.

The dashboard does NOT import FX project Python modules directly (avoids pulling
heavy ML deps); it reads the Excel file and shells out for refreshes.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.email_report import send_html_email
from utils.auto_trade_config import AutoTradeConfig, default_config_path, load_config, save_config
from utils.fx_currency_exposure import (
    currency_leg_counts,
    pair_likelihood_scores,
    plan_equal_bias_orders,
)
from utils.fx_latest_inputs_preview import load_fx_input_preview
from utils.fx_signal_side import classify_signal_side
from utils.mo_dash_layout import (
    fx_data_collect_import_cwd,
    fx_data_collect_package_dir,
    fx_nl_project_root,
    mo_dash_fx_forecast_xlsx,
    mo_dash_workspace_root,
)
from utils.fxcm_forexconnect_trade import fc_pair_display, fc_settings_from_env


def _fxcm_fc_module_reload():
    """Reload FXCM trade helpers on each action so Streamlit never keeps a stale ``execute_fc_market_order``."""
    import utils.fxcm_forexconnect_trade as m

    return importlib.reload(m)

_ETF_PROJECT_ROOT_DEFAULTS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[2],
    Path(r"C:\Dev\Mo_Dash\ETF\ETF Forecaster"),
)


def _fx_project_root_defaults() -> tuple[Path, ...]:
    """Candidate FX non-linear project locations, Mo_Dash first."""
    out: list[Path] = []
    seen: set[str] = set()
    for er in _ETF_PROJECT_ROOT_DEFAULTS:
        try:
            p = fx_nl_project_root(er)
        except Exception:
            continue
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    for p in (
        Path(__file__).resolve().parents[2] / "FX_NonLinear_Forecast_Direction",
    ):
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return tuple(out)


_FX_PROJECT_ROOT_DEFAULTS: tuple[Path, ...] = _fx_project_root_defaults()


def _fx_forecast_xlsx_candidates() -> tuple[Path, ...]:
    """Prefer Mo_Dash FX output (new path, then legacy path), then workbook next to the FX project."""
    mo_new = tuple(mo_dash_fx_forecast_xlsx(er) for er in _ETF_PROJECT_ROOT_DEFAULTS)
    mo_legacy = tuple(
        mo_dash_workspace_root(er)
        / "FX"
        / "non-linear FX forecast - daily"
        / "daily_binary_fx_forecast.xlsx"
        for er in _ETF_PROJECT_ROOT_DEFAULTS
    )
    legacy = tuple(p / "daily_binary_fx_forecast.xlsx" for p in _FX_PROJECT_ROOT_DEFAULTS)
    seen: set[str] = set()
    ordered: list[Path] = []
    for p in mo_new + mo_legacy + legacy:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            ordered.append(p)
    return tuple(ordered)


DASHBOARD_COLUMNS: tuple[str, ...] = (
    "primary",
    "lead_days",
    "recommendation",
    "position",
    "pred_class",
    "p_buy",
    "p_sell",
    "avg_oos_accuracy",
    "binary_oos_sharpe",
    "binary_oos_rows_for_sharpe",
    "global_batch_stability_rate",
    "latest_inner_test_acc",
    "same_as_previous_batch",
    "chosen_variant",
    "latest_panel_close",
    "latest_feature_date",
    "input_feature_date",
    "input_close",
    "target_forecast_date",
    "forecast_error",
)


def _resolve_xlsx_path() -> Path | None:
    env = os.environ.get("FX_FORECAST_XLSX")
    if env:
        p = Path(env)
        if p.exists():
            return p
    for candidate in _fx_forecast_xlsx_candidates():
        if candidate.exists():
            return candidate
    return None


def _resolve_fx_project_root() -> Path | None:
    env = os.environ.get("FX_FORECAST_PROJECT") or os.environ.get("FXNL_PROJECT_ROOT")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    for c in _FX_PROJECT_ROOT_DEFAULTS:
        if c.is_dir():
            return c
    return None


def _five_day_ohlc_all_pairs(etf_root: Path | None) -> pd.DataFrame:
    """Last 5 daily rows per FX pair from ``fx_data_collect`` raw parquets."""
    if etf_root is None or not fx_data_collect_package_dir(etf_root).is_dir():
        return pd.DataFrame()
    try:
        root_pkg = str(fx_data_collect_import_cwd(etf_root))
        if root_pkg not in sys.path:
            sys.path.insert(0, root_pkg)
        from fx_data_collect.config import FX_PAIRS  # noqa: PLC0415
        from fx_data_collect.repository import resolve_repo_root  # noqa: PLC0415
    except Exception:
        return pd.DataFrame()
    try:
        repo = resolve_repo_root()
    except Exception:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for inst in FX_PAIRS:
        pid = inst.replace("/", "_").replace(" ", "_")
        p = repo / "raw" / "fx" / f"{pid}.parquet"
        if not p.is_file():
            continue
        try:
            df = pd.read_parquet(p, columns=["Open", "High", "Low", "Close"])
        except Exception:
            df = pd.read_parquet(p)
        if df.empty or not all(c in df.columns for c in ("Open", "High", "Low", "Close")):
            continue
        tail = df.tail(5).copy().reset_index()
        idx0 = tail.columns[0]
        tail = tail.rename(columns={idx0: "date"})
        tail.insert(0, "pair", inst)
        parts.append(tail)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, axis=0, ignore_index=True)


def _resolve_etf_project_root() -> Path | None:
    for c in _ETF_PROJECT_ROOT_DEFAULTS:
        if c.is_dir() and (c / "project" / "main.py").is_file():
            return c
    return None


def _load_signals(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    signals = pd.read_excel(xlsx_path, sheet_name="daily_signals")
    try:
        meta = pd.read_excel(xlsx_path, sheet_name="metadata")
    except Exception:
        meta = None
    return signals, meta


def _format_for_display(df: pd.DataFrame, min_accuracy: float) -> pd.DataFrame:
    if df.empty:
        return df
    show = df.copy()
    if "avg_oos_accuracy" in show.columns:
        show["avg_oos_accuracy"] = pd.to_numeric(show["avg_oos_accuracy"], errors="coerce")
        show = show[show["avg_oos_accuracy"] >= float(min_accuracy)]
    keep = [c for c in DASHBOARD_COLUMNS if c in show.columns]
    show = show[keep]
    sort_cols: list[str] = []
    asc: list[bool] = []
    if "avg_oos_accuracy" in show.columns:
        sort_cols.append("avg_oos_accuracy")
        asc.append(False)
        if "binary_oos_sharpe" in show.columns:
            sort_cols.append("binary_oos_sharpe")
            asc.append(False)
    if sort_cols:
        show = show.sort_values(sort_cols, ascending=asc, na_position="last")
    return show.reset_index(drop=True)


def _pair_buy_sell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``primary`` with buy / sell counts (close-basis only)."""
    base_cols = ["primary", "n_buy", "n_sell"]
    if df.empty or "primary" not in df.columns:
        return pd.DataFrame(columns=base_cols)
    tmp = df.copy()
    tmp["_side"] = classify_signal_side(tmp)
    sub = tmp[tmp["_side"].isin(["buy", "sell"])]
    if sub.empty:
        return pd.DataFrame(columns=base_cols)
    ct = sub.groupby(["primary"], sort=False)["_side"].value_counts().unstack(fill_value=0)
    for c in ("buy", "sell"):
        if c not in ct.columns:
            ct[c] = 0
    out = ct.rename(columns={"buy": "n_buy", "sell": "n_sell"})[["n_buy", "n_sell"]]
    out = out.reset_index()
    out["n_buy"] = out["n_buy"].astype(int)
    out["n_sell"] = out["n_sell"].astype(int)
    return out.sort_values(["primary"], kind="stable").reset_index(drop=True)


def _fxcm_fc_secret_overrides() -> dict[str, object]:
    keys = (
        "FXCM_FC_USER",
        "FXCM_FC_PASSWORD",
        "FXCM_FC_URL",
        "FXCM_FC_CONNECTION",
        "FXCM_FC_SESSION",
        "FXCM_FC_PIN",
        "FXCM_FC_ACCOUNT",
        "FXCM_FC_ASSUMED_EQUITY",
        "FXCM_FC_ACCOUNT_CURRENCY",
        "FXCM_FC_MAX_LOTS",
        "FXCM_FC_LOTS_DIVISOR",
    )
    out: dict[str, object] = {}
    try:
        sec = getattr(st, "secrets", None)
        if sec is None:
            return out
        for k in keys:
            if k in sec:
                out[k] = sec[k]
    except Exception:
        pass
    return out


def _fxcm_fc_settings_for_ui():
    return fc_settings_from_env({**_fxcm_fc_secret_overrides()})


def _fxcm_fc_try_execute(
    *,
    primary: str,
    is_buy: bool,
    stop_loss_pct: float,
    lots: float | None = None,
    pct_equity: float | None = None,
    leverage: float | None = None,
) -> None:
    try:
        fc = _fxcm_fc_module_reload()
        settings = _fxcm_fc_settings_for_ui()
        msg = fc.execute_fc_market_order(
            primary=primary,
            is_buy=is_buy,
            settings=settings,
            lots=float(lots) if lots is not None else None,
            pct_equity=float(pct_equity) if pct_equity is not None else None,
            leverage=float(leverage) if leverage is not None else None,
            stop_loss_pct=float(stop_loss_pct),
        )
        st.success(msg)
    except Exception as e:
        st.error(str(e))


def _fxcm_fc_try_execute_currency_basket(
    *,
    legs: list[tuple[str, bool]],
    stop_loss_pct: float,
    lots_per_leg: float | None = None,
    pct_equity_each: float | None = None,
    leverage: float | None = None,
) -> None:
    if not legs:
        st.warning("No currencies with a clear dominant bias (ties skipped).")
        return
    try:
        fc = _fxcm_fc_module_reload()
        settings = _fxcm_fc_settings_for_ui()
        msg = fc.execute_fc_market_orders_sequence(
            legs=legs,
            settings=settings,
            lots_per_leg=float(lots_per_leg) if lots_per_leg is not None else None,
            pct_equity_each=float(pct_equity_each) if pct_equity_each is not None else None,
            leverage=float(leverage) if leverage is not None else None,
            stop_loss_pct=float(stop_loss_pct),
        )
        st.success(msg)
    except Exception as e:
        st.error(str(e))


def _df_to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, float_format=lambda x: f"{x:.4g}" if pd.notna(x) else "")


def _refresh_log_path(fx_root: Path) -> Path:
    return fx_root / "daily_binary_fx_forecast.refresh.log"


def _kick_off_refresh(*, fx_root: Path, etf_root: Path | None, refresh_data: bool) -> int | None:
    """Run the daily-report pipeline in a background subprocess. Returns its PID or None."""
    log_path = _refresh_log_path(fx_root)
    log_path.write_text("", encoding="utf-8")
    log_handle = log_path.open("a", encoding="utf-8")

    py = sys.executable or "py"

    def _ps_single_quote(s: str) -> str:
        """PowerShell single-quote with escaping for embedded quotes."""
        return "'" + s.replace("'", "''") + "'"

    def _cmd_to_powershell_invocation(cmd: list[str]) -> str:
        """Build a PowerShell command like: & 'python.exe' -m 'mod' 'arg' ..."""
        if not cmd:
            return ""
        exe = _ps_single_quote(cmd[0])
        args = " ".join(_ps_single_quote(x) for x in cmd[1:])
        return f"& {exe} {args}".strip()

    pre_cmds: list[list[str]] = []
    if refresh_data and etf_root is not None:
        pre_cmds.append([py, "-m", "fx_data_collect.run_collect", "--skip-fred", "--no-chart"])
        pre_cmds.append([py, "-m", "fx_data_collect.run_model_inputs"])

    daily_cmd = [
        py,
        "-m",
        "fxnl.daily_binary_forecast_report",
        "--align-to-next-bar",
        "--inject-live-quote",
    ]
    if etf_root is not None:
        mo_xlsx = mo_dash_fx_forecast_xlsx(etf_root)
        try:
            mo_xlsx.parent.mkdir(parents=True, exist_ok=True)
            daily_cmd.extend(["--out-xlsx", str(mo_xlsx)])
        except OSError:
            pass

    # IMPORTANT: `fx_data_collect` and `fxnl` live in different repo roots.
    # We `cd` before each step so `python -m <module>` resolves correctly.
    ps_fx_root = _ps_single_quote(str(fx_root))

    ps_fc_cwd = _ps_single_quote(str(fx_data_collect_import_cwd(etf_root))) if etf_root is not None else ""
    segments: list[str] = []
    if pre_cmds and etf_root is not None:
        segments.append(f"cd {ps_fc_cwd}")
        segments.extend(_cmd_to_powershell_invocation(cmd) for cmd in pre_cmds)

    segments.append(f"cd {ps_fx_root}")
    segments.append(_cmd_to_powershell_invocation(daily_cmd))

    powershell_line = " ; ".join([s for s in segments if s])

    started = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    log_handle.write(
        f"--- refresh started {started}\n"
        f"--- refresh_data={refresh_data} fx_root={fx_root} etf_root={etf_root}\n"
        f"--- {powershell_line}\n",
    )
    log_handle.flush()

    cwd = etf_root if (refresh_data and etf_root) else fx_root

    # Line-buffered logging from child Pythons (otherwise fxnl stdout can look "stuck" in .refresh.log).
    refresh_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", powershell_line],
            cwd=str(cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=refresh_env,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        return int(proc.pid)
    except Exception as e:
        log_handle.write(f"--- failed to spawn: {e}\n")
        log_handle.close()
        return None


def _refresh_progress(log_path: Path) -> tuple[int, int, str]:
    """Return ``(done, total, last_line)`` derived from the report's stdout log."""
    if not log_path.exists():
        return 0, 0, ""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return 0, 0, ""
    last_line = ""
    total = 0
    done = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        last_line = s
        if "Forecast " in s and "/" in s:
            try:
                tail = s.split("Forecast ", 1)[1]
                a, b = tail.split(maxsplit=1)
                num, den = a.split("/")
                done = max(done, int(num))
                total = max(total, int(den))
            except Exception:
                pass
    return done, total, last_line


def _refresh_pid_alive(pid: int) -> bool:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return str(int(pid)) in (proc.stdout or "")
    except Exception:
        return False


def _render_fx_input_data_preview(etf_root: Path | None) -> None:
    """Show tails of ``fx_data_collect`` inputs merged into the FX repo panel."""
    if etf_root is None:
        st.warning("ETF project root not found, or Mo_Dash path ``FX/FX forecasts/fx_data_collect`` is missing.")
        return
    prev = load_fx_input_preview(etf_root, raw_tail=60, panel_tail=40)
    for e in prev.get("errors", []):
        st.warning(e)
    rr = prev.get("repo_root") or ""
    st.caption(f"Repo: `{rr}`")
    if prev.get("fundamentals_daily") is not None:
        st.subheader("fundamentals_daily.parquet (tail)")
        st.dataframe(prev["fundamentals_daily"], use_container_width=True)
        st.caption(f"shape {prev.get('fundamentals_daily_shape')}")
    if prev.get("technicals_daily") is not None:
        st.subheader("technicals_daily.parquet (tail)")
        st.dataframe(prev["technicals_daily"], use_container_width=True)
        st.caption(f"shape {prev.get('technicals_daily_shape')}")
    if prev.get("manifest") is not None:
        st.subheader("manifest.json")
        st.json(prev["manifest"])
    raw = prev.get("raw_fx") or {}
    if raw:
        st.subheader("raw/fx — OHLCV tail per pair")
        keys = sorted(raw.keys())
        if keys:
            pair = st.selectbox("Pair", keys, key="fx_latest_inputs_pair_select")
            dfp = raw.get(pair)
            if dfp is None or getattr(dfp, "empty", True):
                st.info("No rows for this pair.")
            else:
                st.dataframe(dfp, use_container_width=True)


def _render_pair_likelihood_block(ccy_tbl: pd.DataFrame, signals: pd.DataFrame) -> None:
    """Rank pairs by ``base_score − quote_score`` and show whether the signal agrees."""
    rank = pair_likelihood_scores(ccy_tbl)
    if rank.empty:
        return
    sig = signals.copy()
    if "primary" in sig.columns:
        sig["_side"] = classify_signal_side(sig)
        sides_per_pair = (
            sig[sig["_side"].isin(["buy", "sell"])]
            .assign(side=lambda d: d["_side"].map({"buy": "long", "sell": "short"}))
            .groupby("primary")["side"]
            .agg(lambda s: s.value_counts().idxmax() if len(s) else "")
            .to_dict()
        )
    else:
        sides_per_pair = {}
    rank["signal_side"] = rank["primary"].map(sides_per_pair).fillna("")
    rank["match"] = [
        bool(sig_s) and (sig_s == exp)
        for sig_s, exp in zip(rank["signal_side"], rank["expected_side"])
    ]
    st.subheader("Pair likelihood ranking (currency-score derived)")
    st.caption(
        "For each pair **BASE/QUOTE**, ``pair_score = score(BASE) − score(QUOTE)`` from per-currency scores "
        "above. **Lowest** (most negative) → strongest pair-level **short** bias; **highest** → strongest **long** "
        "bias. ``signal_side`` is the majority direction across this pair's signal rows above (close-basis only). "
        "``match = True`` highlights pairs where the model agrees with the currency-bias ranking."
    )
    st.dataframe(rank, hide_index=True, use_container_width=True)


def _render_currency_block(sub: pd.DataFrame, *, title: str, basket_key_suffix: str) -> None:
    ccy_tbl = currency_leg_counts(sub)
    st.subheader(title)
    st.caption(
        "Each **buy** on **BASE/QUOTE** counts as **long BASE** and **short QUOTE**; each **sell** is the opposite. "
        "**Score** = (n_long − n_short) / (n_long + n_short). **Dominant** is the heavier side; ties are excluded from basket trading."
    )
    if ccy_tbl.empty:
        st.info("No currency legs could be derived (check **primary** format, e.g. `EUR/USD`).")
        return
    disp_ccy = ccy_tbl.copy()
    if "score" in disp_ccy.columns:
        disp_ccy["score"] = pd.to_numeric(disp_ccy["score"], errors="coerce").round(4)
    st.dataframe(disp_ccy, hide_index=True, use_container_width=True)
    _render_pair_likelihood_block(ccy_tbl, sub)
    planned = plan_equal_bias_orders(ccy_tbl)
    if planned:
        st.caption(
            "Planned basket (canonical pair per currency): "
            + " · ".join(f"{p.currency}→{fc_pair_display(p.primary)} {'BUY' if p.is_buy else 'SELL'}" for p in planned)
        )
    basket_col0, basket_col1, basket_col2, basket_col3, basket_col4, basket_col5 = st.columns(
        [0.9, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    basket_use_fixed = basket_col0.checkbox(
        "Fixed lots",
        value=False,
        key=f"fx_fc_basket_fixed_{basket_key_suffix}",
        help="If on, use lots per leg; if off, use % of equity × leverage (converted per pair).",
    )
    basket_lots = basket_col1.number_input(
        "Lots / leg",
        min_value=0.01,
        max_value=1_000_000.0,
        value=1.0,
        step=0.01,
        key=f"fx_fc_basket_lots_{basket_key_suffix}",
        disabled=not basket_use_fixed,
        help="Used when **Fixed lots** is checked.",
    )
    basket_pct = basket_col2.number_input(
        "% equity / leg",
        min_value=0.1,
        max_value=100.0,
        value=10.0,
        step=0.1,
        key=f"fx_fc_basket_pct_{basket_key_suffix}",
        disabled=basket_use_fixed,
    )
    basket_lev = basket_col3.number_input(
        "Leverage",
        min_value=0.1,
        max_value=500.0,
        value=10.0,
        step=0.5,
        key=f"fx_fc_basket_lev_{basket_key_suffix}",
        disabled=basket_use_fixed,
    )
    basket_stop_pct = basket_col4.number_input(
        "Stop %",
        min_value=0.01,
        max_value=50.0,
        value=1.0,
        step=0.05,
        key=f"fx_fc_basket_stop_{basket_key_suffix}",
        help="RATE_STOP when accepted.",
    )
    if basket_col5.button(
        "Trade basket",
        key=f"fx_fc_trade_currency_basket_{basket_key_suffix}",
        help="One ForexConnect session; one market order per non-tie currency.",
    ):
        _fxcm_fc_try_execute_currency_basket(
            legs=[(p.primary, p.is_buy) for p in planned],
            stop_loss_pct=basket_stop_pct,
            lots_per_leg=float(basket_lots) if basket_use_fixed else None,
            pct_equity_each=None if basket_use_fixed else float(basket_pct),
            leverage=None if basket_use_fixed else float(basket_lev),
        )


def _daily_trade_check_path(fx_root: Path | None) -> Path | None:
    """``daily_trade_check.json`` lives next to the workbook in the FX NL project."""
    if fx_root is None or not fx_root.is_dir():
        return None
    return fx_root / "daily_trade_check.json"


def _load_daily_trade_check(fx_root: Path | None) -> dict | None:
    p = _daily_trade_check_path(fx_root)
    if p is None or not p.is_file():
        return None
    try:
        import json  # noqa: PLC0415
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _render_auto_trade_status(report: dict, fx_root: Path | None) -> None:
    """Render the most recent ``daily_trade_check.json`` payload."""
    finished = report.get("finished_at") or report.get("started_at") or "—"
    fx_date = report.get("target_forecast_date") or "—"
    dry = bool(report.get("dry_run", True))
    cfg = report.get("config", {}) or {}
    badge = "🟡 dry-run (auto-trade disabled)" if dry else "🟢 auto-trade enabled"
    st.markdown(
        f"**Last check:** {finished}  ·  **Forecast date:** {fx_date}  ·  {badge}  ·  "
        f"top **{cfg.get('top_n', '?')}** · max open **{cfg.get('max_open', '?')}**"
    )

    errors = report.get("errors") or []
    if errors:
        for e in errors:
            st.error(f"`{e.get('step', '?')}`: {e.get('error', '')}")

    selected = report.get("selected_book") or []
    if selected:
        st.markdown("**Selected book** (top-N picked greedily by `|pair_score|`, no shared currencies):")
        sel_df = pd.DataFrame(selected)
        keep = [c for c in ("selection_rank", "primary", "side", "expected_side", "pair_score", "abs_score", "ranking_rank", "reason") if c in sel_df.columns]
        st.dataframe(sel_df[keep], hide_index=True, use_container_width=True)
    else:
        st.caption("Selected book is empty (no qualifying signals today).")

    skips = report.get("overlap_skips") or []
    if skips:
        with st.popover(f"⚠ Passed over {len(skips)} candidate(s) due to currency overlap"):
            st.dataframe(pd.DataFrame(skips), hide_index=True, use_container_width=True)

    decisions = report.get("decisions") or []
    if decisions:
        st.markdown("**Open positions on FXCM** vs today's selected book:")
        df = pd.DataFrame(decisions)
        for col in ("primary", "side", "lots", "decision", "selection_rank", "today_side", "pair_score", "reason", "gross_pl"):
            if col not in df.columns:
                df[col] = pd.NA
        df = df[["primary", "side", "lots", "decision", "selection_rank", "today_side", "pair_score", "reason", "gross_pl"]]
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.caption("No open positions on FXCM at the time of the last check.")

    planned = report.get("planned_opens") or []
    if planned:
        st.markdown("**Planned new opens (selected book slots not yet on):**")
        st.dataframe(pd.DataFrame(planned), hide_index=True, use_container_width=True)

    actions = report.get("actions") or []
    if actions:
        st.markdown("**Actions executed in the last pass:**")
        st.dataframe(pd.DataFrame(actions), hide_index=True, use_container_width=True)


def _render_auto_trade_history(fx_root: Path | None, max_rows: int = 20) -> None:
    """Show the last few passes from ``daily_trade_check.history.jsonl``."""
    if fx_root is None or not fx_root.is_dir():
        return
    p = fx_root / "daily_trade_check.history.jsonl"
    if not p.is_file():
        st.caption("No history yet (this is the first run).")
        return
    try:
        import json  # noqa: PLC0415
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        st.warning(f"Could not read history file: {e}")
        return

    rows: list[dict] = []
    for line in lines[-int(max_rows):]:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sel = r.get("selected_book") or []
        sel_str = ", ".join(f"#{s.get('selection_rank', '?')} {s.get('primary', '')} {s.get('side', '')}" for s in sel)
        actions = r.get("actions") or []
        kinds = {a.get("kind", "?") for a in actions}
        rows.append(
            {
                "finished_at": r.get("finished_at"),
                "forecast_date": r.get("target_forecast_date"),
                "enabled": (r.get("config") or {}).get("enabled"),
                "selected_book": sel_str,
                "n_open_positions": len(r.get("open_positions") or []),
                "actions_summary": ", ".join(sorted(kinds)) if kinds else "(none)",
                "errors": len(r.get("errors") or []),
            }
        )
    if not rows:
        st.caption("No usable history records yet.")
        return
    st.dataframe(pd.DataFrame(rows[::-1]), hide_index=True, use_container_width=True)
    st.caption(
        f"History file: `{p}` "
        f"(append-only JSONL — also see `daily_trade_check.history.log` for a human-readable summary of every pass)."
    )


def _render_auto_trade_config_form() -> None:
    """Form to edit ``auto_trade.config.json``. Streamlit re-runs on save."""
    cfg = load_config()
    with st.form("auto_trade_config_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
        enabled = c1.toggle(
            "Enable automated trading at 5:15 PM NY",
            value=bool(cfg.enabled),
            help=(
                "When ON, the daily check (part of the 5:15 PM scheduled task) selects the "
                "top-N pairs (greedy, currency-disjoint), KEEPS positions already in that "
                "selected book (no churn), CLOSES positions that aren't, and OPENS the missing "
                "slots. When OFF, the dashboard still shows the analysis but nothing trades."
            ),
        )
        top_n = c2.number_input("Top-N pairs", min_value=1, max_value=10, value=int(cfg.top_n), step=1, help="Size of the selected book each day (default 2).")
        max_open = c3.number_input("Max open", min_value=1, max_value=10, value=int(cfg.max_open), step=1, help="Hard cap on total simultaneously open auto-trades.")

        c4, c5, c6 = st.columns(3)
        pct = c4.number_input("% equity / trade", min_value=0.1, max_value=100.0, value=float(cfg.pct_equity_per_trade), step=0.1)
        lev = c5.number_input("Leverage", min_value=0.1, max_value=500.0, value=float(cfg.leverage), step=0.5)
        stop = c6.number_input("Stop %", min_value=0.01, max_value=50.0, value=float(cfg.stop_pct), step=0.05)

        skip_v = st.checkbox(
            "Skip pairs that are not price-subscribed on FXCM (status ≠ 'T')",
            value=bool(cfg.skip_unsubscribed),
            help="If a top pair like AUD/USD shows subscription_status='V' on your account, skip it and surface a note (open it once in Trading Station to enable trading).",
        )

        submitted = st.form_submit_button("💾 Save auto-trade settings", type="primary")
        if submitted:
            new_cfg = AutoTradeConfig(
                enabled=bool(enabled),
                top_n=int(top_n),
                pct_equity_per_trade=float(pct),
                leverage=float(lev),
                stop_pct=float(stop),
                max_open=int(max_open),
                skip_unsubscribed=bool(skip_v),
            )
            saved = save_config(new_cfg)
            st.success(f"Saved → `{saved}`. The next 5:15 PM run will use these settings.")


def _render_auto_trade_panel(*, etf_root: Path | None, fx_root: Path | None) -> None:
    """Show today's auto-trade pass + the configuration form, both in an expander."""
    cfg = load_config()
    enabled = bool(cfg.enabled)
    title = "🤖 Auto-trade check (daily)" + ("  —  **enabled**" if enabled else "  —  disabled (analysis only)")
    with st.expander(title, expanded=False):
        st.caption(
            f"Config: `{default_config_path()}` · "
            "The 5:15 PM scheduled task runs the analysis and writes the report. Trades only "
            "execute when **Enable automated trading** is on. Click **Run check now** to refresh "
            "this section between scheduled runs."
        )

        report = _load_daily_trade_check(fx_root)
        if report is None:
            st.info(
                "No daily check has run yet (or the workbook hasn't been refreshed). "
                "Use the **↻ Refresh forecasts now** button above to build the workbook and run "
                "the first auto-trade check pass."
            )
        else:
            _render_auto_trade_status(report, fx_root)

        st.divider()
        c1, c2 = st.columns([1.4, 4])
        run_now = c1.button(
            "▶ Run check now",
            help="Runs scripts/daily_trade_check.py once with the current config. "
                 "Reads the existing workbook (does NOT refit models) and updates this report.",
        )
        if run_now:
            if etf_root is None:
                st.error("Cannot find the ETF Forecaster project root — set FXNL_PROJECT_ROOT or relaunch via the Mo_Dash launcher.")
            else:
                ok, output = _run_daily_check_now(etf_root)
                if ok:
                    st.success("Daily check completed; report refreshed.")
                else:
                    st.error("Daily check failed — see output below.")
                with c2:
                    st.code(output[-3000:] if output else "(no output)", language="text")
                st.rerun()

        st.markdown("##### Settings")
        _render_auto_trade_config_form()

        st.markdown("##### History (last 20 passes)")
        _render_auto_trade_history(fx_root, max_rows=20)


def _run_daily_check_now(etf_root: Path) -> tuple[bool, str]:
    """Foreground subprocess invocation of ``scripts/daily_trade_check.py``."""
    venv_py = etf_root / "project" / ".venv" / "Scripts" / "python.exe"
    py = str(venv_py) if venv_py.is_file() else (sys.executable or "py")
    script = etf_root / "project" / "scripts" / "daily_trade_check.py"
    if not script.is_file():
        return False, f"Missing script: {script}"
    try:
        proc = subprocess.run(
            [py, str(script)],
            cwd=str(etf_root / "project"),
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode == 0, out
    except Exception as e:  # noqa: BLE001
        return False, f"Failed to launch daily check: {e}"


def render_fx_forecast_tab(default_email: str) -> None:
    """Render the FX direction forecasts page."""
    st.subheader("FX direction forecasts (daily)")
    st.caption(
        "Workbook from **daily_binary_fx_forecast.xlsx** (built with **--align-to-next-bar** so each row targets the "
        "**next** session after the feature panel — best after the US close when daily data is complete). "
        "Optional live spot: **--inject-live-quote**. Refresh pulls repo data and refits endorsed models."
    )

    fx_root = _resolve_fx_project_root()
    etf_root = _resolve_etf_project_root()

    xlsx_path = _resolve_xlsx_path()
    refresh_state = st.session_state.setdefault("fx_refresh", {"pid": None, "log": None, "started": None, "phase": None})

    rb1, rb2, rb3 = st.columns([1.6, 1.6, 2.2])
    refresh_clicked = rb1.button(
        "↻ Refresh forecasts now",
        type="primary",
        help="Pulls today's live FX spot, rebuilds the panel, and refits each endorsed model. Several minutes.",
        disabled=refresh_state.get("pid") is not None,
    )
    refresh_data_too = rb2.checkbox(
        "Also refresh data first",
        value=True,
        help="Re-pull FX raw parquets from Yahoo and rebuild model_inputs before forecasting (recommended).",
        disabled=refresh_state.get("pid") is not None,
    )
    with rb3:
        if st.button(
            "See latest data",
            help="Parquets from fx_data_collect merged into the FX repo (fundamentals, technicals, raw FX, manifest).",
            disabled=refresh_state.get("pid") is not None,
        ):
            st.session_state["fx_show_latest_inputs"] = True
            st.rerun()

    if refresh_clicked:
        if fx_root is None:
            st.error(
                "Cannot find the FX non-linear project under Mo_Dash "
                "(expected …/Mo_Dash/FX/FX forecasts/non-linear FX forecast - daily_binary_fx_forecast). "
                "Set `FXNL_PROJECT_ROOT` env var if it lives elsewhere."
            )
        else:
            pid = _kick_off_refresh(
                fx_root=fx_root,
                etf_root=etf_root,
                refresh_data=bool(refresh_data_too and etf_root is not None),
            )
            if pid is None:
                st.error("Failed to start the refresh subprocess; see refresh.log next to the workbook.")
            else:
                refresh_state["pid"] = pid
                refresh_state["log"] = str(_refresh_log_path(fx_root))
                refresh_state["started"] = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
                refresh_state["phase"] = "data" if refresh_data_too else "forecast"
                st.session_state.fx_refresh = refresh_state
                st.rerun()

    pid = refresh_state.get("pid")
    if pid:
        log_path = Path(refresh_state.get("log") or "")
        alive = _refresh_pid_alive(int(pid)) if int(pid) else False
        done, total, last_line = _refresh_progress(log_path)

        if alive:
            st.info(
                f"Refresh running (pid {pid}, started {refresh_state.get('started')}). "
                f"{done}/{total or '?'} forecasts complete. Last line: `{last_line[-180:] if last_line else '...'}`"
            )
            if total > 0:
                st.progress(min(1.0, max(0.0, done / max(total, 1))))
            if st.button("Refresh page (poll progress)"):
                st.rerun()
        else:
            refresh_state["pid"] = None
            refresh_state["phase"] = None
            st.session_state.fx_refresh = refresh_state
            st.success(f"Refresh complete ({done}/{total} forecasts). Reloading workbook…")
            st.rerun()

    if st.session_state.get("fx_show_latest_inputs"):
        st.divider()
        st.markdown("#### Latest forecast input data (`fx_data_collect`)")
        _render_fx_input_data_preview(etf_root)
        if st.button("Hide latest data", key="fx_hide_latest_inputs"):
            st.session_state["fx_show_latest_inputs"] = False
            st.rerun()
        st.divider()

    if xlsx_path is None:
        fxnl_roots = _FX_PROJECT_ROOT_DEFAULTS
        manual_cwd = str(fxnl_roots[0]) if fxnl_roots else r"<Mo_Dash>\FX\FX forecasts\non-linear FX forecast - daily_binary_fx_forecast"
        st.warning(
            "No FX forecast workbook found yet. Click **↻ Refresh forecasts now** to build one, "
            "or run the daily report manually:\n\n"
            "```powershell\n"
            f"cd \"{manual_cwd}\"\n"
            "py -3 -m fxnl.daily_binary_forecast_report --align-to-next-bar --inject-live-quote\n"
            "```"
        )
        return

    try:
        signals, meta = _load_signals(xlsx_path)
    except Exception as e:
        st.error(f"Failed to read {xlsx_path}: {e}")
        return

    file_mtime = pd.Timestamp(xlsx_path.stat().st_mtime, unit="s").tz_localize("UTC").tz_convert("America/New_York")
    st.caption(f"Source: `{xlsx_path}`  ·  generated {file_mtime:%Y-%m-%d %H:%M %Z}")

    _render_auto_trade_panel(etf_root=etf_root, fx_root=fx_root)

    ohlc5 = _five_day_ohlc_all_pairs(etf_root)
    if not ohlc5.empty:
        st.subheader("Latest 5 sessions — OHLC (all pairs, raw parquets)")
        disp_ohlc = ohlc5.copy()
        if "date" in disp_ohlc.columns:
            disp_ohlc["date"] = pd.to_datetime(disp_ohlc["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(disp_ohlc, hide_index=True, use_container_width=True)

    if meta is not None and not meta.empty:
        meta_row = meta.iloc[0].to_dict()
        aligned = bool(meta_row.get("align_to_next_bar", False))
        if not aligned:
            st.warning(
                "Source workbook was generated **without** `--align-to-next-bar`. Lead-N forecasts "
                "target panel_end + N bars (not 'tomorrow'). Click **↻ Refresh forecasts now**."
            )

    c1, c2 = st.columns([1, 1])
    min_accuracy = c1.slider("Min avg OOS accuracy", 0.50, 0.95, 0.60, 0.01)
    only_strong = c2.checkbox("Only Strong (≥65%)", value=False)

    show = _format_for_display(signals, min_accuracy=min_accuracy)
    if only_strong and "avg_oos_accuracy" in show.columns:
        show = show[show["avg_oos_accuracy"] >= 0.65]
    if "target_return_basis" in show.columns:
        show = show[show["target_return_basis"].astype(str).str.lower() == "close"]

    if show.empty:
        st.info("No signals match the current filters.")
        return

    n_rows = len(show)
    n_strong = int((show["avg_oos_accuracy"] >= 0.65).sum()) if "avg_oos_accuracy" in show.columns else 0
    pairs = sorted(show["primary"].dropna().unique().tolist()) if "primary" in show.columns else []
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Signals", f"{n_rows}")
    m2.metric("Strong", f"{n_strong}")
    m3.metric("Pairs", f"{len(pairs)}")
    if "target_forecast_date" in show.columns:
        tts = pd.to_datetime(show["target_forecast_date"], errors="coerce")
        valid = tts.dropna()
        if not valid.empty:
            mn, mx = valid.min(), valid.max()
            if pd.Timestamp(mn).normalize() == pd.Timestamp(mx).normalize():
                m4.metric("Forecast date", f"{pd.Timestamp(mx):%Y-%m-%d}")
            else:
                m4.metric(
                    "Forecast date",
                    f"{pd.Timestamp(mn):%Y-%m-%d}–{pd.Timestamp(mx):%Y-%m-%d}",
                )

    if meta is not None and not meta.empty and bool(meta.iloc[0].get("align_to_next_bar", False)):
        meta0 = meta.iloc[0]
        thru = meta0.get("fx_data_through_date") or meta0.get("data_through_date")
        thru_s = f" Data through **{thru}** (NY 5 PM rule)." if thru and str(thru) not in ("", "nan") else ""
        st.caption(
            "`target_forecast_date` is the **next** session after the last feature bar allowed by the "
            "**5:00 PM America/New_York** cutoff (before 5 PM → yesterday's close → forecast **today**; "
            "after 5 PM → today's close → forecast **next business day**)."
            f"{thru_s} `latest_feature_date` is the last bar in the panel. "
            "`input_feature_date` **lags** the panel end when `lead_days` > 1."
        )

    st.dataframe(show, hide_index=True, use_container_width=True)

    _render_currency_block(show, title="Currency exposure (from pair signals)", basket_key_suffix="single")

    summary = _pair_buy_sell_summary(show)
    st.subheader("Summary by pair")
    st.caption(
        "Counts of **buy** vs **sell** for the filtered rows above. "
        "Side is taken from **position**, then **recommendation**, then numeric **pred_class** (1→buy, 0→sell), "
        "then whichever of **p_buy** / **p_sell** is larger."
    )
    if summary.empty:
        st.info("No buy/sell classification possible for the current rows — check position/recommendation/pred_class values.")
    else:
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.markdown("#### Trade from summary (FXCM ForexConnect)")
        st.caption(
            "Log in with **FXCM demo** credentials from `.env` / secrets. **Default sizing:** equity from the API × "
            "**% of equity** × **leverage** = target notional in **account currency**; that amount is converted to the "
            "pair’s **base currency** using subscribed **OFFERS** mids, then sent as **AMOUNT** (base units). "
            "Check **Fixed lots** to send a traditional lot count instead. Stops use **RATE_STOP** when the broker allows."
        )
        fc_use_fixed = st.checkbox(
            "Fixed lots (override equity % sizing)",
            value=False,
            key="fxcm_fc_use_fixed_lots",
        )
        r1, r2, r3, r4 = st.columns([1.0, 1.0, 1.0, 1.2])
        fc_lots = r1.number_input(
            "Lots (if fixed)",
            min_value=0.01,
            max_value=1_000_000.0,
            value=1.0,
            step=0.01,
            key="fxcm_fc_lots",
            disabled=not fc_use_fixed,
        )
        fc_pct = r2.number_input(
            "% of equity",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1,
            key="fxcm_fc_pct_equity",
            disabled=fc_use_fixed,
        )
        fc_leverage = r3.number_input(
            "Leverage multiplier",
            min_value=0.1,
            max_value=500.0,
            value=10.0,
            step=0.5,
            key="fxcm_fc_leverage",
            disabled=fc_use_fixed,
        )
        fc_stop_pct = r4.number_input(
            "Stop loss % (adverse)",
            min_value=0.01,
            max_value=50.0,
            value=1.0,
            step=0.05,
            key="fxcm_fc_stop_loss_pct",
            help="Percent move against you from the quoted side used for RATE_STOP.",
        )

        with st.expander("Where to put FXCM demo credentials (exactly)", expanded=False):
            st.markdown(
                "Put secrets in **one** of these places (never commit; never paste into chat):\n\n"
                "---\n\n"
                "**Option A — `project/.env`** (same folder as `main.py`, i.e. `...\\ETF Forecaster\\project\\.env`):\n\n"
                "```\n"
                "FXCM_FC_USER=your_demo_login\n"
                "FXCM_FC_PASSWORD=your_demo_password\n"
                "FXCM_FC_URL=https://www.fxcorporate.com/Hosts.jsp\n"
                "FXCM_FC_CONNECTION=Demo\n"
                "# Optional — only if FXCM gave you session / PIN:\n"
                "# FXCM_FC_SESSION=\n"
                "# FXCM_FC_PIN=\n"
                "# Optional — if you have several accounts:\n"
                "# FXCM_FC_ACCOUNT=YOUR_ACCOUNT_ID\n"
                "# Optional — if equity cannot be read from API for sizing:\n"
                "# FXCM_FC_ASSUMED_EQUITY=100000\n"
                "# Optional — three-letter account currency if the API omits it (cross conversion for order size):\n"
                "# FXCM_FC_ACCOUNT_CURRENCY=USD\n"
                "# Optional lot caps:\n"
                "# FXCM_FC_MAX_LOTS=100\n"
                "```\n\n"
                "---\n\n"
                "**Option B — `project/.streamlit/secrets.toml`** (same keys as above, TOML syntax):\n\n"
                "```toml\n"
                "FXCM_FC_USER = \"your_demo_login\"\n"
                "FXCM_FC_PASSWORD = \"your_demo_password\"\n"
                "FXCM_FC_URL = \"https://www.fxcorporate.com/Hosts.jsp\"\n"
                "FXCM_FC_CONNECTION = \"Demo\"\n"
                "```\n\n"
                "**Python:** install **`forexconnect`** (`pip install -r requirements.txt`). "
                "If install fails on **Python 3.12+**, use a **3.10 or 3.11** virtualenv for this project — "
                "FXCM’s wheels often target older CPython. Sign FXCM’s **EULA** and use credentials from a **TSII**-style demo as per "
                "[ForexConnectAPI](https://github.com/fxcm/ForexConnectAPI)."
            )
            if st.button("Test FXCM ForexConnect login", key="fxcm_fc_test_connection"):
                try:
                    fc = _fxcm_fc_module_reload()
                    settings = _fxcm_fc_settings_for_ui()
                    msg = fc.test_fc_login(settings)
                    st.success(msg)
                except Exception as e:
                    st.error(str(e))

        for idx, srow in summary.iterrows():
            primary = str(srow["primary"])
            r0, r2, r3, r4, r5 = st.columns([2.6, 0.55, 0.55, 0.65, 0.65])
            r0.markdown(f"**{primary}**")
            r2.write(f"{int(srow['n_buy'])}B")
            r3.write(f"{int(srow['n_sell'])}S")
            sym_lbl = fc_pair_display(primary)
            if r4.button("Buy", key=f"fxcm_fc_buy_{idx}", type="primary", help=f"Market buy {sym_lbl}"):
                if fc_use_fixed:
                    _fxcm_fc_try_execute(
                        primary=primary,
                        is_buy=True,
                        stop_loss_pct=fc_stop_pct,
                        lots=float(fc_lots),
                    )
                else:
                    _fxcm_fc_try_execute(
                        primary=primary,
                        is_buy=True,
                        stop_loss_pct=fc_stop_pct,
                        pct_equity=float(fc_pct),
                        leverage=float(fc_leverage),
                    )
            if r5.button("Sell", key=f"fxcm_fc_sell_{idx}", help=f"Market sell {sym_lbl}"):
                if fc_use_fixed:
                    _fxcm_fc_try_execute(
                        primary=primary,
                        is_buy=False,
                        stop_loss_pct=fc_stop_pct,
                        lots=float(fc_lots),
                    )
                else:
                    _fxcm_fc_try_execute(
                        primary=primary,
                        is_buy=False,
                        stop_loss_pct=fc_stop_pct,
                        pct_equity=float(fc_pct),
                        leverage=float(fc_leverage),
                    )

    st.divider()
    email_to = os.environ.get("EMAIL_TO", default_email)
    st.caption(f"Email recipient: **{email_to}** (override with `EMAIL_TO` in `.env`).")
    if st.button("Email these results to me"):
        try:
            html = (
                f"<h3>FX direction forecasts ({n_rows} signals, {n_strong} strong)</h3>"
                f"<p>Source: {xlsx_path}</p>"
                + _df_to_html_table(show)
            )
            text = show.to_string(index=False)
            send_html_email(
                to_addr=email_to,
                subject="FX Forecaster — next-day direction forecasts",
                html_body=html,
                text_body=text,
            )
            st.success(f"Sent {n_rows} signals to {email_to}.")
        except Exception as e:
            st.error(str(e))
