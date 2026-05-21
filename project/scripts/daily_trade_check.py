"""Daily auto-trade check — runs after the 5:15 PM refresh.

Each pass:

1. Reads the freshly-written FX workbook and ranks pairs by ``|pair_score|``
   (same logic as the dashboard's pair-likelihood block).
2. **Selects today's target book** with a greedy currency-disjoint rule —
   #1 = highest absolute conviction (long or short), then for #2..N walk down
   the ranking and pick the next pair whose BASE and QUOTE are not shared with
   any already-selected pair. Each skip is recorded with its reason.
3. Diffs that book against current FXCM open positions:

   * **KEEP**  — position is already in today's selected book (same pair, same
     direction). Never closed / re-opened (saves the round-trip commission).
   * **CLOSE** — position is NOT in today's selected book.
   * **OPEN**  — slot in the selected book that isn't currently held (only fires
     when ``auto_trade.config.enabled`` is ``True``).

4. Persists results in three artifacts (next to the workbook):

   * ``daily_trade_check.json`` — latest pass payload (read by the dashboard).
   * ``daily_trade_check.history.jsonl`` — append-only structured history.
   * ``daily_trade_check.history.log`` — append-only human-readable log of every
     pass: timestamp, target forecast date, ranking, selected book + reasoning,
     overlap skips, decisions, actions.

The script is invoked as ``python scripts/daily_trade_check.py`` from
``Mo_Dash/ETF/ETF Forecaster/project/`` with the Mo_Dash venv. The 5:15 PM task
runs it as step 4 of ``run_fx_daily_refresh.ps1``.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------------------
# Bootstrap: make ``project/`` importable so ``utils.*`` and ``ui.*`` resolve when
# this file is run directly (Task Scheduler invokes it as a plain script).
# --------------------------------------------------------------------------------------

_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

try:
    from dotenv import load_dotenv  # noqa: PLC0415

    for _env in (_PROJECT_DIR / ".env", _PROJECT_DIR.parent / ".env"):
        if _env.is_file():
            load_dotenv(_env, override=False)
    load_dotenv(override=False)
except Exception:  # pragma: no cover — dotenv is best-effort
    pass

from utils.auto_trade_config import AutoTradeConfig, load_config, save_config
from utils.fx_currency_exposure import (
    currency_leg_counts,
    pair_likelihood_scores,
    parse_fx_pair,
)
from utils.fx_signal_side import classify_signal_side
from utils.mo_dash_layout import fx_nl_project_root, mo_dash_workspace_root
from utils.fxcm_forexconnect_trade import (
    close_fc_trade,
    execute_fc_market_order,
    fc_settings_from_env,
    list_open_fc_trades,
)


logger = logging.getLogger("daily_trade_check")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# --------------------------------------------------------------------------------------
# Workbook helpers
# --------------------------------------------------------------------------------------

DEFAULT_MIN_ACCURACY = 0.55  # mirrors the dashboard's "Min OOS accuracy" slider default


def _workbook_path() -> Path:
    return fx_nl_project_root(_PROJECT_DIR.parent) / "daily_binary_fx_forecast.xlsx"


def _status_path() -> Path:
    return fx_nl_project_root(_PROJECT_DIR.parent) / "daily_trade_check.json"


def _history_jsonl_path() -> Path:
    return fx_nl_project_root(_PROJECT_DIR.parent) / "daily_trade_check.history.jsonl"


def _history_log_path() -> Path:
    return fx_nl_project_root(_PROJECT_DIR.parent) / "daily_trade_check.history.log"


def _load_signals(xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not xlsx.is_file():
        raise FileNotFoundError(f"Workbook missing: {xlsx}")
    signals = pd.read_excel(xlsx, sheet_name="daily_signals")
    try:
        meta = pd.read_excel(xlsx, sheet_name="metadata")
    except Exception:
        meta = pd.DataFrame()
    return signals, meta


def _forecast_date_from_meta(meta: pd.DataFrame, signals: pd.DataFrame) -> str | None:
    """Best-effort: pick the most common ``target_forecast_date`` in the signal sheet."""
    if "target_forecast_date" in signals.columns and not signals.empty:
        ts = pd.to_datetime(signals["target_forecast_date"], errors="coerce")
        ts = ts.dropna()
        if not ts.empty:
            return ts.mode().iloc[0].strftime("%Y-%m-%d")
    if not meta.empty and "key" in meta.columns and "value" in meta.columns:
        row = meta[meta["key"].astype(str).str.lower() == "target_forecast_date"]
        if not row.empty:
            return str(row["value"].iloc[0])
    return None


# --------------------------------------------------------------------------------------
# Ranking — same path as the dashboard's pair-likelihood block
# --------------------------------------------------------------------------------------


def _rank_pairs_today(signals: pd.DataFrame, min_accuracy: float) -> pd.DataFrame:
    """Return pairs sorted by ``|pair_score|`` descending with ``rank`` + ``expected_side``."""
    if signals.empty:
        return pd.DataFrame(
            columns=["primary", "expected_side", "pair_score", "abs_score", "rank"],
        )
    df = signals.copy()
    if "avg_oos_accuracy" in df.columns:
        df["avg_oos_accuracy"] = pd.to_numeric(df["avg_oos_accuracy"], errors="coerce")
        df = df[df["avg_oos_accuracy"] >= float(min_accuracy)]
    df["_side"] = classify_signal_side(df)
    df = df[df["_side"].isin(["buy", "sell"])]
    if df.empty:
        return pd.DataFrame(
            columns=["primary", "expected_side", "pair_score", "abs_score", "rank"],
        )
    ccy = currency_leg_counts(df)
    pairs = pair_likelihood_scores(ccy)
    if pairs.empty:
        return pd.DataFrame(
            columns=["primary", "expected_side", "pair_score", "abs_score", "rank"],
        )
    pairs = pairs[pairs["expected_side"].isin(["long", "short"])].copy()
    pairs["abs_score"] = pairs["pair_score"].abs()
    pairs = pairs.sort_values("abs_score", ascending=False, kind="stable").reset_index(drop=True)
    pairs["rank"] = pairs.index + 1
    return pairs[["primary", "expected_side", "pair_score", "abs_score", "rank"]]


# --------------------------------------------------------------------------------------
# Per-position classification
# --------------------------------------------------------------------------------------


def _normalize_pair(primary: str) -> str:
    """Always store pairs as ``EUR/USD`` regardless of input formatting."""
    pq = parse_fx_pair(primary)
    if pq is None:
        return primary.strip().upper()
    return f"{pq[0]}/{pq[1]}"


def _side_from_expected(expected_side: str) -> str:
    return "buy" if str(expected_side).lower() == "long" else "sell"


def _select_top_pairs_no_overlap(
    ranking: pd.DataFrame,
    *,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy currency-disjoint selection by descending ``|pair_score|``.

    Returns ``(selected, overlap_skips)``:

    * ``selected``       — list of dicts (selection_rank, primary, side, expected_side,
      pair_score, abs_score, ranking_rank) in order of selection.
    * ``overlap_skips``  — list of dicts for each candidate that was passed over
      because it shared a currency with an already-selected pair (includes the
      currency that caused the skip — useful for the log).
    """
    selected: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    if ranking.empty or top_n <= 0:
        return selected, skips

    used: set[str] = set()
    for _, row in ranking.iterrows():
        primary = _normalize_pair(str(row["primary"]))
        pq = parse_fx_pair(primary)
        if pq is None:
            continue
        base, quote = pq
        expected_side = str(row["expected_side"])
        if expected_side not in ("long", "short"):
            continue

        overlap = used & {base, quote}
        if overlap:
            skips.append(
                {
                    "primary": primary,
                    "ranking_rank": int(row["rank"]),
                    "expected_side": expected_side,
                    "pair_score": float(row["pair_score"]),
                    "abs_score": float(row["abs_score"]),
                    "skipped_because": f"shares {sorted(overlap)[0]} with already-selected pair(s)",
                }
            )
            continue

        selected.append(
            {
                "selection_rank": len(selected) + 1,
                "primary": primary,
                "side": _side_from_expected(expected_side),
                "expected_side": expected_side,
                "pair_score": float(row["pair_score"]),
                "abs_score": float(row["abs_score"]),
                "ranking_rank": int(row["rank"]),
                "reason": (
                    f"highest |pair_score| ({float(row['abs_score']):.4f})"
                    if not selected
                    else f"highest remaining |pair_score| disjoint from selected currencies (used={sorted(used)})"
                ),
            }
        )
        used.update({base, quote})
        if len(selected) >= int(top_n):
            break

    return selected, skips


def _build_book_diff(
    *,
    open_positions: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Diff today's selected book against current open positions.

    Returns ``(decisions, planned_opens)``:

    * ``decisions``     — one per open position (decision = ``keep`` or ``close``,
      with the reason). KEEP whenever ``(primary, side)`` matches a row in
      ``selected``; CLOSE otherwise (never close-and-reopen).
    * ``planned_opens`` — one per ``selected`` row that isn't held yet.
    """
    selected_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (_normalize_pair(s["primary"]), s["side"]): s for s in selected
    }

    decisions: list[dict[str, Any]] = []
    held_keys: set[tuple[str, str]] = set()
    for pos in open_positions:
        key = (_normalize_pair(str(pos["primary"])), str(pos["side"]))
        sel = selected_by_key.get(key)
        if sel is not None:
            held_keys.add(key)
            decisions.append(
                {
                    "primary": key[0],
                    "side": key[1],
                    "decision": "keep",
                    "selection_rank": int(sel["selection_rank"]),
                    "today_side": sel["expected_side"],
                    "pair_score": sel["pair_score"],
                    "reason": (
                        f"already on and matches selection #{sel['selection_rank']} "
                        f"({sel['primary']} {sel['expected_side']}, |score| {sel['abs_score']:.4f}) - kept (no churn)"
                    ),
                }
            )
        else:
            # Look up today's full-ranking row (if any) to surface useful context
            # in the close reason even when the position isn't in the selected book.
            decisions.append(
                {
                    "primary": key[0],
                    "side": key[1],
                    "decision": "close",
                    "selection_rank": None,
                    "today_side": None,
                    "pair_score": None,
                    "reason": "not in today's selected top-N book",
                }
            )

    planned_opens: list[dict[str, Any]] = []
    for sel in selected:
        key = (_normalize_pair(sel["primary"]), sel["side"])
        if key in held_keys:
            continue
        planned_opens.append(
            {
                "selection_rank": int(sel["selection_rank"]),
                "primary": sel["primary"],
                "side": sel["side"],
                "expected_side": sel["expected_side"],
                "pair_score": sel["pair_score"],
                "abs_score": sel["abs_score"],
                "ranking_rank": int(sel["ranking_rank"]),
                "reason": sel["reason"],
            }
        )

    return decisions, planned_opens


# --------------------------------------------------------------------------------------
# Status JSON
# --------------------------------------------------------------------------------------


def _write_status(payload: dict[str, Any]) -> Path:
    p = _status_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p


def _append_history(payload: dict[str, Any]) -> None:
    """Append the run to both the structured JSONL and human-readable log files.

    Failures are swallowed (logged) so an I/O glitch here never aborts the trade
    pass. The dashboard panel reads the JSONL when the user clicks "Show history".
    """
    p_jsonl = _history_jsonl_path()
    p_log = _history_log_path()
    try:
        p_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with p_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    except OSError as e:
        logger.warning("Could not append history JSONL %s: %s", p_jsonl, e)

    try:
        with p_log.open("a", encoding="utf-8") as f:
            f.write(_format_history_text(payload))
            f.write("\n")
    except OSError as e:
        logger.warning("Could not append history log %s: %s", p_log, e)


def _format_history_text(p: dict[str, Any]) -> str:
    """Multi-line human-readable summary of one pass."""
    lines: list[str] = []
    started = p.get("started_at", "?")
    finished = p.get("finished_at", "?")
    fx_date = p.get("target_forecast_date", "?")
    dry = bool(p.get("dry_run", True))
    cfg = p.get("config", {}) or {}
    lines.append("=" * 90)
    lines.append(
        f"[{started} -> {finished}] forecast={fx_date} "
        f"top_n={cfg.get('top_n', '?')} max_open={cfg.get('max_open', '?')} "
        f"enabled={cfg.get('enabled', '?')} dry_run={dry}"
    )

    rank = p.get("ranking") or []
    if rank:
        lines.append("Ranking (top 10 by |pair_score|):")
        for r in rank[:10]:
            lines.append(
                f"  #{r.get('rank', '?'):>2}  {str(r.get('primary', '')):<10} {str(r.get('expected_side', '')):<6} "
                f"score={float(r.get('pair_score', 0.0)):+.4f}  |score|={float(r.get('abs_score', 0.0)):.4f}"
            )

    sel = p.get("selected_book") or []
    if sel:
        lines.append(f"Selected book (top {len(sel)}, currency-disjoint):")
        for s in sel:
            lines.append(
                f"  ##{s.get('selection_rank', '?')} {str(s.get('primary', '')):<10} {str(s.get('side', '')):<5} "
                f"|score|={float(s.get('abs_score', 0.0)):.4f}  rank={s.get('ranking_rank', '?')}"
            )
            lines.append(f"      reason: {s.get('reason', '')}")
    else:
        lines.append("Selected book: (empty)")

    skips = p.get("overlap_skips") or []
    if skips:
        lines.append("Overlap skips (candidates passed over for currency overlap):")
        for sk in skips:
            lines.append(
                f"  - {str(sk.get('primary', '')):<10} ranking_rank={sk.get('ranking_rank', '?')} "
                f"|score|={float(sk.get('abs_score', 0.0)):.4f}  {sk.get('skipped_because', '')}"
            )

    opens = p.get("open_positions") or []
    if opens:
        lines.append("Open positions on FXCM:")
        for o in opens:
            lines.append(
                f"  {str(o.get('primary', '')):<10} {str(o.get('side', '')):<5} "
                f"lots={float(o.get('lots', 0.0)):g} amount={int(o.get('amount', 0))} "
                f"open_rate={float(o.get('open_rate', 0.0))} gross_pl={float(o.get('gross_pl', 0.0))}"
            )
    else:
        lines.append("Open positions on FXCM: (none)")

    dec = p.get("decisions") or []
    if dec:
        lines.append("Decisions:")
        for d in dec:
            lines.append(
                f"  {str(d.get('decision', '')).upper():<5} {str(d.get('primary', '')):<10} {str(d.get('side', '')):<5} -- {d.get('reason', '')}"
            )

    plan = p.get("planned_opens") or []
    if plan:
        lines.append("Planned opens:")
        for pp in plan:
            lines.append(
                f"  OPEN  {str(pp.get('primary', '')):<10} {str(pp.get('side', '')):<5} -- {pp.get('reason', '')}"
            )

    actions = p.get("actions") or []
    if actions:
        lines.append("Actions executed:")
        for a in actions:
            label = str(a.get("kind", "")).upper()
            pair = str(a.get("primary", ""))
            status = str(a.get("status", ""))
            msg = a.get("message") or a.get("error") or ""
            lines.append(f"  {label} {pair} [{status}] {msg}")
    else:
        if dry:
            lines.append("Actions executed: (none - dry run because auto-trade disabled)")
        else:
            lines.append("Actions executed: (none)")

    errs = p.get("errors") or []
    if errs:
        lines.append("Errors:")
        for e in errs:
            lines.append(f"  [{e.get('step', '?')}] {e.get('error', '')}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------


def run_daily_check(*, dry_run_override: bool | None = None) -> dict[str, Any]:
    """Run one pass. Returns the status payload (also written to JSON + history)."""
    cfg: AutoTradeConfig = load_config()
    # ensure the JSON file exists on first run so the dashboard can write defaults
    if not Path(cfg._path).is_file():
        save_config(cfg)

    enabled = bool(cfg.enabled) if dry_run_override is None else (not dry_run_override and cfg.enabled)

    started_at = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    out: dict[str, Any] = {
        "started_at": started_at,
        "config": {
            "enabled": cfg.enabled,
            "top_n": cfg.top_n,
            "pct_equity_per_trade": cfg.pct_equity_per_trade,
            "leverage": cfg.leverage,
            "stop_pct": cfg.stop_pct,
            "max_open": cfg.max_open,
            "skip_unsubscribed": cfg.skip_unsubscribed,
        },
        "dry_run": not enabled,
        "errors": [],
    }

    # ----- 1. Load workbook and rank --------------------------------------------------
    try:
        xlsx = _workbook_path()
        signals, meta = _load_signals(xlsx)
        out["workbook_path"] = str(xlsx)
        out["workbook_mtime"] = dt.datetime.fromtimestamp(xlsx.stat().st_mtime).isoformat(timespec="seconds")
        out["target_forecast_date"] = _forecast_date_from_meta(meta, signals)

        ranking = _rank_pairs_today(signals, DEFAULT_MIN_ACCURACY)
        out["ranking"] = ranking.to_dict(orient="records") if not ranking.empty else []
    except Exception as e:  # noqa: BLE001
        msg = f"Failed to build ranking from workbook: {e}"
        logger.error(msg)
        out["errors"].append({"step": "ranking", "error": msg, "trace": traceback.format_exc()})
        _finalize_and_persist(out)
        return out

    # ----- 2. Select today's currency-disjoint top-N (BEFORE closing anything) --------
    #
    # IMPORTANT: selection runs first so that a position that is already on AND still
    # in today's top-N keeps its trade row (we never close-and-reopen the same pair).
    selected, overlap_skips = _select_top_pairs_no_overlap(ranking, top_n=int(cfg.top_n))
    out["selected_book"] = selected
    out["overlap_skips"] = overlap_skips
    if selected:
        logger.info(
            "Selected book: %s",
            ", ".join(f"#{s['selection_rank']} {s['primary']} {s['side']}" for s in selected),
        )

    # ----- 3. Query FXCM (open positions) ---------------------------------------------
    open_positions: list[dict[str, Any]] = []
    settings = None
    try:
        settings = fc_settings_from_env()
        open_positions = list_open_fc_trades(settings=settings)
        logger.info("Found %d open positions on FXCM account.", len(open_positions))
    except Exception as e:  # noqa: BLE001
        msg = f"Could not list FXCM open positions: {e}"
        logger.error(msg)
        out["errors"].append({"step": "list_trades", "error": msg, "trace": traceback.format_exc()})
        out["open_positions"] = []
        out["decisions"] = []
        out["planned_opens"] = []
        out["actions"] = []
        _finalize_and_persist(out)
        return out

    # ----- 4. Build keep/close diff (selection-first) ---------------------------------
    decisions, planned_opens = _build_book_diff(
        open_positions=open_positions,
        selected=selected,
    )
    # Cap auto-opens by max_open (after accounting for kept positions).
    n_kept = sum(1 for d in decisions if d["decision"] == "keep")
    open_slots = max(int(cfg.max_open) - n_kept, 0)
    planned_opens = planned_opens[:open_slots]

    # Enrich decisions with trade-row metadata (trade_id, lots, etc.) for display + execution.
    pos_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (_normalize_pair(str(p["primary"])), str(p["side"])): p for p in open_positions
    }
    for d in decisions:
        pos = pos_by_key.get((d["primary"], d["side"]))
        if pos is not None:
            d["trade_id"] = str(pos.get("trade_id", ""))
            d["lots"] = float(pos.get("lots", 0.0))
            d["amount"] = int(pos.get("amount", 0))
            d["open_rate"] = float(pos.get("open_rate", 0.0))
            d["gross_pl"] = float(pos.get("gross_pl", 0.0))

    out["open_positions"] = open_positions
    out["decisions"] = decisions
    out["planned_opens"] = planned_opens
    out["actions"] = []

    # ----- 5. Execute (only if enabled) -----------------------------------------------
    if not enabled:
        logger.info("DRY-RUN: auto-trade disabled in config.")
        _finalize_and_persist(out)
        return out

    # 5a. Close any open position that is NOT in today's selected book.
    #     (Positions already in the selected book are KEPT — no close-and-reopen.)
    for d in decisions:
        if d["decision"] != "close":
            continue
        try:
            msg = close_fc_trade(trade_id=d["trade_id"], settings=settings)
            out["actions"].append(
                {"kind": "close", "primary": d["primary"], "side": d["side"], "trade_id": d["trade_id"], "status": "ok", "message": msg}
            )
            logger.info("Closed %s %s (%s): %s", d["primary"], d["side"], d["trade_id"], msg)
        except Exception as e:  # noqa: BLE001
            err = f"close failed for {d['primary']} {d['side']} ({d.get('trade_id', '?')}): {e}"
            logger.error(err)
            out["actions"].append(
                {"kind": "close", "primary": d["primary"], "side": d["side"], "trade_id": d.get("trade_id", ""), "status": "error", "error": err}
            )

    # 5b. Open the planned new top-N (one at a time so a single failure doesn't cascade).
    for plan in planned_opens:
        try:
            msg = execute_fc_market_order(
                primary=plan["primary"],
                is_buy=(plan["side"] == "buy"),
                settings=settings,
                pct_equity=cfg.pct_equity_per_trade,
                leverage=cfg.leverage,
                stop_loss_pct=cfg.stop_pct,
            )
            out["actions"].append(
                {"kind": "open", "primary": plan["primary"], "side": plan["side"], "status": "ok", "message": msg}
            )
            logger.info("Opened %s %s: %s", plan["primary"], plan["side"], msg)
        except Exception as e:  # noqa: BLE001
            err_str = str(e)
            # Common case: pair not price-subscribed on this FXCM account (status 'V').
            if cfg.skip_unsubscribed and "subscription_status" in err_str:
                out["actions"].append(
                    {"kind": "skip_unsubscribed", "primary": plan["primary"], "side": plan["side"], "status": "skipped", "message": err_str}
                )
                logger.warning("Skipping %s %s -- not price-subscribed on FXCM account.", plan["primary"], plan["side"])
                continue
            err = f"open failed for {plan['primary']} {plan['side']}: {e}"
            logger.error(err)
            out["actions"].append(
                {"kind": "open", "primary": plan["primary"], "side": plan["side"], "status": "error", "error": err}
            )

    _finalize_and_persist(out)
    return out


def _finalize_and_persist(out: dict[str, Any]) -> None:
    """Stamp ``finished_at``, write the live status JSON, and append the history."""
    out["finished_at"] = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    _write_status(out)
    _append_history(out)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Daily FX auto-trade check.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force analysis only; never execute trades even if config.enabled is true.",
    )
    args = parser.parse_args(argv)
    try:
        payload = run_daily_check(dry_run_override=args.dry_run if args.dry_run else None)
    except Exception as e:  # noqa: BLE001
        logger.exception("daily_trade_check crashed")
        _write_status(
            {
                "started_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
                "finished_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
                "dry_run": True,
                "errors": [{"step": "main", "error": str(e), "trace": traceback.format_exc()}],
            }
        )
        return 1
    return 0 if not payload.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
