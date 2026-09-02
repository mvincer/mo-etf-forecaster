"""Orchestrates the staged search + outer walk-forward and persists the track record.

Selection discipline:
- Stages 1-4 (block ablation + model search) see only the DEVELOPMENT slice (the first
  `dev_frac` of pooled group history).
- The outer walk-forward runs strictly AFTER the development slice; those predictions form
  the OOS track record and are never used to pick blocks or hyperparameters.
- Champion switching along the way uses only trailing OOS data available at that point
  (sticky rule), so the champion series is itself point-in-time.

Run:  python -m etf_forecaster.pipeline.train_search [--full] [--tickers SPY,QQQ] [--jobs 8]
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from etf_forecaster import store
from etf_forecaster.config import all_tickers, search_cfg, ticker_group, universe_cfg
from etf_forecaster.features.assemble import load_features
from etf_forecaster.labels import make_labels
from etf_forecaster.models.magnitude import QuantileMagnitude
from etf_forecaster.search.ablation import AblationRun
from etf_forecaster.search.blocks import select_columns
from etf_forecaster.search.tuning import model_search
from etf_forecaster.validation.walkforward import (
    Dataset,
    build_dataset,
    pool_datasets,
    run_walkforward,
    score_predictions,
)

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ data prep

def _load_ticker_data(ticker: str, horizons: list[int]) -> dict[int, Dataset] | None:
    from etf_forecaster.data import yahoo
    feats = load_features(ticker)
    bars = yahoo.load_bars(ticker)
    if feats is None or bars is None:
        return None
    labels = make_labels(bars, ticker, tuple(horizons)).reindex(feats.index)
    out = {}
    for h in horizons:
        ds = build_dataset(feats, labels, h, ticker)
        if ds is not None:
            out[h] = ds
    return out or None


def _dev_end(ds: Dataset, dev_frac: float) -> np.datetime64:
    ud = np.unique(ds.dates)
    return ud[int(len(ud) * dev_frac)]


# ------------------------------------------------------------------ stage 1-4 per group

def search_group(group: str, members: list[str], horizon: int,
                 data: dict[str, dict[int, Dataset]], cfg: dict, *, fast: bool,
                 n_jobs: int = 2) -> dict | None:
    parts = [data[t][horizon] for t in members if t in data and horizon in data[t]]
    pooled = pool_datasets(parts)
    if pooled is None:
        return None
    fastcfg = cfg["fast"]
    dev_frac = fastcfg["dev_frac"]
    dev_end = _dev_end(pooled, dev_frac)
    geometry = {
        "min_train": fastcfg["screen_min_train"] if fast else cfg["walkforward"]["min_train_bars"],
        "refit_every": fastcfg["screen_refit_every"] if fast else 63,
        "end_date": dev_end,
    }
    ab = AblationRun(
        pooled, horizon=horizon, geometry=geometry,
        screen_model=cfg["ablation"]["screen_model"],
        screen_params={"n_estimators": 200}, group=group, n_jobs=n_jobs)
    result = ab.run(
        cfg["ablation"]["candidate_blocks"],
        greedy_min_gain=cfg["ablation"]["greedy_min_gain"],
        exhaustive_max=(fastcfg["exhaustive_max_blocks"] if fast
                        else cfg["ablation"]["exhaustive_max_blocks"]))

    roster = fastcfg["s4_roster"] if fast else cfg["models"]["full"]
    n_trials = cfg["model_search"]["n_trials_fast"] if fast else cfg["model_search"]["n_trials"]
    top, s4_records = model_search(
        pooled, horizon=horizon, geometry=geometry, block_sets=result["top_sets"],
        roster=roster, window_grid=cfg["model_search"]["window_grid"],
        n_trials=n_trials, group=group, n_jobs=n_jobs)
    result["top_configs"] = top
    result["records"] = ab.records + s4_records
    result["dev_end"] = str(pd.Timestamp(dev_end).date())
    return result


# ------------------------------------------------------------------ outer walk-forward

def outer_walkforward(ticker: str, horizon: int, ds: Dataset, configs: list[dict],
                      dev_end: np.datetime64, cfg: dict, *, fast: bool,
                      n_jobs: int = 2) -> tuple[pd.DataFrame, dict]:
    """OOS predictions after dev_end for every config + point-in-time champion series."""
    fastcfg = cfg["fast"]
    refit = fastcfg["outer_refit_every"] if fast else cfg["walkforward"]["refit_every"]
    sel = cfg["selection"]
    all_preds: list[pd.DataFrame] = []
    per_config: dict[str, pd.DataFrame] = {}

    for i, c in enumerate(configs):
        cols = select_columns(ds.feature_names, c["blocks"], window=c.get("window", 20))
        sub = ds.subset_columns(cols)
        label = f"{c['model']}#{i}"
        preds = run_walkforward(
            sub, c["model"], c.get("params", {}),
            min_train=cfg["walkforward"]["min_train_bars"] // 2,
            refit_every=refit, horizon=horizon, start_date=dev_end,
            n_jobs=n_jobs, calibrate=True,
            calib_frac=cfg["walkforward"]["calibration_frac"])
        if not len(preds):
            continue
        preds["model"] = label
        preds["horizon"] = horizon
        per_config[label] = preds.set_index("date")
        all_preds.append(preds)

    if not per_config:
        return pd.DataFrame(), {}

    # ensemble = mean probability across configs
    merged = pd.concat([p["p_up"].rename(k) for k, p in per_config.items()], axis=1).sort_index()
    base = next(iter(per_config.values()))
    ens = pd.DataFrame({
        "p_up": merged.mean(axis=1),
        "y": base["y"].reindex(merged.index),
        "ret": base["ret"].reindex(merged.index),
    }).dropna(subset=["p_up"])
    ens["ticker"] = ticker
    ens["model"] = "ensemble"
    ens["horizon"] = horizon
    per_config["ensemble"] = ens
    all_preds.append(ens.reset_index().rename(columns={"index": "date"}))

    # point-in-time champion: re-evaluate on trailing OOS every `refit` dates, sticky
    dates = merged.index.to_numpy()
    labels_ = list(per_config.keys())
    champion = labels_[0]
    champ_rows = []
    trailing = cfg["selection"]["trailing_window"]
    for s in range(0, len(dates), refit):
        chunk = dates[s: s + refit]
        if s >= trailing // 2:
            past = slice(max(0, s - trailing), s)
            scores = {}
            for k in labels_:
                pp = per_config[k].reindex(dates[past]).dropna(subset=["p_up"])
                if len(pp) >= 50:
                    sc = score_predictions(pp.reset_index())
                    scores[k] = sc["log_loss"]
            if scores:
                best = min(scores, key=scores.get)
                cur = scores.get(champion, np.inf)
                if best != champion and scores[best] < cur - sel["sticky_margin"]:
                    champion = best
        pc = per_config[champion].reindex(chunk).dropna(subset=["p_up"]).copy()
        pc["champion_of"] = champion
        champ_rows.append(pc)
    champ = pd.concat(champ_rows) if champ_rows else pd.DataFrame()
    if len(champ):
        champ["ticker"] = ticker
        champ["model"] = "champion"
        champ["horizon"] = horizon
        all_preds.append(champ.reset_index().rename(columns={"index": "date"}))

    final_scores = {k: score_predictions(v.reset_index()) for k, v in per_config.items()}
    champion_meta = {
        "final_champion": champion,
        "configs": [{k: c[k] for k in ("model", "params", "blocks", "window")}
                    for c in configs],
        "oos_scores": {k: {m: (None if not np.isfinite(v) else round(float(v), 5))
                           for m, v in s.items() if m != "n"} | {"n": s["n"]}
                       for k, s in final_scores.items()},
    }
    out = pd.concat(all_preds, ignore_index=True)
    return out, champion_meta


def magnitude_quantiles(ticker: str, horizon: int, ds: Dataset, config: dict,
                        dev_end: np.datetime64, cfg: dict, *, fast: bool,
                        n_jobs: int = 2) -> pd.DataFrame:
    """Walk-forward conformal quantiles for the champion block set."""
    fastcfg = cfg["fast"]
    refit = fastcfg["magnitude_refit_every"] if fast else 63
    cols = select_columns(ds.feature_names, config["blocks"], window=config.get("window", 20))
    sub = ds.subset_columns(cols)
    udates = np.unique(sub.dates)
    lo = int(np.searchsorted(udates, dev_end))
    qs = tuple(cfg["magnitude"]["quantiles"])
    rows = []
    for s in range(lo, len(udates), refit):
        train_end_d = udates[s - horizon]
        test_d = udates[s: s + refit]
        tr = sub.dates < train_end_d
        te = np.isin(sub.dates, test_d)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        tr_idx = np.flatnonzero(tr)
        cut = int(len(tr_idx) * 0.85)
        try:
            qm = QuantileMagnitude(quantiles=qs, n_jobs=n_jobs,
                                   params={"n_estimators": 150})
            qm.fit(sub.X[tr_idx[:cut]], sub.ret[tr_idx[:cut]],
                   X_calib=sub.X[tr_idx[cut:]], y_calib=sub.ret[tr_idx[cut:]],
                   alpha=cfg["magnitude"]["conformal_alpha"])
            pred = qm.predict(sub.X[te])
        except Exception as exc:  # noqa: BLE001
            log.warning("magnitude fit failed %s h=%d: %s", ticker, horizon, exc)
            continue
        chunk = pd.DataFrame({"date": sub.dates[te]})
        for q in qs:
            chunk[f"q{int(q * 100):02d}"] = pred[q]
        rows.append(chunk)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["ticker"] = ticker
    out["horizon"] = horizon
    return out


# ------------------------------------------------------------------ driver

def _one_group_cell(group: str, members: list[str], horizon: int, cfg: dict,
                    fast: bool, jobs_inner: int) -> dict | None:
    data = {}
    for t in members:
        d = _load_ticker_data(t, [horizon])
        if d:
            data[t] = d
    try:
        return search_group(group, members, horizon, data, cfg, fast=fast, n_jobs=jobs_inner)
    except Exception as exc:  # noqa: BLE001
        log.exception("group search failed %s h=%d: %s", group, horizon, exc)
        return None


def _one_ticker_cell(ticker: str, horizon: int, group_result: dict, cfg: dict,
                     fast: bool, jobs_inner: int) -> tuple[pd.DataFrame, dict, pd.DataFrame] | None:
    data = _load_ticker_data(ticker, [horizon])
    if not data or horizon not in data[ticker]:
        return None
    ds = data[ticker][horizon]
    dev_end = np.datetime64(group_result["dev_end"])
    configs = group_result["top_configs"][:3]
    if not configs:
        return None
    preds, meta = outer_walkforward(ticker, horizon, ds, configs, dev_end, cfg,
                                    fast=fast, n_jobs=jobs_inner)
    if not len(preds):
        return None
    champ_cfg = next((c for i, c in enumerate(configs)
                      if f"{c['model']}#{i}" == meta["final_champion"]), configs[0])
    quants = magnitude_quantiles(ticker, horizon, ds, champ_cfg, dev_end, cfg,
                                 fast=fast, n_jobs=jobs_inner)
    meta["champion_config"] = {k: champ_cfg[k] for k in ("model", "params", "blocks", "window")}
    meta["dev_end"] = group_result["dev_end"]
    return preds, meta, quants


def run(*, fast: bool = True, tickers: list[str] | None = None,
        horizons: list[int] | None = None, jobs: int = 8) -> None:
    from joblib import Parallel, delayed

    cfg = search_cfg()
    horizons = horizons or cfg["horizons"]
    fastcfg = cfg["fast"]
    screened = fastcfg["horizons_screened"] if fast else horizons
    borrow = {2: 3, 4: 5} if fast else {}
    groups = universe_cfg()["groups"]
    tickers = tickers or all_tickers()
    jobs_inner = max(2, 32 // jobs)

    # ---- stages 1-4 at group level
    t0 = time.time()
    cells = [(g, m, h) for g, m in groups.items() for h in screened
             if any(t in tickers for t in m)]
    log.info("ablation: %d group cells (jobs=%d)", len(cells), jobs)
    results = Parallel(n_jobs=jobs, verbose=5)(
        delayed(_one_group_cell)(g, m, h, cfg, fast, jobs_inner) for g, m, h in cells)
    group_results: dict[tuple[str, int], dict] = {}
    all_records: list[dict] = []
    for (g, m, h), res in zip(cells, results):
        if res is None:
            continue
        group_results[(g, h)] = res
        all_records.extend(res.pop("records"))
    for h, src in borrow.items():
        for g in groups:
            if (g, src) in group_results and (g, h) not in group_results:
                borrowed = dict(group_results[(g, src)])
                borrowed["horizon"] = h
                group_results[(g, h)] = borrowed
    store.append_ablation(all_records)
    log.info("ablation stages done in %.0f min", (time.time() - t0) / 60)

    # ---- outer walk-forward per ticker x horizon
    t0 = time.time()
    todo = []
    for t in tickers:
        g = ticker_group(t)
        for h in horizons:
            if (g, h) in group_results:
                todo.append((t, h, group_results[(g, h)]))
    log.info("outer walk-forward: %d ticker cells", len(todo))
    outs = Parallel(n_jobs=jobs, verbose=5)(
        delayed(_one_ticker_cell)(t, h, gr, cfg, fast, jobs_inner) for t, h, gr in todo)

    champions: dict = store.read_champions()
    frames: list[pd.DataFrame] = []
    for (t, h, _), out in zip(todo, outs):
        if out is None:
            continue
        preds, meta, quants = out
        if len(quants):
            preds = preds.merge(quants, on=["date", "ticker", "horizon"], how="left")
        frames.append(preds)
        champions[f"{t}|{h}"] = meta
    if frames:
        hist = pd.concat(frames, ignore_index=True)
        hist["expected_return"] = hist.get("q50")
        store.append_history(hist)
    store.write_champions(champions)
    store.rebuild_scorecard()
    log.info("outer walk-forward done in %.0f min", (time.time() - t0) / 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full budgets (monthly job)")
    ap.add_argument("--tickers", default="")
    ap.add_argument("--horizons", default="")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(fast=not args.full,
        tickers=[t.strip() for t in args.tickers.split(",") if t.strip()] or None,
        horizons=[int(h) for h in args.horizons.split(",") if h.strip()] or None,
        jobs=args.jobs)


if __name__ == "__main__":
    main()
