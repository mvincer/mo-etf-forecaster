"""Infer **buy** / **sell** on the quoted FX pair from dashboard / workbook columns."""

from __future__ import annotations

import pandas as pd


def side_from_text(s: pd.Series) -> pd.Series:
    """Map free-text position/recommendation to 'buy' or 'sell'; else NA."""
    t = s.astype(str).str.lower()
    t = t.replace({"nan": "", "none": "", "<na>": ""})
    has_buy = t.str.contains(r"\bbuy\b|\blong\b|bullish|\bbull\b", regex=True, na=False)
    has_sell = t.str.contains(r"\bsell\b|\bshort\b|bearish|\bbear\b", regex=True, na=False)
    has_up = t.str.contains(r"\bup\b", regex=True, na=False)
    has_down = t.str.contains(r"\bdown\b", regex=True, na=False)
    buy = (has_buy | has_up) & ~has_sell & ~has_down
    sell = (has_sell | has_down) & ~has_buy & ~has_up
    side = pd.Series(pd.NA, index=s.index, dtype="object")
    side = side.mask(buy, "buy")
    side = side.mask(sell, "sell")
    return side


def classify_signal_side(df: pd.DataFrame) -> pd.Series:
    """Infer 'buy' or 'sell' per row from position, recommendation, pred_class, or p_buy/p_sell."""
    idx = df.index
    out = pd.Series(pd.NA, index=idx, dtype="object")
    if "position" in df.columns:
        out = out.fillna(side_from_text(df["position"]))
    if "recommendation" in df.columns:
        out = out.fillna(side_from_text(df["recommendation"]))
    if "pred_class" in df.columns:
        need = out.isna()
        if need.any():
            pc = df["pred_class"]
            num = pd.to_numeric(pc, errors="coerce")
            pred_side = pd.Series(pd.NA, index=idx, dtype="object")
            pred_side = pred_side.mask(num == 1, "buy")
            pred_side = pred_side.mask(num == 0, "sell")
            pred_side = pred_side.mask(num == -1.0, "sell")
            out = out.fillna(pred_side)
            still = out.isna() & pc.notna() & num.isna()
            if still.any():
                out = out.fillna(side_from_text(pc))
    if "p_buy" in df.columns and "p_sell" in df.columns:
        pb = pd.to_numeric(df["p_buy"], errors="coerce")
        ps = pd.to_numeric(df["p_sell"], errors="coerce")
        m = out.isna() & pb.notna() & ps.notna() & (pb != ps)
        guess = pd.Series(pd.NA, index=idx, dtype="object")
        guess = guess.mask(m & (pb > ps), "buy")
        guess = guess.mask(m & (ps > pb), "sell")
        out = out.fillna(guess)
    return out
