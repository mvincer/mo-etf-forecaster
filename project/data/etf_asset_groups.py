"""
Map tickers to a canonical *underlying* so we (1) pick at most one fund per economic exposure
in the top-volume list and (2) drop cointegration pairs that are economically the same tracker.

Unknown tickers default to their own id so they remain eligible.
"""

from __future__ import annotations

import numpy as np

# Tickers sharing the same value are treated as the same underlying asset / index exposure.
UNDERLYING_MAP: dict[str, str] = {
    # US large cap — S&P 500 and near-clone trackers from different issuers
    "SPY": "us_sp500",
    "VOO": "us_sp500",
    "IVV": "us_sp500",
    "SPLG": "us_sp500",
    # Broad US total market
    "VTI": "us_total_stock_market",
    "ITOT": "us_total_stock_market",
    # Nasdaq-100
    "QQQ": "ndx_100",
    # Russell 2000
    "IWM": "us_russell_2000",
    # Gold bullion
    "GLD": "gold_bullion",
    "IAU": "gold_bullion",
    # Silver / energy commodities
    "SLV": "silver_bullion",
    "USO": "wti_crude",
    "UNG": "natural_gas",
    # Crypto spot / trust trackers (same coin, different wrappers)
    "IBIT": "bitcoin_spot_etf",
    "GBTC": "bitcoin_spot_etf",
    "BITO": "bitcoin_futures_etf",
    "ETHA": "ethereum_spot_etf",
    "ETHE": "ethereum_spot_etf",
    # International core
    "EFA": "msci_eafe",
    "VEA": "msci_eafe",
    "IEFA": "msci_eafe",
    "SCHF": "msci_eafe",
    "EEM": "msci_emerging",
    "VWO": "msci_emerging",
    "IEMG": "msci_emerging",
    # US aggregate bonds
    "AGG": "us_agg_bond",
    "BND": "us_agg_bond",
    # S&P MidCap 400
    "MDY": "sp_midcap_400",
    "IJH": "sp_midcap_400",
    "SPMD": "sp_midcap_400",
    # Semiconductors (same theme, different issuers)
    "SMH": "semiconductors",
    "SOXX": "semiconductors",
    # Biotech broad
    "XBI": "us_biotech",
    "IBB": "us_biotech",
    # Leveraged same benchmark (still one economic “beta bucket” for diversification of *funds*)
    "SPXL": "sp500_3x_long",
    "UPRO": "sp500_3x_long",
    "SPXS": "sp500_3x_short",
    "SSO": "sp500_2x_long",
    "TQQQ": "ndx_3x_long",
    "SQQQ": "ndx_3x_short",
    "QLD": "ndx_2x_long",
    "TZA": "russell2000_3x_short",
    # High-dividend US equity (overlapping economic tilt)
    "VYM": "us_high_dividend_core",
    "SCHD": "us_high_dividend_core",
    "DVY": "us_high_dividend_core",
    "HDV": "us_high_dividend_core",
    # Dividend growth
    "DGRO": "us_dividend_growth",
    "DGRW": "us_dividend_growth",
    # Investment-grade corporate intermediate
    "VCIT": "usd_ig_corp_intermediate",
    "BIV": "usd_ig_corp_intermediate",
    # Short / ultra-short Treasuries
    "VGSH": "usd_tsy_short",
    "SHY": "usd_tsy_short",
    "BSV": "usd_tsy_short",
    "SHV": "usd_tsy_ultrashort",
    # IG treasuries (intermediate)
    "VGIT": "usd_tsy_intermediate",
    "IEI": "usd_tsy_intermediate",
    # US REITs (broad listed real estate)
    "VNQ": "us_reit_broad",
    "SCHH": "us_reit_broad",
    "IYR": "us_reit_broad",
    "RWR": "us_reit_broad",
    # Preferred stock
    "PFF": "usd_preferred_stock",
    "PGX": "usd_preferred_stock",
    # Clean energy / climate equity (high overlap)
    "ICLN": "clean_energy_equity",
    "QCLN": "clean_energy_equity",
    "SMOG": "clean_energy_equity",
    # Oil & gas producers (similar equity beta)
    "XOP": "us_oil_gas_producers",
    "OIH": "us_oil_gas_producers",
    # Auto / EV thematic overlap
    "DRIV": "ev_auto_thematic",
    "IDRV": "ev_auto_thematic",
    # ESG broad US
    "ESGU": "us_esg_broad",
    "ESGV": "us_esg_broad",
    "USSG": "us_esg_broad",
    # MLP energy infrastructure (similar sleeve)
    "AMLP": "mlp_energy_infra",
    "MLPA": "mlp_energy_infra",
    "ENFR": "mlp_energy_infra",
    # Global mining metals (overlapping)
    "PICK": "global_mining_metals",
    "XME": "global_mining_metals",
    # US financials sector vs cap-weighted financials ETF overlap
    "VFH": "us_financials_sector",
    "IYF": "us_financials_sector",
    # Cloud / software tech (often bundled)
    "IGV": "us_software_cloud",
    "FINX": "us_software_cloud",
    # Cyber
    "HACK": "cyber_security_equity",
    "CIBR": "cyber_security_equity",
    # Robotics / automation
    "BOTZ": "robotics_automation",
    "ROBO": "robotics_automation",
    # Homebuilders / housing
    "XHB": "us_homebuilders",
    "ITB": "us_homebuilders",
    # Covered-call income on major indices (distinct from pure index but highly redundant pairwise)
    "XYLD": "covered_call_sp500",
    "QYLD": "covered_call_ndx100",
    "RYLD": "covered_call_russell2000",
    # JEPI/JEPQ are option-income on S&P / Nasdaq — keep separate ids (different structures)
}

# |β−1| below this (log–log OLS) with correlation above threshold ⇒ treat as duplicate trackers.
BETA_NEAR_ONE_TOL = 0.04
LOG_LEVEL_CORR_DUP = 0.997


def underlying_group(ticker: str) -> str:
    t = ticker.upper().strip()
    return UNDERLYING_MAP.get(t, f"asset:{t}")


def diverse_top_by_underlying(
    scores: list[tuple[str, float]],
    top_n: int = 100,
) -> list[str]:
    """
    `scores` sorted by volume descending. Keep the highest-volume fund per underlying_group
    until `top_n` tickers are chosen.
    """
    seen: set[str] = set()
    out: list[str] = []
    for sym, _vol in scores:
        g = underlying_group(sym)
        if g in seen:
            continue
        seen.add(g)
        out.append(sym)
        if len(out) >= top_n:
            break
    return out


def same_underlying(a: str, b: str) -> bool:
    return underlying_group(a) == underlying_group(b)


def is_near_duplicate_tracker(beta: float, ly, lx) -> bool:
    """Very similar log levels (β≈1) and near-perfect correlation ⇒ same economic tape."""
    if abs(float(beta) - 1.0) >= BETA_NEAR_ONE_TOL:
        return False
    y = np.asarray(ly.values, dtype=float)
    x = np.asarray(lx.values, dtype=float)
    if len(y) < 10 or len(x) != len(y):
        return False
    if np.std(y) < 1e-12 or np.std(x) < 1e-12:
        return True
    c = np.corrcoef(y, x)[0, 1]
    if np.isnan(c):
        return False
    return float(c) >= LOG_LEVEL_CORR_DUP


def should_skip_cointegration_pair(y_sym: str, x_sym: str, beta: float, ly, lx) -> bool:
    if same_underlying(y_sym, x_sym):
        return True
    if is_near_duplicate_tracker(beta, ly, lx):
        return True
    return False
