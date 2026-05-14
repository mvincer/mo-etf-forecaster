"""
Expanded macro / market catalog: Yahoo Finance tickers and FRED series IDs.

FRED IDs are best-effort; failures are recorded in the manifest (some OECD series are monthly → ffilled to daily).
"""

from __future__ import annotations

# --- Yahoo Finance: daily OHLC (mostly US sessions; indices may update overnight) ---
YAHOO_MARKETS: dict[str, str] = {
    # Commodities
    "gold": "GC=F",
    "crude_wti": "CL=F",
    "crude_brent": "BZ=F",
    # Vol / US benchmarks
    "vix": "^VIX",
    "vvix": "^VVIX",
    "sp500": "^GSPC",
    "spy": "SPY",
    "nasdaq_comp": "^IXIC",
    "qqq": "QQQ",
    # Americas
    "tsx": "^GSPTSE",  # S&P/TSX Composite
    # Europe
    "cac40": "^FCHI",
    "dax": "^GDAXI",
    # Asia / Pacific
    "nikkei": "^N225",
    "ftse100": "^FTSE",
    "smi": "^SSMI",
    "asx200": "^AXJO",
    "nz50": "^NZ50",
    # --- Extra FX-relevant (10+) ---
    "dxy_futures": "DX-Y.NYB",
    "eem": "EEM",
    "hyg": "HYG",
    "tlt": "TLT",
    "iwm": "IWM",
    "copper": "HG=F",
    "silver": "SI=F",
    "natgas": "NG=F",
    "uup": "UUP",
}

# --- FRED: macro rates & levels (daily where published; else lower freq ffilled) ---
# Labels prefixed fund_* in output. IDs verified where possible; invalid IDs surface as download failures.
EXPANDED_FRED_SERIES: dict[str, str] = {
    # US money / curve
    "us_eff": "DFF",
    "us_sofr": "SOFR",
    "us_tbill_3m": "DTB3",
    "us_2y": "DGS2",
    "us_10y": "DGS10",
    "us_dxy_broad": "DTWEXBGS",
    # Euro area / DE / FR (shared ECB short rate; separate long yields OECD)
    "eu_ecb_deposit": "ECBDFR",
    "de_10y_oecd": "IRLTLT01DEM156N",
    "fr_10y_oecd": "IRLTLT01FRM156N",
    # UK
    "uk_bank_rate": "BOERUKM",
    "uk_10y_oecd": "IRLTLT01GBM156N",
    "uk_sonia": "UKSONIAIUSD",
    # Japan
    "jp_overnight_proxy": "IRSTCB01JPM156N",
    "jp_10y_oecd": "IRLTLT01JPM156N",
    # Switzerland / Canada / AU / NZ (OECD 10y often monthly)
    "ch_10y_oecd": "IRLTLT01CHM156N",
    "ca_10y_oecd": "IRLTLT01CAM156N",
    "au_call_overnight": "IRSTCI01AUM156N",
    "au_10y_oecd": "IRLTLT01AUM156N",
    "nz_10y_oecd": "IRLTLT01NZM156N",
    # Canada short-rate OECD proxy (often monthly)
    "ca_overnight_proxy": "IRSTCB01CAM156N",
    # Equity index levels on FRED (often monthly — still useful)
    "idx_us_sp500": "SP500",
    "idx_us_nasdaq": "NASDAQCOM",
    "idx_jp_nikkei": "NIKKEI225",
    "idx_uk_ftse_monthly": "SPASTT01GBM661N",
    "idx_de_equity_monthly": "SPASTT01DEM661N",
    # Policy / stress
    "ted_spread": "TEDRATE",
    "hy_oas": "BAMLH0A0HYM2",
    # EURUSD macro reference (not a rate)
    "dexuseu": "DEXUSEU",
}
