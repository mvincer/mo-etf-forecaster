from __future__ import annotations

from pathlib import Path

# Calendar anchor: all FX series trimmed to [START_DATE, latest available]
START_DATE = "2000-01-01"

FX_PAIRS: list[str] = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "USD/CHF",
    "AUD/USD",
    "NZD/USD",
    "USD/CAD",
    "EUR/JPY",
    "EUR/GBP",
    "GBP/JPY",
]

# Yahoo Finance suffix for spot FX (fallback / tail extension)
YAHOO_FX_MAP: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/GBP": "EURGBP=X",
    "GBP/JPY": "GBPJPY=X",
}

# Human label -> FRED series id (rates, USD index, equity proxies). Monthly series forward-filled when merged daily.
FRED_SERIES: dict[str, str] = {
    "dxy": "DTWEXBGS",
    "us_overnight_eff": "DFF",
    "us_sofr": "SOFR",
    "us_2y": "DGS2",
    "us_10y": "DGS10",
    "eu_overnight_ecb": "ECBDFR",
    "de_10y": "IRLTLT01DEM156N",
    "uk_bank_rate": "BOERUKM",
    "uk_10y": "IRLTLT01GBM156N",
    "jp_overnight_call": "IRSTCB01JPM156N",
    "jp_10y": "IRLTLT01JPM156N",
    "ch_10y": "IRLTLT01CHM156N",
    "ca_10y": "IRLTLT01CAM156N",
    "au_overnight": "IRSTCI01AUM156N",
    "idx_us_sp500": "SP500",
    "idx_us_nasdaq": "NASDAQCOM",
    "idx_jp_nikkei": "NIKKEI225",
    "idx_uk_ftse": "SPASTT01GBM661N",
    "idx_de_dax": "SPASTT01DEM661N",
}


def default_repo_root() -> Path:
    """Under package folder unless ``FX_DATA_REPO_ROOT`` is set (handled in repository.py)."""
    return Path(__file__).resolve().parent / "data"
