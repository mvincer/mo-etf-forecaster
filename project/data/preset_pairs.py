"""Preset lead–lag ETF pairs (structural relationships). Category 9 (sentiment) has no price pair."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresetPair:
    category: str
    sym_a: str
    sym_b: str
    leader: str
    follower: str


PRESET_PAIRS: list[PresetPair] = [
    PresetPair("1. Size: large vs small", "SPY", "IWM", "SPY", "IWM"),
    PresetPair("2a. Supply chain: metals vs industrials", "XME", "XLI", "XME", "XLI"),
    PresetPair("2b. Supply chain: E&P vs refiners", "XOP", "CRAK", "XOP", "CRAK"),
    PresetPair("3. Factor: growth vs value", "IVW", "IVE", "IVW", "IVE"),
    PresetPair("4. Asset class: equities vs long bonds", "SPY", "TLT", "TLT", "SPY"),
    PresetPair("5. Concentration: cap-weight vs equal-weight", "SPY", "RSP", "SPY", "RSP"),
    PresetPair("6a. Geography: US vs emerging", "SPY", "EEM", "SPY", "EEM"),
    PresetPair("6b. Geography: US vs developed ex-US", "SPY", "EFA", "SPY", "EFA"),
    PresetPair("7. Risk premium: high beta vs low vol", "SPHB", "SPLV", "SPHB", "SPLV"),
    PresetPair("8a. Commodity vs equity: crude vs energy", "USO", "XLE", "USO", "XLE"),
    PresetPair("8b. Commodity vs equity: copper vs miners", "CPER", "COPX", "CPER", "COPX"),
    PresetPair("10. Real estate vs regional banks", "IYR", "KRE", "KRE", "IYR"),
]


def all_preset_tickers() -> set[str]:
    s: set[str] = set()
    for p in PRESET_PAIRS:
        s.add(p.sym_a)
        s.add(p.sym_b)
    return s
