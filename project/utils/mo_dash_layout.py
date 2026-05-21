"""Resolve Mo_Dash workspace paths (FX workbook, ``fx_data_collect`` package, etc.)."""

from __future__ import annotations

import os
from pathlib import Path

# Under workspace root (see Mo_Dash/STRUCTURE.txt).
_MO_DASH_FX_FORECAST_PARTS: tuple[str, ...] = (
    "FX",
    "FX forecasts",
    "non-linear FX forecast - daily_binary_fx_forecast",
)


def mo_dash_workspace_root(etf_root: Path) -> Path:
    """Folder that contains ``FX/``, ``ETF/``, ``Dashboard/`` (not the ETF app git root).

    Order:

    1. ``MO_DASH_ROOT`` env.
    2. Repo at ``…/Mo_Dash/ETF/ETF Forecaster`` → ``…/Mo_Dash``.
    3. Sibling ``<parent>/Mo_Dash`` when ``etf_root`` is ``…/ETF Forecaster`` (classic layout).
    4. Legacy: ``<etf_root>/Mo_Dash``.
    """
    env = (os.environ.get("MO_DASH_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    er = etf_root.resolve()
    if (
        er.name.lower() == "etf forecaster"
        and er.parent.name.lower() == "etf"
        and er.parent.parent.name.lower() == "mo_dash"
    ):
        return er.parent.parent
    sib = er.parent / "Mo_Dash"
    if sib.is_dir() and (sib / "FX").is_dir():
        return sib.resolve()
    return er / "Mo_Dash"


def fx_data_collect_package_dir(etf_root: Path) -> Path:
    """Directory that **contains** the ``fx_data_collect`` package (…/fx_data_collect/__init__.py)."""
    return mo_dash_workspace_root(etf_root) / "FX" / "FX forecasts" / "fx_data_collect"


def fx_data_collect_import_cwd(etf_root: Path) -> Path:
    """Working directory for ``python -m fx_data_collect.*`` (parent of the package folder)."""
    return fx_data_collect_package_dir(etf_root).parent


def mo_dash_fx_forecast_xlsx(etf_root: Path) -> Path:
    return mo_dash_workspace_root(etf_root).joinpath(
        *_MO_DASH_FX_FORECAST_PARTS,
        "daily_binary_fx_forecast.xlsx",
    )


def fx_nl_project_root(etf_root: Path) -> Path:
    """Directory of the FX non-linear project (contains ``fxnl/`` and ``daily_binary_fx_forecast.xlsx``).

    Inside Mo_Dash this is ``…/Mo_Dash/FX/FX forecasts/non-linear FX forecast - daily_binary_fx_forecast/``.
    Set ``FXNL_PROJECT_ROOT`` to override.
    """
    env = (os.environ.get("FXNL_PROJECT_ROOT") or os.environ.get("FX_FORECAST_PROJECT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return mo_dash_workspace_root(etf_root).joinpath(*_MO_DASH_FX_FORECAST_PARTS)
