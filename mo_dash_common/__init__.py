"""Shared utilities across Mo_Dash tabs.

This package is the single source of truth for cross-tab helpers
(path resolution, layout discovery, etc.). It is vendored into per-tab
published repos by ``Mo_Dash/scripts/publish_subtrees.ps1`` so each
sub-repo remains runnable standalone.

Inside the Mo_Dash monorepo, this package sits at the workspace root
(``Mo_Dash/mo_dash_common/``). Per-tab code adds ``Mo_Dash/`` to
``sys.path`` (handled automatically by the
``utils/mo_dash_layout.py`` shim under ``ETF/ETF Forecaster/project``).
"""

from __future__ import annotations

from . import layout

__all__ = ["layout"]
