"""Backward-compat shim. Real implementation lives in ``mo_dash_common.layout``.

Why this exists:

* Inside the Mo_Dash monorepo, ``mo_dash_common/`` sits at the workspace root
  (``Mo_Dash/mo_dash_common/``). This shim walks up from its own location until
  it finds that package, then adds the parent directory to ``sys.path`` so
  ``from mo_dash_common import layout`` resolves.
* When this folder is published as a standalone sub-repo (e.g.
  ``mo-etf-forecaster``) by ``scripts/publish_subtrees.ps1``, the publisher
  vendors ``mo_dash_common/`` at the sub-repo root. The same walk-up logic
  finds it there too, so existing ``from utils.mo_dash_layout import …``
  imports keep working without change.

Prefer ``from mo_dash_common import layout`` (or
``from mo_dash_common.layout import …``) in new code.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_mo_dash_common_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "mo_dash_common"
        if (candidate / "__init__.py").is_file():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


_bootstrap_mo_dash_common_on_path()

from mo_dash_common.layout import (  # noqa: E402  (import after sys.path setup)
    fx_data_collect_import_cwd,
    fx_data_collect_package_dir,
    fx_nl_project_root,
    mo_dash_fx_forecast_xlsx,
    mo_dash_workspace_root,
)

__all__ = [
    "mo_dash_workspace_root",
    "fx_data_collect_package_dir",
    "fx_data_collect_import_cwd",
    "mo_dash_fx_forecast_xlsx",
    "fx_nl_project_root",
]
