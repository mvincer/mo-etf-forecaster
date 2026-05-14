"""Run FXCM ForexConnect orders under Python 3.7 (``.venv-fc``).

The main dashboard may use Python 3.11+; this worker is spawned with a JSON payload
path so ``forexconnect`` loads in-process here (``_FC_WORKER_PROCESS`` prevents the
parent from delegating to this worker again).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: fxcm_fc_worker.py <payload.json>", file=sys.stderr)
        return 2
    payload_path = Path(sys.argv[1])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    os.environ["_FC_WORKER_PROCESS"] = "1"
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from utils.fxcm_forexconnect_trade import (  # noqa: PLC0415
        FxcmForexConnectSettings,
        execute_fc_market_order,
        execute_fc_market_orders_sequence,
        test_fc_login,
    )

    op = payload.get("op")
    settings = FxcmForexConnectSettings(**payload["settings"])

    if op == "login_test":
        print(test_fc_login(settings))
        return 0
    if op == "single":
        lots_v = payload.get("lots")
        pc = payload.get("pct_equity")
        lv = payload.get("leverage")
        print(
            execute_fc_market_order(
                primary=str(payload["primary"]),
                is_buy=bool(payload["is_buy"]),
                settings=settings,
                lots=float(lots_v) if lots_v is not None else None,
                pct_equity=float(pc) if pc is not None else None,
                leverage=float(lv) if lv is not None else None,
                stop_loss_pct=float(payload.get("stop_loss_pct", 1.0)),
                order_wait_sec=float(payload.get("order_wait_sec", 25.0)),
            )
        )
        return 0
    if op == "sequence":
        legs = [(str(a), bool(b)) for a, b in payload["legs"]]
        lpl = payload.get("lots_per_leg")
        pce = payload.get("pct_equity_each")
        lev = payload.get("leverage")
        print(
            execute_fc_market_orders_sequence(
                legs=legs,
                settings=settings,
                lots_per_leg=float(lpl) if lpl is not None else None,
                pct_equity_each=float(pce) if pce is not None else None,
                leverage=float(lev) if lev is not None else None,
                stop_loss_pct=float(payload.get("stop_loss_pct", 1.0)),
                order_wait_sec=float(payload.get("order_wait_sec", 25.0)),
            )
        )
        return 0

    print(f"unknown op: {op}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
