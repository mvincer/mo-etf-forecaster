"""LSTM / TCN direction classifiers over the lagged-candle block.

The feature panel already contains the last-X-candle history per row (px_lag_ret_1..X and
px_lag_rng_1..K), so each tabular row is a ready-made sequence tensor and the walk-forward
machinery stays identical to the tree models. Degrades to unavailable if torch is missing.
"""

from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger(__name__)


class SequenceClassifier:
    """sklearn-ish wrapper: fit(X, y, feature_names=...) / predict_proba(X)."""

    def __init__(self, arch: str = "lstm", hidden: int = 32, layers: int = 1,
                 channels: int = 32, levels: int = 4, kernel: int = 3,
                 dropout: float = 0.2, epochs: int = 25, lr: float = 1e-3,
                 batch: int = 256, seed: int = 7, feature_names: list[str] | None = None):
        self.arch = arch
        self.hidden = hidden
        self.layers = layers
        self.channels = channels
        self.levels = levels
        self.kernel = kernel
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed
        self.feature_names = feature_names

    # ---------------------------------------------------------------- helpers
    def _seq_layout(self, names: list[str]) -> tuple[list[int], list[int]]:
        ret_idx = sorted(
            ((int(m.group(1)), i) for i, n in enumerate(names)
             if (m := re.match(r"px_lag_ret_(\d+)$", n))),
        )
        rng_idx = sorted(
            ((int(m.group(1)), i) for i, n in enumerate(names)
             if (m := re.match(r"px_lag_rng_(\d+)$", n))),
        )
        return [i for _, i in ret_idx], [i for _, i in rng_idx]

    def _to_tensor(self, X: np.ndarray) -> np.ndarray:
        """(n, T, C): channel 0 = returns (reversed so oldest first), channel 1 = ranges."""
        ret = X[:, self._ret_cols][:, ::-1]
        chans = [ret]
        if self._rng_cols:
            rng = np.zeros_like(ret)
            r = X[:, self._rng_cols][:, ::-1]
            rng[:, -r.shape[1]:] = r
            chans.append(rng)
        t = np.stack(chans, axis=-1)
        return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---------------------------------------------------------------- api
    def fit(self, X, y, feature_names: list[str] | None = None):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch not available for sequence models") from exc

        names = feature_names or self.feature_names
        if names is None:
            raise ValueError("SequenceClassifier needs feature_names")
        self._ret_cols, self._rng_cols = self._seq_layout(list(names))
        if len(self._ret_cols) < 5:
            raise ValueError("not enough px_lag_ret columns for a sequence model")

        torch.manual_seed(self.seed)
        Xt = torch.from_numpy(self._to_tensor(np.asarray(X, dtype=float)))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        n, T, C = Xt.shape

        if self.arch == "lstm":
            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    s.rnn = nn.LSTM(C, self.hidden, num_layers=self.layers,
                                    batch_first=True, dropout=0.0)
                    s.head = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.hidden, 1))

                def forward(s, x):
                    out, _ = s.rnn(x)
                    return s.head(out[:, -1]).squeeze(-1)
        else:  # tcn: stack of dilated causal convs
            class Net(nn.Module):
                def __init__(s):
                    super().__init__()
                    layers_ = []
                    ch_in = C
                    for lvl in range(self.levels):
                        layers_ += [
                            nn.Conv1d(ch_in, self.channels, self.kernel,
                                      dilation=2**lvl, padding=(self.kernel - 1) * 2**lvl),
                            nn.ReLU(), nn.Dropout(self.dropout),
                        ]
                        ch_in = self.channels
                    s.conv = nn.Sequential(*layers_)
                    s.head = nn.Linear(self.channels, 1)

                def forward(s, x):
                    z = s.conv(x.transpose(1, 2))[:, :, : x.shape[1]]
                    return s.head(z[:, :, -1]).squeeze(-1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = Net().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        lossf = nn.BCEWithLogitsLoss()
        ds = torch.utils.data.TensorDataset(Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=True)
        net.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss = lossf(net(xb.to(device)), yb.to(device))
                loss.backward()
                opt.step()
        self._net = net.eval()
        self._device = device
        return self

    def predict_proba(self, X):
        import torch
        Xt = torch.from_numpy(self._to_tensor(np.asarray(X, dtype=float))).to(self._device)
        with torch.no_grad():
            p = torch.sigmoid(self._net(Xt)).cpu().numpy()
        return np.column_stack([1 - p, p])
