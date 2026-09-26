"""A small MLP with BatchNorm and dropout, with a hand-written backward pass.

Architecture: Linear(D->H, no bias when normalised) -> BatchNorm1d(H) -> ReLU
-> Dropout(p) -> Linear(H->C). Loss: mean softmax cross-entropy.

BatchNorm follows torch.nn.BatchNorm1d conventions: training mode normalises
with the biased batch variance and updates running statistics with the
unbiased batch variance, ``running = (1 - momentum) * running + momentum * x``,
and increments ``num_batches_tracked``. Dropout is inverted dropout (scale
1/(1-p) at train time, identity at eval time).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .rng import RNG


@dataclass
class Param:
    data: list[float]
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return len(self.data)


@dataclass
class ForwardCache:
    x: list[list[float]]
    z1: list[list[float]]
    xhat: list[list[float]]
    invstd: list[float]
    y: list[list[float]]
    mask: list[list[float]] | None
    h: list[list[float]]
    logits: list[list[float]]
    extras: dict = field(default_factory=dict)


class MLP:
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float = 0.0,
        norm: str = "batchnorm",
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
        init_rng: RNG | None = None,
    ):
        if norm not in ("batchnorm", "none"):
            raise ValueError(f"unknown norm {norm!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.dropout = dropout
        self.norm = norm
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.params: dict[str, Param] = {}
        self.buffers: dict[str, list] = {}
        rng = init_rng if init_rng is not None else RNG(0)
        # torch.nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        b1 = 1.0 / math.sqrt(in_dim)
        self.params["fc1.weight"] = Param([rng.uniform(-b1, b1) for _ in range(hidden * in_dim)], (hidden, in_dim))
        if norm == "none":
            self.params["fc1.bias"] = Param([rng.uniform(-b1, b1) for _ in range(hidden)], (hidden,))
        else:
            self.params["bn.weight"] = Param([1.0] * hidden, (hidden,))
            self.params["bn.bias"] = Param([0.0] * hidden, (hidden,))
            self.buffers["bn.running_mean"] = [0.0] * hidden
            self.buffers["bn.running_var"] = [1.0] * hidden
            self.buffers["bn.num_batches_tracked"] = [0]
        b2 = 1.0 / math.sqrt(hidden)
        self.params["fc2.weight"] = Param([rng.uniform(-b2, b2) for _ in range(out_dim * hidden)], (out_dim, hidden))
        self.params["fc2.bias"] = Param([rng.uniform(-b2, b2) for _ in range(out_dim)], (out_dim,))

    # ------------------------------------------------------------------ utils
    def config(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "out_dim": self.out_dim,
            "dropout": self.dropout,
            "norm": self.norm,
            "bn_momentum": self.bn_momentum,
            "bn_eps": self.bn_eps,
        }

    def _rows(self, name: str) -> list[list[float]]:
        p = self.params[name]
        n_out, n_in = p.shape
        d = p.data
        return [d[i * n_in:(i + 1) * n_in] for i in range(n_out)]

    def draw_dropout_mask(self, batch: int, rng: RNG) -> list[list[float]] | None:
        """Draw an inverted-dropout mask; consumes exactly batch*hidden uniforms."""
        if self.dropout == 0.0:
            return None
        keep_scale = 1.0 / (1.0 - self.dropout)
        p = self.dropout
        return [[(keep_scale if rng.random() >= p else 0.0) for _ in range(self.hidden)] for _ in range(batch)]

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: list[list[float]],
        train: bool,
        dropout_rng: RNG | None = None,
        mask: list[list[float]] | None = None,
        update_buffers: bool = True,
    ) -> ForwardCache:
        """Forward pass.

        In train mode, the dropout mask is taken from ``mask`` if given, else
        drawn from ``dropout_rng``; BatchNorm uses batch statistics and (if
        ``update_buffers``) updates the running buffers in place. Eval mode is
        a pure function of parameters and buffers.
        """
        B = len(x)
        W1 = self._rows("fc1.weight")
        z1 = [[sum(w * xi for w, xi in zip(row, xb)) for row in W1] for xb in x]
        H = self.hidden
        if self.norm == "none":
            b = self.params["fc1.bias"].data
            y = [[z + bi for z, bi in zip(zb, b)] for zb in z1]
            xhat, invstd = y, [1.0] * H
        else:
            gamma = self.params["bn.weight"].data
            beta = self.params["bn.bias"].data
            if train:
                if B < 2:
                    raise ValueError("BatchNorm in train mode needs batch size >= 2")
                mean = [sum(z1[b][i] for b in range(B)) / B for i in range(H)]
                var = [sum((z1[b][i] - mean[i]) ** 2 for b in range(B)) / B for i in range(H)]
                if update_buffers:
                    mom = self.bn_momentum
                    rm = self.buffers["bn.running_mean"]
                    rv = self.buffers["bn.running_var"]
                    unbiased = B / (B - 1)
                    for i in range(H):
                        rm[i] = (1.0 - mom) * rm[i] + mom * mean[i]
                        rv[i] = (1.0 - mom) * rv[i] + mom * var[i] * unbiased
                    self.buffers["bn.num_batches_tracked"][0] += 1
            else:
                mean = self.buffers["bn.running_mean"]
                var = self.buffers["bn.running_var"]
            invstd = [1.0 / math.sqrt(v + self.bn_eps) for v in var]
            xhat = [[(zb[i] - mean[i]) * invstd[i] for i in range(H)] for zb in z1]
            y = [[gamma[i] * xb[i] + beta[i] for i in range(H)] for xb in xhat]
        a = [[v if v > 0.0 else 0.0 for v in yb] for yb in y]
        if train and self.dropout > 0.0:
            if mask is None:
                if dropout_rng is None:
                    raise ValueError("train-mode dropout needs a mask or dropout_rng")
                mask = self.draw_dropout_mask(B, dropout_rng)
            h = [[v * m for v, m in zip(ab, mb)] for ab, mb in zip(a, mask)]
        else:
            mask = None
            h = a
        W2 = self._rows("fc2.weight")
        b2 = self.params["fc2.bias"].data
        logits = [[sum(w * hi for w, hi in zip(row, hb)) + bc for row, bc in zip(W2, b2)] for hb in h]
        return ForwardCache(x=x, z1=z1, xhat=xhat, invstd=invstd, y=y, mask=mask, h=h, logits=logits,
                            extras={"relu": a, "train": train})

    def features(self, x: list[list[float]]) -> list[list[float]]:
        """Eval-mode post-ReLU hidden features (the representation we track)."""
        return self.forward(x, train=False).extras["relu"]

    # ------------------------------------------------------------- loss/grad
    @staticmethod
    def cross_entropy(logits: list[list[float]], y: list[int]) -> tuple[float, list[list[float]], float]:
        """Mean CE loss, its gradient w.r.t. logits, and accuracy."""
        B = len(logits)
        loss = 0.0
        correct = 0
        dlogits = []
        for lb, yb in zip(logits, y):
            mx = max(lb)
            ex = [math.exp(v - mx) for v in lb]
            s = sum(ex)
            loss += -(lb[yb] - mx - math.log(s))
            p = [e / s for e in ex]
            p[yb] -= 1.0
            dlogits.append([g / B for g in p])
            if max(range(len(lb)), key=lb.__getitem__) == yb:
                correct += 1
        return loss / B, dlogits, correct / B

    def backward(self, cache: ForwardCache, dlogits: list[list[float]]) -> dict[str, list[float]]:
        """Gradients of the loss w.r.t. every parameter (flat, row-major)."""
        B = len(dlogits)
        H, D, C = self.hidden, self.in_dim, self.out_dim
        h = cache.h
        grads: dict[str, list[float]] = {}
        gW2 = [0.0] * (C * H)
        for c in range(C):
            base = c * H
            for b in range(B):
                g = dlogits[b][c]
                if g != 0.0:
                    hb = h[b]
                    for i in range(H):
                        gW2[base + i] += g * hb[i]
        grads["fc2.weight"] = gW2
        grads["fc2.bias"] = [sum(dlogits[b][c] for b in range(B)) for c in range(C)]
        W2 = self._rows("fc2.weight")
        dh = [[sum(dl[c] * W2[c][i] for c in range(C)) for i in range(H)] for dl in dlogits]
        if cache.mask is not None:
            dh = [[g * m for g, m in zip(gb, mb)] for gb, mb in zip(dh, cache.mask)]
        dy = [[g if yv > 0.0 else 0.0 for g, yv in zip(gb, yb)] for gb, yb in zip(dh, cache.y)]
        if self.norm == "none":
            grads["fc1.bias"] = [sum(dy[b][i] for b in range(B)) for i in range(H)]
            dz = dy
        else:
            gamma = self.params["bn.weight"].data
            xhat = cache.xhat
            grads["bn.weight"] = [sum(dy[b][i] * xhat[b][i] for b in range(B)) for i in range(H)]
            grads["bn.bias"] = [sum(dy[b][i] for b in range(B)) for i in range(H)]
            dxhat = [[dy[b][i] * gamma[i] for i in range(H)] for b in range(B)]
            if cache.extras["train"]:
                s1 = [sum(dxhat[b][i] for b in range(B)) for i in range(H)]
                s2 = [sum(dxhat[b][i] * xhat[b][i] for b in range(B)) for i in range(H)]
                dz = [[cache.invstd[i] / B * (B * dxhat[b][i] - s1[i] - xhat[b][i] * s2[i]) for i in range(H)]
                      for b in range(B)]
            else:
                dz = [[dxhat[b][i] * cache.invstd[i] for i in range(H)] for b in range(B)]
        gW1 = [0.0] * (H * D)
        x = cache.x
        for b in range(B):
            xb = x[b]
            dzb = dz[b]
            for i in range(H):
                g = dzb[i]
                if g != 0.0:
                    base = i * D
                    for j in range(D):
                        gW1[base + j] += g * xb[j]
        grads["fc1.weight"] = gW1
        return grads

    # ---------------------------------------------------------- state access
    def state(self) -> dict:
        """Plain-data deep copy of parameters and buffers."""
        return {
            "params": {k: list(p.data) for k, p in self.params.items()},
            "buffers": {k: list(v) for k, v in self.buffers.items()},
        }

    def load_state(self, state: dict) -> None:
        if state["params"].keys() != self.params.keys() or state["buffers"].keys() != self.buffers.keys():
            raise KeyError("state keys do not match model")
        for k, v in state["params"].items():
            if len(v) != self.params[k].numel:
                raise ValueError(f"size mismatch for {k}")
            self.params[k].data = list(v)
        for k, v in state["buffers"].items():
            self.buffers[k] = list(v)
