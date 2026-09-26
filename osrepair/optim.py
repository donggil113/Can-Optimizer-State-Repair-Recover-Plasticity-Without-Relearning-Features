"""Adam and SGD(momentum) with explicit, inspectable state.

Adam follows Kingma & Ba (2015), Algorithm 1:

    m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat_t = m_t / (1 - beta1^t)
    v_hat_t = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)

Standard Adam has one step counter ``t`` shared by both moments. Here each
moment carries its own *age* (``t_m`` for m, ``t_v`` for v). In ordinary
training both ages advance together and equal the shared counter, so the
update is exactly Algorithm 1. Ages only diverge after a state intervention;
they let the bias correction of each moment reflect how much gradient mass
that moment actually holds, i.e. ``1 - beta^age`` is the total EMA weight on
observed gradients. Ages may be non-integer after moment mixing.

Note the eps placement: eps is added to sqrt(v_hat) (Algorithm 1 and
torch.optim.Adam), not the "efficient version" with a rescaled eps-hat from
Section 2 of the paper. Parity with torch.optim.Adam is NOT verified in this
environment (torch is not installed).
"""

from __future__ import annotations

import copy
import math

from .model import Param


class Adam:
    kind = "adam"

    def __init__(self, params: dict[str, Param], lr: float, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError("betas must be in [0, 1)")
        self.params = params
        self.hparams = {"lr": float(lr), "beta1": float(b1), "beta2": float(b2), "eps": float(eps)}
        self.state = {
            name: {"m": [0.0] * p.numel, "v": [0.0] * p.numel, "t_m": 0.0, "t_v": 0.0}
            for name, p in params.items()
        }
        self.num_steps = 0

    def compute_update(self, grads: dict[str, list[float]], lr_scale: float = 1.0) -> dict[str, list[float]]:
        """Advance the moments with ``grads`` and return the parameter deltas.

        Deltas are returned rather than applied so that controls (e.g. update
        norm matching) can transform them before :func:`apply_update`.
        """
        lr = self.hparams["lr"] * lr_scale
        b1, b2, eps = self.hparams["beta1"], self.hparams["beta2"], self.hparams["eps"]
        deltas = {}
        for name, g in grads.items():
            st = self.state[name]
            st["t_m"] += 1.0
            st["t_v"] += 1.0
            bc1 = 1.0 - b1 ** st["t_m"]
            bc2 = 1.0 - b2 ** st["t_v"]
            m, v = st["m"], st["v"]
            d = [0.0] * len(g)
            for i, gi in enumerate(g):
                mi = b1 * m[i] + (1.0 - b1) * gi
                vi = b2 * v[i] + (1.0 - b2) * gi * gi
                m[i] = mi
                v[i] = vi
                d[i] = -lr * (mi / bc1) / (math.sqrt(vi / bc2) + eps)
            deltas[name] = d
        self.num_steps += 1
        return deltas

    def bias_corrected(self, name: str) -> tuple[list[float], list[float]]:
        """(m_hat, v_hat) for a parameter using the current ages.

        Undefined (returned as zeros) for a moment whose age is 0, i.e. one that
        has not seen any gradient since initialisation or reset.
        """
        st = self.state[name]
        b1, b2 = self.hparams["beta1"], self.hparams["beta2"]
        bc1 = 1.0 - b1 ** st["t_m"]
        bc2 = 1.0 - b2 ** st["t_v"]
        mh = [x / bc1 for x in st["m"]] if bc1 > 0.0 else [0.0] * len(st["m"])
        vh = [x / bc2 for x in st["v"]] if bc2 > 0.0 else [0.0] * len(st["v"])
        return mh, vh

    def state_dict(self) -> dict:
        return {"kind": self.kind, "hparams": dict(self.hparams), "num_steps": self.num_steps,
                "state": copy.deepcopy(self.state)}

    def load_state_dict(self, sd: dict) -> None:
        if sd["kind"] != self.kind:
            raise ValueError(f"cannot load {sd['kind']} state into {self.kind}")
        if sd["state"].keys() != self.state.keys():
            raise KeyError("optimizer state keys do not match parameters")
        self.hparams = dict(sd["hparams"])
        self.num_steps = sd["num_steps"]
        self.state = copy.deepcopy(sd["state"])


class SGD:
    """SGD with heavy-ball momentum, torch convention (dampening = 0):
    buf = mu * buf + g;  theta -= lr * buf."""

    kind = "sgd"

    def __init__(self, params: dict[str, Param], lr: float, momentum: float = 0.0):
        self.params = params
        self.hparams = {"lr": float(lr), "momentum": float(momentum)}
        self.state = {name: {"buf": [0.0] * p.numel, "has_buf": False} for name, p in params.items()}
        self.num_steps = 0

    def compute_update(self, grads: dict[str, list[float]], lr_scale: float = 1.0) -> dict[str, list[float]]:
        lr = self.hparams["lr"] * lr_scale
        mu = self.hparams["momentum"]
        deltas = {}
        for name, g in grads.items():
            st = self.state[name]
            if mu == 0.0:
                deltas[name] = [-lr * gi for gi in g]
                continue
            if not st["has_buf"]:
                # torch initialises the buffer to the first gradient.
                st["buf"] = list(g)
                st["has_buf"] = True
            else:
                st["buf"] = [mu * b + gi for b, gi in zip(st["buf"], g)]
            deltas[name] = [-lr * b for b in st["buf"]]
        self.num_steps += 1
        return deltas

    def state_dict(self) -> dict:
        return {"kind": self.kind, "hparams": dict(self.hparams), "num_steps": self.num_steps,
                "state": copy.deepcopy(self.state)}

    def load_state_dict(self, sd: dict) -> None:
        if sd["kind"] != self.kind:
            raise ValueError(f"cannot load {sd['kind']} state into {self.kind}")
        self.hparams = dict(sd["hparams"])
        self.num_steps = sd["num_steps"]
        self.state = copy.deepcopy(sd["state"])


def apply_update(params: dict[str, Param], deltas: dict[str, list[float]]) -> None:
    for name, d in deltas.items():
        p = params[name]
        p.data = [w + dw for w, dw in zip(p.data, d)]


def make_optimizer(spec: dict, params: dict[str, Param]):
    kind = spec["kind"]
    if kind == "adam":
        return Adam(params, lr=spec["lr"], betas=(spec.get("beta1", 0.9), spec.get("beta2", 0.999)),
                    eps=spec.get("eps", 1e-8))
    if kind == "sgd":
        return SGD(params, lr=spec["lr"], momentum=spec.get("momentum", 0.0))
    raise ValueError(f"unknown optimizer kind {kind!r}")
