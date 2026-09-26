"""Training state, checkpoint branching and state-only verification.

A checkpoint captures everything that determines the future of a run:
model parameters, normalisation buffers, optimizer state and hyperparameters,
every RNG stream (data sampling, dropout) and the stream position. Restoring a
checkpoint twice yields two fully independent states (no shared lists), which
is what makes branches *paired*: absent an intervention they produce
bit-identical trajectories.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable

from .interventions import apply_intervention
from .model import MLP
from .optim import apply_update, make_optimizer
from .rng import RNG, bit_hash, bitwise_equal, derive_seed
from .stream import TaskStream

RNG_STREAMS = ("data", "dropout")

LrSchedule = Callable[[int], float]
UpdateTransform = Callable[[int, dict], dict]


@dataclass
class TrainState:
    model: MLP
    optimizer: object
    rngs: dict[str, RNG]
    step: int = 0
    # Controls attached to a run; not part of the checkpoint.
    lr_schedule: LrSchedule | None = None
    update_transform: UpdateTransform | None = None
    history: list = field(default_factory=list)


def init_state(model_cfg: dict, opt_spec: dict, seed: int) -> TrainState:
    model = MLP(init_rng=RNG(derive_seed("init", seed)), **model_cfg)
    opt = make_optimizer(opt_spec, model.params)
    rngs = {name: RNG(derive_seed(name, seed)) for name in RNG_STREAMS}
    return TrainState(model=model, optimizer=opt, rngs=rngs)


# ------------------------------------------------------------- checkpoints
def snapshot(state: TrainState) -> dict:
    """Deep, plain-data copy of the complete training state."""
    return {
        "model_config": state.model.config(),
        "model": state.model.state(),
        "optimizer": state.optimizer.state_dict(),
        "rng": {k: r.get_state() for k, r in state.rngs.items()},
        "step": state.step,
    }


def restore(ckpt: dict) -> TrainState:
    """Build a new, independent TrainState from a checkpoint."""
    ckpt = copy.deepcopy(ckpt)
    model = MLP(**ckpt["model_config"])
    model.load_state(ckpt["model"])
    osd = ckpt["optimizer"]
    spec = {"kind": osd["kind"], **osd["hparams"]}
    opt = make_optimizer(spec, model.params)
    opt.load_state_dict(osd)
    rngs = {}
    for k, s in ckpt["rng"].items():
        r = RNG(0)
        r.set_state(s)
        rngs[k] = r
    return TrainState(model=model, optimizer=opt, rngs=rngs, step=ckpt["step"])


def branch(ckpt: dict, intervention: dict, reference_opt_sd: dict | None = None) -> tuple[TrainState, dict]:
    """Restore ``ckpt`` and apply an optimizer intervention to the copy."""
    st = restore(ckpt)
    new_sd, record = apply_intervention(st.optimizer.state_dict(), intervention, reference_opt_sd)
    st.optimizer.load_state_dict(new_sd)
    return st, record


def hash_parts(ckpt: dict) -> dict:
    return {
        "params": bit_hash(ckpt["model"]["params"]),
        "buffers": bit_hash(ckpt["model"]["buffers"]),
        "optimizer": bit_hash(ckpt["optimizer"]),
        "rng": bit_hash(ckpt["rng"]),
        "step": ckpt["step"],
    }


# -------------------------------------------------------------- training
def _norm(xs: list[float]) -> float:
    return math.sqrt(sum(v * v for v in xs))


def train_step(state: TrainState, stream: TaskStream) -> dict:
    task = stream.task_at(state.step)
    x, y = stream.sample_batch(task, state.rngs["data"])
    model = state.model
    cache = model.forward(x, train=True, dropout_rng=state.rngs["dropout"])
    loss, dlogits, acc = MLP.cross_entropy(cache.logits, y)
    grads = model.backward(cache, dlogits)
    scale = state.lr_schedule(state.step) if state.lr_schedule is not None else 1.0
    deltas = state.optimizer.compute_update(grads, lr_scale=scale)
    if state.update_transform is not None:
        deltas = state.update_transform(state.step, deltas)
    apply_update(model.params, deltas)
    upd_norms = {k: _norm(d) for k, d in deltas.items()}
    rec = {
        "step": state.step,
        "task": task,
        "train_loss": loss,
        "train_acc": acc,
        "grad_norm": _norm([g for v in grads.values() for g in v]),
        "update_norm": math.sqrt(sum(n * n for n in upd_norms.values())),
        "update_norms": upd_norms,
        "lr_scale": scale,
        "batch_hash": bit_hash([x, y]),
        "dropout_hash": bit_hash(cache.mask) if cache.mask is not None else None,
    }
    state.step += 1
    return rec


def evaluate(model: MLP, stream: TaskStream, tasks: list[int]) -> dict:
    out = {}
    for k in tasks:
        x, y = stream.eval_set(k)
        logits = model.forward(x, train=False).logits
        loss, _, acc = MLP.cross_entropy(logits, y)
        out[k] = {"loss": loss, "acc": acc}
    return out


# ------------------------------------------------------ state-only checks
def verify_state_only(reference: TrainState, candidate: TrainState, probe_x: list[list[float]]) -> dict:
    """Check that ``candidate`` differs from ``reference`` only in optimizer state.

    ``state_only`` is True iff weights, buffers, RNG streams, stream position,
    optimizer hyperparameters and forward outputs are all bitwise equal, so
    only optimizer moments/ages may differ. ``outside_optimizer_unchanged``
    drops the hyperparameter condition (used for declared hparam controls).

    Train-mode forwards are run on restored clones so that neither state's
    BatchNorm buffers nor dropout RNG are advanced by the check.
    """
    ref_ck, cand_ck = snapshot(reference), snapshot(candidate)
    res = {
        "weights_equal": bitwise_equal(ref_ck["model"]["params"], cand_ck["model"]["params"]),
        "buffers_equal": bitwise_equal(ref_ck["model"]["buffers"], cand_ck["model"]["buffers"]),
        "rng_equal": bitwise_equal(ref_ck["rng"], cand_ck["rng"]),
        "step_equal": ref_ck["step"] == cand_ck["step"],
        "optimizer_hparams_equal": bitwise_equal(ref_ck["optimizer"]["hparams"], cand_ck["optimizer"]["hparams"]),
    }
    res["eval_forward_equal"] = bitwise_equal(
        reference.model.forward(probe_x, train=False).logits,
        candidate.model.forward(probe_x, train=False).logits,
    )
    a, b = restore(ref_ck), restore(cand_ck)
    fa = a.model.forward(probe_x, train=True, dropout_rng=a.rngs["dropout"])
    fb = b.model.forward(probe_x, train=True, dropout_rng=b.rngs["dropout"])
    res["train_forward_equal"] = bitwise_equal(fa.logits, fb.logits) and bitwise_equal(fa.mask, fb.mask)
    res["post_forward_buffers_equal"] = bitwise_equal(a.model.buffers, b.model.buffers)
    res["optimizer_state_changed"] = not bitwise_equal(ref_ck["optimizer"]["state"], cand_ck["optimizer"]["state"])
    # Snapshotting/forwarding in this function must not have mutated the inputs.
    res["inputs_unmodified"] = (bitwise_equal(snapshot(reference), ref_ck)
                                and bitwise_equal(snapshot(candidate), cand_ck))
    # Nothing outside the optimizer changed (weights, buffers, RNG, position, outputs).
    res["outside_optimizer_unchanged"] = all(res[k] for k in (
        "weights_equal", "buffers_equal", "rng_equal", "step_equal", "eval_forward_equal",
        "train_forward_equal", "post_forward_buffers_equal", "inputs_unmodified"))
    # Additionally, optimizer hyperparameters are unchanged: only moments/ages may differ.
    res["state_only"] = res["outside_optimizer_unchanged"] and res["optimizer_hparams_equal"]
    return res


# ---------------------------------------------------------- diagnostics
def linear_cka(X: list[list[float]], Y: list[list[float]]) -> float | None:
    """Linear CKA (Kornblith et al., 2019) between two n x d feature matrices."""
    n = len(X)

    def center(M):
        d = len(M[0])
        mu = [sum(M[i][j] for i in range(n)) / n for j in range(d)]
        return [[M[i][j] - mu[j] for j in range(d)] for i in range(n)]

    def gram(M):
        return [[sum(a * b for a, b in zip(M[i], M[j])) for j in range(n)] for i in range(n)]

    Kx, Ky = gram(center(X)), gram(center(Y))
    xy = sum(Kx[i][j] * Ky[i][j] for i in range(n) for j in range(n))
    xx = math.sqrt(sum(v * v for row in Kx for v in row))
    yy = math.sqrt(sum(v * v for row in Ky for v in row))
    if xx == 0.0 or yy == 0.0:
        return None
    return xy / (xx * yy)


def feature_diagnostics(model: MLP, probe_x: list[list[float]], ref_features: list[list[float]] | None,
                        ref_params: dict | None) -> dict:
    feats = model.features(probe_x)
    H = len(feats[0])
    dead = sum(1 for i in range(H) if all(f[i] == 0.0 for f in feats)) / H
    out = {"dead_frac": dead, "w1_norm": _norm(model.params["fc1.weight"].data)}
    if ref_features is not None:
        out["cka_to_branch"] = linear_cka(ref_features, feats)
    if ref_params is not None:
        w0 = ref_params["fc1.weight"]
        w = model.params["fc1.weight"].data
        out["w1_rel_change"] = _norm([a - b for a, b in zip(w, w0)]) / max(_norm(w0), 1e-12)
    return out


def adam_state_diagnostics(state: TrainState, stream: TaskStream) -> dict:
    """Optimizer-state statistics at the current point, computed on a clone.

    Uses the gradient of the *next* training batch, which an online learner
    observes at that step anyway (no oracle information). Returns an empty
    dict for non-Adam optimizers.
    """
    if state.optimizer.kind != "adam":
        return {}
    probe = restore(snapshot(state))
    task = stream.task_at(probe.step)
    x, y = stream.sample_batch(task, probe.rngs["data"])
    cache = probe.model.forward(x, train=True, dropout_rng=probe.rngs["dropout"])
    _, dl, _ = MLP.cross_entropy(cache.logits, y)
    grads = probe.model.backward(cache, dl)
    ratios, cos_num, g_sq, m_sq = [], 0.0, 0.0, 0.0
    for name, g in grads.items():
        mh, vh = probe.optimizer.bias_corrected(name)
        for gi, mi, vi in zip(g, mh, vh):
            if vi > 0.0:
                ratios.append(gi * gi / vi)
            cos_num += gi * mi
            g_sq += gi * gi
            m_sq += mi * mi
    ratios.sort()
    med = ratios[len(ratios) // 2] if ratios else None
    ages = [(st["t_m"], st["t_v"]) for st in probe.optimizer.state.values()]
    return {
        "g2_over_vhat_median": med,
        "g2_over_vhat_p90": ratios[int(0.9 * (len(ratios) - 1))] if ratios else None,
        "cos_mhat_g": cos_num / math.sqrt(g_sq * m_sq) if g_sq > 0 and m_sq > 0 else None,
        "age_m": ages[0][0],
        "age_v": ages[0][1],
    }
