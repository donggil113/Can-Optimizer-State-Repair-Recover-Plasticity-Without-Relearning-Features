"""Deterministic, cloneable random streams and bit-exact hashing."""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from typing import Any


def derive_seed(*parts: Any) -> int:
    """Derive a 64-bit seed from a tuple of ints/strings.

    Uses sha256 of a canonical JSON encoding, so it does not depend on
    PYTHONHASHSEED (unlike the builtin ``hash``).
    """
    payload = json.dumps(list(parts), separators=(",", ":"), sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


class RNG:
    """A named Mersenne Twister stream whose full state can be captured.

    Every source of randomness in training (data sampling, dropout masks,
    initialisation) owns one of these, and nothing touches the global
    ``random`` module, so a checkpoint that captures all streams captures all
    randomness.
    """

    def __init__(self, seed: int):
        self._r = random.Random(seed)

    def get_state(self) -> tuple:
        return self._r.getstate()

    def set_state(self, state: tuple) -> None:
        self._r.setstate(state)

    def random(self) -> float:
        return self._r.random()

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._r.gauss(mu, sigma)

    def uniform(self, a: float, b: float) -> float:
        return self._r.uniform(a, b)

    def randrange(self, n: int) -> int:
        return self._r.randrange(n)

    def shuffle(self, xs: list) -> None:
        self._r.shuffle(xs)


def _feed(h: "hashlib._Hash", obj: Any) -> None:
    """Feed ``obj`` into ``h`` so that bitwise-equal objects hash equally.

    Floats are hashed by their IEEE-754 bytes, so 0.0 and -0.0 differ and any
    last-bit difference changes the hash.
    """
    if isinstance(obj, bool):
        h.update(b"B" + (b"1" if obj else b"0"))
    elif isinstance(obj, int):
        h.update(b"I" + str(obj).encode() + b";")
    elif isinstance(obj, float):
        h.update(b"F" + struct.pack("<d", obj))
    elif isinstance(obj, str):
        h.update(b"S" + str(len(obj)).encode() + b":" + obj.encode())
    elif obj is None:
        h.update(b"N")
    elif isinstance(obj, (list, tuple)):
        h.update(b"L" + str(len(obj)).encode() + b"[")
        for x in obj:
            _feed(h, x)
        h.update(b"]")
    elif isinstance(obj, dict):
        h.update(b"D" + str(len(obj)).encode() + b"{")
        for k in sorted(obj):
            _feed(h, str(k))
            _feed(h, obj[k])
        h.update(b"}")
    else:
        raise TypeError(f"cannot hash object of type {type(obj).__name__}")


def bit_hash(obj: Any) -> str:
    """sha256 hex digest of a nested structure of builtin scalars/containers."""
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


def bitwise_equal(a: Any, b: Any) -> bool:
    """Structural equality that distinguishes 0.0 from -0.0 and treats NaN==NaN."""
    if isinstance(a, float) and isinstance(b, float):
        return struct.pack("<d", a) == struct.pack("<d", b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(bitwise_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(bitwise_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def is_finite(xs: list[float]) -> bool:
    return all(math.isfinite(x) for x in xs)
