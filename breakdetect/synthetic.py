"""Small seeded generator of synthetic series with and without a structural break.

Each series has a break-free history and an online segment. When a break is
present it starts at a uniformly drawn online index and changes one property
of the process: the mean, the variance, the serial dependence or the tails.
Values are rounded to float32, the precision of the competition data, and
some series are snapped to a decimal price grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BREAK_KINDS = ("mean", "variance", "dependence", "tail")


@dataclass(frozen=True)
class SyntheticSeries:
    """One series: float64 arrays holding float32 values and the break description."""

    history: np.ndarray
    online: np.ndarray
    break_index: int | None
    kind: str


def _innovations(rng: np.random.Generator, n: int, df: float | None) -> np.ndarray:
    """Unit-variance Gaussian or Student-t innovations."""
    if df is None:
        return rng.standard_normal(n)
    return rng.standard_t(df, n) / np.sqrt(df / (df - 2.0))


def _simulate(
    rng: np.random.Generator,
    n_history: int,
    n_online: int,
    kind: str,
    break_index: int | None,
) -> np.ndarray:
    """Simulate an AR(1)-GARCH(1,1) path whose parameters may change at the break."""
    n = n_history + n_online
    split = n if break_index is None else n_history + break_index
    phi = rng.uniform(-0.5, 0.8)
    df = None if rng.random() < 0.5 else float(rng.choice([3.0, 5.0, 10.0]))
    garch = rng.random() < 0.35
    alpha, beta = (rng.uniform(0.03, 0.15), 0.0) if garch else (0.0, 0.0)
    if garch:
        beta = rng.uniform(0.75, 0.95 - alpha)

    shift, scale, phi_after, df_after = 0.0, 1.0, phi, df
    if kind == "mean":
        shift = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 2.0)
    elif kind == "variance":
        ratio = rng.uniform(1.5, 4.0)
        scale = np.sqrt(ratio if rng.random() < 0.5 else 1.0 / ratio)
    elif kind == "dependence":
        delta = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.8)
        phi_after = float(np.clip(phi + delta, -0.9, 0.9))
        if abs(phi_after - phi) < 0.3:
            phi_after = float(np.clip(phi - delta, -0.9, 0.9))
    elif kind == "tail":
        df_after = 3.0 if df is None else None

    burn = 300
    eps_before = _innovations(rng, burn + split, df)
    eps_after = _innovations(rng, n - split, df_after)
    eps = np.concatenate([eps_before, eps_after])
    x = np.empty(burn + n)
    h = 1.0
    prev = 0.0
    prev_e = 0.0
    for i in range(burn + n):
        after = i >= burn + split
        if garch:
            h = (1.0 - alpha - beta) + alpha * prev_e * prev_e + beta * h
        e = np.sqrt(h) * eps[i]
        a = phi_after if after else phi
        value = a * prev + e
        prev, prev_e = value, e
        x[i] = value * (scale if after else 1.0) + (shift if after else 0.0)
    return x[burn:]


def generate(
    n_series: int = 200,
    seed: int = 0,
    history_range: tuple[int, int] = (1000, 5000),
    online_range: tuple[int, int] = (10, 999),
    break_fraction: float = 0.8,
    grid_fraction: float = 0.1,
) -> list[SyntheticSeries]:
    """Generate ``n_series`` reproducible series (same seed, same output)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_series):
        n_history = int(rng.integers(history_range[0], history_range[1] + 1))
        n_online = int(rng.integers(online_range[0], online_range[1] + 1))
        if rng.random() < break_fraction:
            kind = str(rng.choice(BREAK_KINDS))
            break_index = int(rng.integers(0, n_online))
        else:
            kind, break_index = "none", None
        x = _simulate(rng, n_history, n_online, kind, break_index)
        level = rng.uniform(-50.0, 50.0)
        spread = 10.0 ** rng.uniform(-2.0, 1.0)
        x = level + spread * x
        if rng.random() < grid_fraction:
            tick = spread * float(rng.choice([0.02, 0.1, 0.5]))
            x = np.round(x / tick) * tick
        x = x.astype(np.float32).astype(np.float64)
        out.append(SyntheticSeries(x[:n_history].copy(), x[n_history:].copy(), break_index, kind))
    return out
