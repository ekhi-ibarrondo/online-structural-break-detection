"""Per-step monotone CDF maps ("step maps").

A member's raw score at online step t is mapped to (0, 1) with the empirical
CDF of the training scores observed at the same step.  The stored artifact per
step t is the sorted training sample ``S_t`` (concatenated, with offsets).
Sparse steps borrow rows from a symmetric band of neighbouring steps until the
band holds at least ``N_MIN = 500`` scores; steps t >= 1000 reuse ``S_999``.

Evaluation is a strictly monotone interpolant with exponential tails:

- Knots: for each distinct sample value v_j, the position q_j is the mean of
  the mid-rank positions ``(i + 0.5) / n`` of its duplicates.
- Interior: linear interpolation between adjacent knots.
- Lower tail (s < v_first):  p = q_first * exp(-m_L * (v_first - s) / q_first),
  with m_L the slope of the first interior segment.
- Upper tail (s > v_last):   p = 1 - (1 - q_last) * exp(-m_R * (s - v_last) / (1 - q_last)),
  with m_R the slope of the last interior segment.
- The map is C^1 at both borders, strictly monotone and stays in (0, 1) up to
  float64 saturation far outside the training support.
- A band with a single distinct value maps to the constant 0.5.

The maps are label-free: they only use training scores and their steps.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "N_MIN",
    "T_MAX",
    "StepMaps",
    "evaluate_rows",
    "fit_step_maps",
    "load_step_maps",
    "save_step_maps",
]

T_MAX = 1000
N_MIN = 500


@dataclass(frozen=True)
class StepMaps:
    """Concatenated sorted per-step samples + offsets (length T_MAX + 1).

    The artifact is the sorted samples; the interpolation knots are derived
    in memory (cached per t) and are a pure function of the sample.
    """

    values: np.ndarray  # float64, concatenation of the 1000 sorted band samples
    offsets: np.ndarray  # int64, shape (T_MAX + 1,)
    _knots_cache: dict = field(default_factory=dict, compare=False, repr=False)

    def sample(self, t: int) -> np.ndarray:
        t = min(max(int(t), 0), T_MAX - 1)
        return self.values[int(self.offsets[t]) : int(self.offsets[t + 1])]

    def knots(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(kv, kq): distinct sample values and their mean mid-rank positions.

        q_j = mean over the duplicates of value v_j of ``(i + 0.5) / n`` =
        ``(start_j + count_j / 2) / n``.
        """
        t = min(max(int(t), 0), T_MAX - 1)
        cached = self._knots_cache.get(t)
        if cached is not None:
            return cached
        seg = self.sample(t)
        kv, counts = np.unique(seg, return_counts=True)
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))
        kq = (starts + counts / 2.0) / seg.size
        self._knots_cache[t] = (kv, kq)
        # Python-list mirrors + tail parameters for the scalar path (identical
        # IEEE values; the tail parameters are computed once and shared by the
        # scalar and vectorised branches so both agree bit for bit).
        if kv.size >= 2:
            m_l = float((kq[1] - kq[0]) / (kv[1] - kv[0]))
            m_r = float((kq[-1] - kq[-2]) / (kv[-1] - kv[-2]))
            r_hi = float(1.0 - kq[-1])
            tails = (float(kv[0]), float(kv[-1]), float(kq[0]), float(kq[-1]), m_l, m_r, r_hi)
        else:
            tails = None
        self._knots_cache[(t, "aux")] = (kv.tolist(), kq.tolist(), tails)
        return kv, kq

    def _aux(self, t: int):
        t = min(max(int(t), 0), T_MAX - 1)
        cached = self._knots_cache.get((t, "aux"))
        if cached is None:
            self.knots(t)
            cached = self._knots_cache[(t, "aux")]
        return cached

    def prewarm(self) -> None:
        """Materialise all 1000 knot arrays."""
        for t in range(T_MAX):
            self.knots(t)

    def evaluate(self, t: int, s: float | np.ndarray) -> float | np.ndarray:
        """Strictly monotone CDF value(s) at step ``t``.

        The scalar and vectorised branches use the same operation order and
        the same ``np.exp`` ufunc, so they are bit-identical.
        """
        scalar = np.isscalar(s) or np.ndim(s) == 0
        if scalar:
            kv, kq, tails = self._aux(t)
            m = len(kv)
            if m == 1:
                return 0.5
            kv0, kv_last, kq0, _kq_last, m_l, m_r, r_hi = tails
            sf = float(s)
            if sf < kv0:
                return float(kq0 * np.exp(-(m_l * (kv0 - sf) / kq0)))
            if sf > kv_last:
                return float(1.0 - r_hi * np.exp(-(m_r * (sf - kv_last) / r_hi)))
            j = bisect_right(kv, sf) - 1
            j = 0 if j < 0 else (m - 2 if j > m - 2 else j)
            return kq[j] + (sf - kv[j]) * (kq[j + 1] - kq[j]) / (kv[j + 1] - kv[j])
        kv, kq = self.knots(t)
        m = kv.size
        if m == 1:
            return np.full(np.shape(s), 0.5, dtype=np.float64)
        _, _, tails = self._aux(t)
        kv0, kv_last, kq0, _kq_last, m_l, m_r, r_hi = tails
        s_arr = np.asarray(s, dtype=np.float64)
        j = np.clip(np.searchsorted(kv, s_arr, side="right") - 1, 0, m - 2)
        p = kq[j] + (s_arr - kv[j]) * (kq[j + 1] - kq[j]) / (kv[j + 1] - kv[j])
        low = s_arr < kv0
        if np.any(low):
            p[low] = kq0 * np.exp(-(m_l * (kv0 - s_arr[low]) / kq0))
        high = s_arr > kv_last
        if np.any(high):
            p[high] = 1.0 - r_hi * np.exp(-(m_r * (s_arr[high] - kv_last) / r_hi))
        return p


def fit_step_maps(scores: np.ndarray, times: np.ndarray, n_min: int = N_MIN) -> StepMaps:
    """Fit the per-step maps over a training population.

    ``scores``: float64 scores; ``times``: int online step (0-indexed) per row.
    Deterministic: the band expansion depends only on per-step counts and the
    stored samples are plain value sorts.
    """
    scores = np.asarray(scores, dtype=np.float64)
    times = np.asarray(times)
    if scores.shape != times.shape:
        raise ValueError("scores and times must have the same shape")
    if scores.size == 0:
        raise ValueError("cannot fit maps on an empty population")
    if int(times.min()) < 0 or int(times.max()) >= T_MAX:
        raise ValueError(f"times must lie in [0, {T_MAX - 1}]")

    counts = np.bincount(times, minlength=T_MAX)
    bounds = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    order = np.argsort(times, kind="stable")
    by_t = scores[order]

    total = int(scores.size)
    chunks: list[np.ndarray] = []
    offsets = np.zeros(T_MAX + 1, dtype=np.int64)
    for t in range(T_MAX):
        lo = t
        hi = t
        h = 0
        while int(bounds[hi + 1] - bounds[lo]) < n_min and (lo > 0 or hi < T_MAX - 1):
            h += 1
            lo = max(t - h, 0)
            hi = min(t + h, T_MAX - 1)
        band = np.sort(by_t[int(bounds[lo]) : int(bounds[hi + 1])])
        if band.size == 0:
            raise RuntimeError(f"empty band at t={t} with population {total}")
        chunks.append(band)
        offsets[t + 1] = offsets[t] + band.size
    return StepMaps(values=np.concatenate(chunks), offsets=offsets)


def evaluate_rows(maps: StepMaps, scores: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Vectorised ``F_t(score)`` for row-aligned (score, t) pairs."""
    scores = np.asarray(scores, dtype=np.float64)
    times = np.asarray(times)
    out = np.empty(scores.shape[0], dtype=np.float64)
    tt = np.clip(times, 0, T_MAX - 1)
    for t in np.unique(tt):
        rows = np.flatnonzero(tt == t)
        out[rows] = maps.evaluate(int(t), scores[rows])
    return out


def save_step_maps(path, maps: StepMaps) -> None:
    np.savez_compressed(path, values=maps.values, offsets=maps.offsets)


def load_step_maps(path) -> StepMaps:
    with np.load(path) as d:
        return StepMaps(values=d["values"].copy(), offsets=d["offsets"].copy())
