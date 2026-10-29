"""Per-step CDF maps: turn a member's raw score at online step t into a probability.

Each map was fitted on the members' training predictions, one empirical CDF per online
step t (t is 0-based and clamped to 999). Three storage formats exist; ``load_map``
picks the class from the keys of the ``.npz`` file.
"""

from __future__ import annotations

import operator
from bisect import bisect_right

import numpy as np

T_MAX = 1000


class SampleMap:
    """CDF built on demand from the raw training sample of each step (``values``/``offsets``)."""

    __slots__ = ("_aux", "offsets", "values")

    def __init__(self, values: np.ndarray, offsets: np.ndarray) -> None:
        self.values = values
        self.offsets = offsets
        self._aux: list = [None] * T_MAX

    def sample(self, t: int) -> np.ndarray:
        """Training sample of step ``t``."""
        t = min(max(int(t), 0), T_MAX - 1)
        return self.values[int(self.offsets[t]) : int(self.offsets[t + 1])]

    def _build(self, t: int):
        """Knots (unique values, mid-rank quantiles) and exponential tail slopes of step ``t``."""
        seg = self.sample(t)
        kv, counts = np.unique(seg, return_counts=True)
        starts = np.concatenate(([0], np.cumsum(counts[:-1])))
        kq = (starts + counts / 2.0) / seg.size
        if kv.size >= 2:
            m_l = float((kq[1] - kq[0]) / (kv[1] - kv[0]))
            m_r = float((kq[-1] - kq[-2]) / (kv[-1] - kv[-2]))
            r_hi = float(1.0 - kq[-1])
            tails = (float(kv[0]), float(kv[-1]), float(kq[0]), float(kq[-1]), m_l, m_r, r_hi)
        else:
            tails = None
        if kv.size == seg.size:
            kv = seg
        aux = (memoryview(kv), memoryview(kq), tails, int(kv.size))
        self._aux[t] = aux
        return aux

    def prewarm(self) -> None:
        """Build the knots of every step once (before forking worker processes)."""
        for t in range(T_MAX):
            if self._aux[t] is None:
                self._build(t)

    def evaluate(self, t: int, s) -> float:
        """Linear interpolation between knots, exponential tails outside the sample range."""
        t = min(max(int(t), 0), T_MAX - 1)
        kv, kq, tails, m = self._aux[t] or self._build(t)
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


class AnchorTimeMap:
    """CDF anchors at a few steps (``times``/``offsets``/``x``/``u``), linear in t between them."""

    def __init__(self, payload) -> None:
        self.times = np.array(payload["times"], copy=True)
        self.offsets = np.array(payload["offsets"], copy=True)
        self.values = np.array(payload["x"], copy=True)
        self.u = np.array(payload["u"], copy=True)
        t, off, x, u = self.times, self.offsets, self.values, self.u
        if (
            t.dtype != np.int64
            or off.dtype != np.int64
            or x.dtype != np.float64
            or u.dtype != np.float64
            or any(a.ndim != 1 for a in (t, off, x, u))
            or not t.size
            or off.shape != (t.size + 1,)
            or off[0] != 0
            or off[-1] != x.size
            or x.shape != u.shape
            or np.any(np.diff(off) <= 0)
            or np.any(np.diff(t) <= 0)
            or not np.isfinite(x).all()
            or not np.isfinite(u).all()
            or np.any(u < 0)
            or np.any(u > 1)
        ):
            raise ValueError("Invalid map artifact.")
        self.anchors = [(x[a:b], u[a:b]) for a, b in zip(off[:-1], off[1:], strict=True)]
        for xx, uu in self.anchors:
            if np.any(np.diff(xx) <= 0) or np.any(np.diff(uu) < 0):
                raise ValueError("Invalid map anchors.")
        for a in (t, off, x, u):
            a.flags.writeable = False

    def prewarm(self) -> None:
        """Nothing to precompute."""

    def evaluate(self, t, raw) -> float:
        """Interpolate the score in the two anchors around ``t``, then linearly in ``t``."""
        t = operator.index(t)
        if not np.isfinite(raw):
            raise ValueError("Invalid map input.")
        hi = min(np.searchsorted(self.times, t), len(self.times) - 1)
        lo = max(hi - 1, 0)
        if self.times[hi] == t:
            lo = hi
        a = 0.0 if hi == lo else (float(t) - self.times[lo]) / (self.times[hi] - self.times[lo])
        a = min(max(a, 0.0), 1.0)
        values = np.asarray([raw], dtype=np.float64)
        left, right = np.interp(values, *self.anchors[lo]), np.interp(values, *self.anchors[hi])
        return float(((1 - a) * left + a * right)[0])


_BAND_KEYS = {"format", "values", "quantiles", "offsets", "midpoints", "alpha"}
_BAND_FORMAT = "anchor_cdf_v1"


def _validate_band(x, u) -> None:
    """A band holds sorted values with the frozen mid-point quantiles (i + 0.5) / n."""
    if (
        x.dtype != np.float64
        or u.dtype != np.float64
        or x.ndim != 1
        or x.shape != u.shape
        or not len(x)
        or not np.isfinite(x).all()
        or np.any(np.diff(x) < 0)
        or not np.array_equal(u, (np.arange(len(x)) + 0.5) / len(x))
    ):
        raise ValueError("invalid map band / frozen quantiles")


class BandMap:
    """Two quantile bands per step mixed with a per-step weight ``alpha``."""

    def __init__(self, payload) -> None:
        if (
            set(payload) != _BAND_KEYS
            or np.asarray(payload["format"]).shape != ()
            or str(payload["format"]) != _BAND_FORMAT
        ):
            raise ValueError("map format/keys")
        for key in _BAND_KEYS - {"format"}:
            a = np.asarray(payload[key])
            dtype = np.int64 if key in ("offsets", "midpoints") else np.float64
            if a.dtype != dtype or a.ndim != 1 or not np.isfinite(a).all():
                raise ValueError(f"map dtype/shape/finite: {key}")
            setattr(self, key, np.array(a, copy=True, order="C"))
        if (
            self.offsets.shape != (T_MAX + 1,)
            or self.midpoints.shape != (T_MAX,)
            or self.alpha.shape != (T_MAX,)
            or self.values.shape != self.quantiles.shape
            or self.offsets[0] != 0
            or self.offsets[-1] != self.values.size
            or np.any(np.diff(self.offsets) <= 1)
            or np.any(self.midpoints <= self.offsets[:-1])
            or np.any(self.midpoints >= self.offsets[1:])
            or np.any(self.alpha < 0)
            or np.any(self.alpha > 1)
            or self.alpha[-1] not in (0.0, 1.0)
        ):
            raise ValueError("map offsets/midpoints/alpha")
        self._bands = []
        for t in range(T_MAX):
            start, mid, end = self.offsets[t], self.midpoints[t], self.offsets[t + 1]
            bands = (
                self.values[start:mid],
                self.quantiles[start:mid],
                self.values[mid:end],
                self.quantiles[mid:end],
            )
            _validate_band(*bands[:2])
            _validate_band(*bands[2:])
            self._bands.append(bands)
        for key in _BAND_KEYS - {"format"}:
            getattr(self, key).flags.writeable = False

    def prewarm(self) -> None:
        """Nothing to precompute."""

    def evaluate(self, t, s) -> float:
        """Mix the two band CDFs of step ``t``."""
        t = operator.index(t)
        if t < 0:
            raise ValueError("negative causal step")
        if not np.isfinite(s):
            raise ValueError("nonfinite map score")
        t = min(t, T_MAX - 1)
        xl, ul, xr, ur = self._bands[t]
        a = self.alpha[t]
        left = np.interp(float(s), xl, ul)
        right = np.interp(float(s), xr, ur)
        return float((1 - a) * left + a * right)


def load_map(path):
    """Load a per-step map, validate it and precompute its knots."""
    with np.load(path, allow_pickle=False) as d:
        if {"times", "offsets", "x", "u"} <= set(d.files):
            step_map = AnchorTimeMap({k: d[k] for k in d.files})
        elif "format" in d.files:
            step_map = BandMap({k: d[k] for k in d.files})
        else:
            step_map = SampleMap(values=d["values"].copy(), offsets=d["offsets"].copy())
            off = step_map.offsets
            if (
                off.shape != (T_MAX + 1,)
                or int(off[0]) != 0
                or int(off[-1]) != step_map.values.shape[0]
                or not bool(np.all(np.diff(off) > 0))
                or not bool(np.isfinite(step_map.values).all())
            ):
                raise ValueError(f"map {path} is malformed")
    step_map.prewarm()
    return step_map
