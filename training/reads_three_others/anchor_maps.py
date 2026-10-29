"""Stride-16 anchor CDF map of the reads_three_others scores ("anchor_cdf_v1").

Anchors are the training steps t = 0, 16, 32, ...; each anchor holds the
sorted training scores of its step (widened by +-16 steps until it has at
least 500 rows) with quantiles (i + 0.5) / n; duplicate scores are kept.  A
step between two anchors interpolates linearly between their ``np.interp``
CDFs.  The stored payload has, for each of the 1000 steps, the lower then the
upper anchor (values / quantiles), the ``offsets``/``midpoints`` that delimit
them and the interpolation weight ``alpha``.
"""
from __future__ import annotations

import operator

import numpy as np

FORMAT = "anchor_cdf_v1"
T_MAX = 1000
KEYS = {"format", "values", "quantiles", "offsets", "midpoints", "alpha"}


def cdf_maps(pred, tt):
    """Anchor CDFs of training predictions at stride-16 steps (checks inputs: no infinite loop for n < 500)."""
    pred, tt = np.asarray(pred), np.asarray(tt)
    if (pred.dtype != np.float64 or pred.ndim != 1 or pred.shape != tt.shape
            or tt.dtype.kind not in "iu" or pred.size < 500 or not np.isfinite(pred).all()
            or np.any(tt < 0) or np.any(tt >= T_MAX) or np.any(tt % 16)):
        raise ValueError("map needs >=500 finite float64 train predictions at stride16 steps 0..999")
    times = np.unique(tt)
    anchors = []
    for t in times:
        mask = tt == t
        radius = 0
        while mask.sum() < 500:
            radius += 16
            mask = np.abs(tt.astype(int)-int(t)) <= radius
        x = np.sort(pred[mask])
        u = (np.arange(len(x))+.5)/len(x)
        anchors.append((x, u))
    return times, anchors


def pack_anchors(maps):
    times, anchors = maps
    times = np.asarray(times)
    if (times.ndim != 1 or times.dtype.kind not in "iu" or not times.size
            or times.size != len(anchors) or np.any(np.diff(times) <= 0)
            or times[0] < 0 or times[-1] >= T_MAX or np.any(times % 16)):
        raise ValueError("invalid anchor times")
    values, quantiles, offsets, midpoints, alpha = [], [], [0], [], []
    for x, u in anchors:
        _validate_band(np.asarray(x), np.asarray(u))
    for t in range(T_MAX):
        hi = min(np.searchsorted(times, t), len(times)-1)
        lo = max(hi-1, 0)
        if times[hi] == t:
            lo = hi
        a = 0. if hi == lo else (float(t)-times[lo])/(times[hi]-times[lo])
        a = min(max(a, 0.), 1.)
        xl, ul = anchors[lo]
        xr, ur = anchors[hi]
        values.extend((xl, xr))
        quantiles.extend((ul, ur))
        midpoints.append(offsets[-1] + len(xl))
        offsets.append(midpoints[-1] + len(xr))
        alpha.append(a)
    return dict(format=np.asarray(FORMAT), values=np.concatenate(values), quantiles=np.concatenate(quantiles),
                offsets=np.asarray(offsets, np.int64), midpoints=np.asarray(midpoints, np.int64),
                alpha=np.asarray(alpha, np.float64))


def _validate_band(x, u):
    if (x.dtype != np.float64 or u.dtype != np.float64 or x.ndim != 1 or x.shape != u.shape
            or not len(x) or not np.isfinite(x).all() or np.any(np.diff(x) < 0)
            or not np.array_equal(u, (np.arange(len(x))+.5)/len(x))):
        raise ValueError("invalid map band / quantiles")


class AnchorCDF:
    def __init__(self, payload):
        if set(payload) != KEYS or np.asarray(payload["format"]).shape != () or str(payload["format"]) != FORMAT:
            raise ValueError("map format/keys")
        for key in KEYS - {"format"}:
            a = np.asarray(payload[key])
            dtype = np.int64 if key in ("offsets", "midpoints") else np.float64
            if a.dtype != dtype or a.ndim != 1 or not np.isfinite(a).all():
                raise ValueError(f"map dtype/shape/finite: {key}")
            setattr(self, key, np.array(a, copy=True, order="C"))
        if (self.offsets.shape != (1001,) or self.midpoints.shape != (1000,) or self.alpha.shape != (1000,)
                or self.values.shape != self.quantiles.shape or self.offsets[0] != 0
                or self.offsets[-1] != self.values.size or np.any(np.diff(self.offsets) <= 1)
                or np.any(self.midpoints <= self.offsets[:-1]) or np.any(self.midpoints >= self.offsets[1:])
                or np.any(self.alpha < 0) or np.any(self.alpha > 1) or self.alpha[-1] not in (0., 1.)):
            raise ValueError("map offsets/midpoints/alpha")
        self._bands = []
        for t in range(T_MAX):
            start, mid, end = self.offsets[t], self.midpoints[t], self.offsets[t+1]
            bands = (self.values[start:mid], self.quantiles[start:mid],
                     self.values[mid:end], self.quantiles[mid:end])
            _validate_band(*bands[:2])
            _validate_band(*bands[2:])
            self._bands.append(bands)
        for key in KEYS - {"format"}:
            getattr(self, key).flags.writeable = False

    def prewarm(self):
        """Validation and all views are prepared eagerly by __init__."""

    def evaluate(self, t, s):
        t = operator.index(t)
        if t < 0:
            raise ValueError("negative causal step")
        if not np.isfinite(s):
            raise ValueError("nonfinite map score")
        t = min(t, T_MAX-1)
        xl, ul, xr, ur = self._bands[t]
        a = self.alpha[t]
        left = np.interp(float(s), xl, ul)
        right = np.interp(float(s), xr, ur)
        return float((1-a)*left+a*right)

    def evaluate_rows(self, scores, times):
        scores, times = np.asarray(scores), np.asarray(times)
        if scores.ndim != 1 or scores.shape != times.shape or times.dtype.kind not in "iu":
            raise ValueError("map score/time rows")
        return np.asarray([self.evaluate(t, s) for t, s in zip(times, scores, strict=True)], np.float64)

    def nbytes(self):
        return sum(getattr(self, k).nbytes for k in KEYS - {"format"})


def save_anchor_cdf(path, maps):
    payload = pack_anchors(maps)
    AnchorCDF(payload)
    with open(path, "xb") as fh:
        np.savez(fh, **payload)


def load_anchor_cdf(path):
    with np.load(path, allow_pickle=False) as data:
        return AnchorCDF({k: data[k] for k in data.files})
