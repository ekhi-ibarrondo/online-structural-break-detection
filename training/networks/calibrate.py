"""Calibration of a full-data sequence-network fit: shrinkage b, anchor CDF map and fallback curve.

    python -m networks.calibrate --preds "$OUT_DIR/broken_or_not/full/preds_full.npz" \\
        --data "$OUT_DIR/search_split" --out "$OUT_DIR/broken_or_not"

The released score is ``raw = FT logit - b * d_FT`` where ``d_FT``
is the series' pseudo-online offset (mean FT logit over windows of its own
history) and ``b`` is the OLS slope, over the label-0 series, of the mean FT
logit at steps 25 <= t < 250 on ``d_FT``.  Then, on these raw scores of every
training row with t % 16 == 0:

* map: one anchor per step t = 0, 16, ..., 992 holding the sorted scores of
  that step (widened by +-16 steps until it has at least 500 rows); tied
  scores collapse to one knot at their mid-rank.  Written as ``times``,
  ``offsets``, ``x``, ``u`` (+ ``seed = fold = -1``) to ``<name>_map.npz``;
* fallback: the median raw score of each anchor step, linearly
  interpolated over t = 0..999 (``fallback_raw.npy``);
* ``member_config.json``: the score formula, the weight file and b.

The original calibration recomputed the raw scores with the CPU (numpy /
numba float64) member; the float64 GPU logits saved by the fit are the same
computation up to the last bits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

T_MAX = 1000
STRIDE = 16
MIN_ROWS = 500


def search_geometry(data: Path):
    """Online lengths and tau_index of the 8,000 search series, ascending ids."""
    import pyarrow.parquet as pq

    yid = pq.read_table(data / 'y_train.search.parquet', columns=['id']).column('id').to_numpy().astype(np.int64)
    ids, n_on = np.unique(yid, return_counts=True)
    yi = pq.read_table(data / 'y_train_index.search.parquet', columns=['id', 'tau_index'])
    tau = dict(zip(yi.column('id').to_numpy().tolist(), yi.column('tau_index').to_numpy().tolist(), strict=True))
    return ids, n_on.astype(np.int64), np.array([int(tau[int(i)]) for i in ids], np.int64)


def shrinkage_b(traw, d, row_series, times, tau_index):
    """OLS slope of the mean raw logit at 25 <= t < 250 on the per-series offset, over label-0 series."""
    sel = (times >= 25) & (times < 250) & (tau_index[row_series] < 0)
    u, inv = np.unique(row_series[sel], return_inverse=True)
    ybar = np.bincount(inv, weights=traw[sel]) / np.bincount(inv)
    return float(np.cov(d[u], ybar)[0, 1] / np.var(d[u], ddof=1))


def cdf_maps(pred, tt):
    """Label-free anchor maps fitted on training predictions at stride-16 steps."""
    times = np.unique(tt)
    anchors = []
    for t in times:
        mask = tt == t
        radius = 0
        while mask.sum() < MIN_ROWS:
            radius += STRIDE
            mask = np.abs(tt.astype(int) - int(t)) <= radius
        x = np.sort(pred[mask])
        u = (np.arange(len(x)) + .5) / len(x)
        xu, first, counts = np.unique(x, return_index=True, return_counts=True)
        if len(xu) < len(x):
            # a block of tied scores maps to its mid-rank
            u = (first + counts / 2.) / len(x)
            x = xu
        anchors.append((x, u))
    return times, anchors


def pack_map(times, anchors) -> dict:
    lens = [len(x) for x, _ in anchors]
    return dict(times=np.asarray(times, np.int64), offsets=np.r_[0, np.cumsum(lens)].astype(np.int64),
                x=np.concatenate([x for x, _ in anchors]).astype(np.float64),
                u=np.concatenate([u for _, u in anchors]).astype(np.float64),
                seed=np.array(-1, np.int64), fold=np.array(-1, np.int64))


def fallback_curve(raw, times, anchor_times) -> np.ndarray:
    med = np.array([np.median(raw[times == t]) for t in anchor_times])
    return np.interp(np.arange(T_MAX, dtype=np.float64), anchor_times.astype(np.float64), med)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preds', type=Path, required=True, help='preds_full.npz of a full-data fit')
    ap.add_argument('--data', type=Path, required=True, help='folder with the *.search.parquet files')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--name', default=None, help='prefix of the map file (default: the name of --out)')
    a = ap.parse_args()
    name = a.name or a.out.name
    with np.load(a.preds, allow_pickle=False) as z:
        ids, rows, traw, d = z['ids'], z['train_rows'], z['train_raw_FT'], z['d_FT']
    sids, n_on, tau_index = search_geometry(a.data)
    assert np.array_equal(sids, ids), 'predictions and data cover different series'
    row_series_all = np.repeat(np.arange(len(ids)), n_on)
    t_all = np.concatenate([np.arange(n) for n in n_on])
    row_series, times = row_series_all[rows], t_all[rows]
    assert np.all(times % STRIDE == 0)
    b = shrinkage_b(traw, d, row_series, times, tau_index)
    raw = traw - b * d[row_series]
    anchor_times, anchors = cdf_maps(raw, times)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / f'{name}_map.npz', **pack_map(anchor_times, anchors))
    np.save(a.out / 'fallback_raw.npy', fallback_curve(raw, times, anchor_times))
    config = dict(score='FT logit - b * d_FT', weights='member.npz', b=b, fallback='fallback_raw.npy')
    (a.out / 'member_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'b': b, 'anchors': len(anchor_times), 'rows': int(len(raw))}))


if __name__ == '__main__':
    main()
