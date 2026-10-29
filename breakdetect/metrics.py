"""TS-AUC, the evaluation metric of the ADIA Lab Structural Break Real-Time challenge.

At each online step t the scores of all series that are still running at t are ranked;
TS-AUC is the fraction of (positive, negative) pairs taken within the same step whose
positive point scores higher, with ties counting 0.5. Equivalently it is the average of
the per-step Mann-Whitney AUCs weighted by ``n_pos(t) * n_neg(t)``. The leaderboard shows
``100 * TS-AUC`` points. A point is positive from the break onwards (the break step itself
is labelled 1).
"""

from __future__ import annotations

import numpy as np

__all__ = ["labels_from_break", "points", "ts_auc", "ts_auc_bruteforce", "ts_auc_decomposed"]


def labels_from_break(break_index: int | None, n_online: int) -> np.ndarray:
    """Per-step labels: 1 from ``break_index`` (0-based, inclusive) on; ``None`` or -1 means no break."""
    n_online = int(n_online)
    if n_online <= 0:
        raise ValueError(f"n_online must be positive, got {n_online}")
    labels = np.zeros(n_online, dtype=np.int8)
    if break_index is None or int(break_index) < 0:
        return labels
    break_index = int(break_index)
    if break_index >= n_online:
        raise ValueError(f"break_index={break_index} out of range for n_online={n_online}")
    labels[break_index:] = 1
    return labels


def _stack(scores_list, labels_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the series and flatten them into (step, score, label) arrays."""
    if len(scores_list) != len(labels_list):
        raise ValueError("scores_list and labels_list must have the same length")
    if len(scores_list) == 0:
        raise ValueError("empty dataset")
    t_parts, s_parts, l_parts = [], [], []
    for i, (s, lab) in enumerate(zip(scores_list, labels_list, strict=True)):
        s = np.asarray(s, dtype=np.float64)
        lab = np.asarray(lab)
        if s.ndim != 1 or lab.ndim != 1:
            raise ValueError(f"series {i}: scores and labels must be 1-D")
        if s.shape[0] != lab.shape[0]:
            raise ValueError(f"series {i}: len(scores)={s.shape[0]} != len(labels)={lab.shape[0]}")
        if s.shape[0] == 0:
            raise ValueError(f"series {i} is empty")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"series {i}: scores contain non-finite values")
        if not np.all((lab == 0) | (lab == 1)):
            raise ValueError(f"series {i}: labels must be 0/1")
        t_parts.append(np.arange(s.shape[0], dtype=np.int64))
        s_parts.append(s)
        l_parts.append(lab.astype(np.int64))
    return np.concatenate(t_parts), np.concatenate(s_parts), np.concatenate(l_parts)


def ts_auc_decomposed(scores_list, labels_list) -> dict[str, object]:
    """TS-AUC with the per-step AUC, weights and class counts.

    One lexsort over (step, score) gives the ranks within each step; tied scores share the
    average rank (Mann-Whitney with tie correction). Steps with a single class have weight 0.
    """
    t_all, s_all, l_all = _stack(scores_list, labels_list)
    n = t_all.shape[0]
    t_max = int(t_all.max()) + 1

    order = np.lexsort((s_all, t_all))
    ts = t_all[order]
    ss = s_all[order]
    ls = l_all[order].astype(np.float64)

    new_grp = np.empty(n, dtype=bool)
    new_grp[0] = True
    np.not_equal(ts[1:], ts[:-1], out=new_grp[1:])
    grp_first = np.flatnonzero(new_grp)
    grp_id = np.cumsum(new_grp) - 1
    pos_in_grp = np.arange(n, dtype=np.int64) - grp_first[grp_id]

    new_blk = new_grp.copy()
    new_blk[1:] |= ss[1:] != ss[:-1]
    blk_first = np.flatnonzero(new_blk)
    blk_id = np.cumsum(new_blk) - 1
    blk_sizes = np.diff(np.append(blk_first, n))
    ranks = (pos_in_grp[blk_first] + 1.0 + (blk_sizes - 1) / 2.0)[blk_id]

    n_all = np.bincount(ts, minlength=t_max).astype(np.float64)
    n_pos = np.bincount(ts, weights=ls, minlength=t_max)
    n_neg = n_all - n_pos
    pos_rank_sum = np.bincount(ts, weights=ranks * ls, minlength=t_max)

    w = n_pos * n_neg
    auc = np.full(t_max, np.nan)
    mask = w > 0
    auc[mask] = (pos_rank_sum[mask] - n_pos[mask] * (n_pos[mask] + 1.0) / 2.0) / w[mask]
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        raise ValueError("no step has both classes present: TS-AUC is undefined")
    return {
        "ts_auc": float(np.dot(w[mask], auc[mask]) / w_sum),
        "auc": auc,
        "w": w,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_all": n_all,
    }


def ts_auc(scores_list, labels_list) -> float:
    """TS-AUC in [0, 1]."""
    return float(ts_auc_decomposed(scores_list, labels_list)["ts_auc"])


def points(scores_list, labels_list) -> float:
    """Leaderboard points: 100 * TS-AUC."""
    return 100.0 * ts_auc(scores_list, labels_list)


def ts_auc_bruteforce(scores_list, labels_list) -> float:
    """Reference implementation by explicit pair counting (small inputs only)."""
    t_all, s_all, l_all = _stack(scores_list, labels_list)
    concordant = 0.0
    total_pairs = 0
    for t in range(int(t_all.max()) + 1):
        at_t = t_all == t
        s_t = s_all[at_t]
        l_t = l_all[at_t]
        pos = s_t[l_t == 1]
        neg = s_t[l_t == 0]
        total_pairs += pos.size * neg.size
        for p in pos:
            for q in neg:
                if p > q:
                    concordant += 1.0
                elif p == q:
                    concordant += 0.5
    if total_pairs == 0:
        raise ValueError("no step has both classes present: TS-AUC is undefined")
    return concordant / total_pairs
