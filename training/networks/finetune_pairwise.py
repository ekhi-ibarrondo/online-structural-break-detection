"""Second stage of which_one_broke: pairwise fine-tune of the ``richer`` network, from its linear-probe state.

    python -m networks.train_full --variant richer --data "$OUT_DIR/search_split" \\
        --out "$OUT_DIR/which_one_broke_base"
    python -m networks.finetune_pairwise --data "$OUT_DIR/search_split" \\
        --source "$OUT_DIR/which_one_broke_base/full" --out "$OUT_DIR/which_one_broke"

The pretraining and the linear probe are those of the ``richer`` full-data
fit (``net_lp.pt`` = state after the best linear-probe epoch).  Only the
fine-tune changes: a fresh AdamW and the FT schedule of the default recipe, with
the per-step BCE replaced by a pairwise ranking loss.  In every batch, for every
online step t with at least one positive and one negative series, all
(positive i, negative j) pairs contribute softplus(o_j - o_i); the sum is
divided by ``norm_pairs``, the number of same-t pairs in the epoch's batch plan
divided by the number of batches (each pair weighs the same in the whole
epoch).  Batches without pairs only advance the scheduler.  The best epoch is
chosen by the inner TS-AUC, as in the default recipe.  Writes ``<out>/full/``:
``net_ft.pt``, ``preds_full.npz`` (float64 FT logits at t % 16 == 0 and d_FT)
and ``info.json``.
"""
from __future__ import annotations

import os

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import sequence_net as J  # noqa: E402


def split(n=8000):
    """Full-data split: every series trains; inner set = 10 % (rng [SEED, 9, 9, 11])."""
    s, k = 9, 9
    tr = np.arange(n)
    rng = np.random.default_rng([J.SEED, s, k, 11])
    perm = rng.permutation(len(tr))
    n_iv = int(round(J.INNER_FRAC * len(tr)))
    iv, pt = np.sort(tr[perm[:n_iv]]), np.sort(tr[perm[n_iv:]])
    return pt, iv


def pair_count(lab, series):
    """Number of same-t (positive, negative) pairs among ``series`` (labels only)."""
    m = max(len(lab[i]) for i in series)
    pos, neg = np.zeros(m, np.int64), np.zeros(m, np.int64)
    for i in series:
        y = np.asarray(lab[i])
        assert np.isin(y, (0, 1)).all()
        pos[:len(y)] += y.astype(np.int64)
        neg[:len(y)] += 1 - y.astype(np.int64)
    return int(np.dot(pos, neg))


def norm_pairs(lab, series, plan):
    counts = [pair_count(lab, series[ch]) for ch in plan]
    return sum(counts) / len(plan), counts


def pair_loss(lo, y, on, ti, norm):
    """All positive-negative cross-series pairs at the same online t; float32.

    The online logits are scattered into a (series, t) grid first, which drops
    context and padding.  Chunking t bounds GPU memory without subsampling pairs.
    """
    torch, _ = J.torch_mod()
    assert norm > 0
    b, q = torch.where(on)
    if not len(b):
        return lo.float().sum() * 0
    mt = int(ti[b, q].max()) + 1
    scores = torch.zeros((lo.shape[0], mt), dtype=torch.float32, device=lo.device)
    pos = torch.zeros_like(scores, dtype=torch.bool)
    neg = torch.zeros_like(pos)
    scores[b, ti[b, q]] = lo.float()[b, q]
    pos[b, ti[b, q]] = y[b, q] > 0.5
    neg[b, ti[b, q]] = y[b, q] <= 0.5
    total = lo.float()[on].sum() * 0
    for start in range(0, mt, 32):
        sc = scores[:, start:start + 32].T
        valid = pos[:, start:start + 32].T[:, :, None] & neg[:, start:start + 32].T[:, None, :]
        differences = sc[:, None, :] - sc[:, :, None]
        total = total + torch.nn.functional.softplus(differences[valid]).sum()
    return total / norm


def supervised_pairwise(net, rd, rows_fit_series, iv_series, wpos, wneg, s, k, dev, amp, feat_dtype, labels,
                        pair_history):
    """``sequence_net.supervised(..., 'ft')`` with the pairwise loss (same optimiser, schedule, seeds and selection)."""
    torch, nn = J.torch_mod()
    for p in net.parameters():
        p.requires_grad_(True)
    groups = [dict(params=[p for n_, p in net.named_parameters() if n_.startswith('d')], lr=J.FT_LR_HEAD),
              dict(params=[p for n_, p in net.named_parameters() if not n_.startswith('d')], lr=J.FT_LR_ENC)]
    epochs = J.FT_EPOCHS
    opt = torch.optim.AdamW(groups, weight_decay=J.WD)
    L = np.minimum(J.H_KEEP, rd.HS.len[rows_fit_series] - J.P) + rd.OS.len[rows_fit_series]
    nb = math.ceil(len(rows_fit_series) / J.FT_BATCH)
    total, warm = epochs * nb, max(1, nb // 2)

    def lr_lambda(step):
        w = min(1.0, (step + 1) / warm)
        return w * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, best_ep, best_state, hist = -np.inf, -1, None, []
    t0 = time.time()
    for ep in range(epochs):
        torch.manual_seed(J.SEED + 300 + 10 * s + k + 7919 * (ep + 1) + 55)
        rng = np.random.default_rng([J.SEED, s, k, 400 + ep, 1])
        plan = J.batches_by_len(L, J.FT_BATCH, rng)
        epoch_norm, counts = norm_pairs(labels, rows_fit_series, plan)
        assert epoch_norm > 0, 'epoch has no pairs'
        pair_history.append(dict(ep=ep, norm_pairs=epoch_norm, total_pairs=sum(counts), nb=nb,
                                 zero_pair_batches=sum(c == 0 for c in counts)))
        rng = np.random.default_rng([J.SEED, s, k, 400 + ep, 1])
        net.train()
        tot, te = 0.0, time.time()
        for ch in J.batches_by_len(L, J.FT_BATCH, rng):
            b = rows_fit_series[ch]
            if pair_count(labels, b) == 0:
                sched.step()
                continue
            X, Yseq, on, hl, W, ti = rd.batch(b, feat_dtype, wpos, wneg)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                lo, _ = net(X, down=True)
            loss = pair_loss(lo.float(), Yseq, on, ti, epoch_norm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], J.CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
        auc = J.eval_inner(net, rd, iv_series, feat_dtype, amp)
        hist.append(dict(ep=ep, train=tot / nb, inner_tsauc=auc, secs=round(time.time() - te, 1)))
        J.log(f'  ft s{s}k{k} ep{ep} train {tot / nb:.5f} inner TS-AUC {auc:.5f} ({time.time() - te:.1f}s)')
        if auc > best + 1e-9:
            best, best_ep = auc, ep
            best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    for p in net.parameters():
        p.requires_grad_(True)
    return dict(best_ep=best_ep, best_inner_tsauc=best, hist=hist, secs=time.time() - t0)


def run(data: Path, source: Path, root: Path) -> None:
    torch, _ = J.torch_mod()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    out = root / 'full'
    assert not out.exists() or not any(out.iterdir()), 'refusing to overwrite an existing fit'
    pt, iv = split()
    ids, hist, onl, lab = J.load_data(data, want=set(np.r_[pt, iv].tolist()))
    dev, amp, fd = torch.device('cuda:0'), True, torch.float32
    net = J.make_net().to(dev)
    net.load_state_dict(torch.load(source / 'net_lp.pt', map_location=dev, weights_only=True))
    rd = J.RealData(hist, onl, lab, dev)
    tt_fit = np.concatenate([np.arange(len(onl[i])) for i in pt])
    y_fit = np.concatenate([lab[i] for i in pt]).astype(float)
    npos, nneg = J.pair_weights(tt_fit, y_fit)
    wbar = (np.dot(npos, nneg) * 2.0) / len(tt_fit)
    wpos, wneg = (nneg / wbar).astype(np.float32), (npos / wbar).astype(np.float32)
    pairs = []
    ft = supervised_pairwise(net, rd, pt, iv, wpos, wneg, 9, 9, dev, amp, fd, lab, pairs)
    assert len(pairs) == J.FT_EPOCHS == len(ft['hist']) and all(p['norm_pairs'] > 0 for p in pairs)
    out.mkdir(parents=True, exist_ok=True)
    state = {key: val.detach().cpu().clone() for key, val in net.state_dict().items()}
    torch.save(state, out / 'net_ft.pt')
    net64 = J.make_net().double().to(dev)
    net64.load_state_dict({key: val.double() for key, val in state.items()})
    net64.eval()
    ti = time.time()
    allidx = np.arange(8000)
    lo, _ = J.infer_all(net64, rd, allidx, want_pl=False)
    d, _ = J.offsets(net64, rd, allidx)
    n_on = np.array([len(o) for o in onl], np.int64)
    tt = np.concatenate([np.arange(v) for v in n_on])
    train_rows = np.flatnonzero(tt % 16 == 0)
    flat = np.concatenate(lo)
    arrays = dict(ids=ids, train_rows=train_rows, train_raw_FT=flat[train_rows], d_FT=d, series=allidx)
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(out / 'preds_full.npz', **arrays)
    info = dict(kind='model_full', loss='pairwise_t', n_pt=len(pt), n_iv=len(iv), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0), ft=ft, norm_pairs=pairs, weights=dict(wbar=float(wbar)),
                infer_secs=time.time() - ti, source_net_lp_sha=J.sha_file(source / 'net_lp.pt'))
    info['net_ft.pt_sha'] = J.sha_file(out / 'net_ft.pt')
    info['preds_full.npz_sha'] = J.sha_file(out / 'preds_full.npz')
    (out / 'info.json').write_text(json.dumps(info, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    J.log(f'pairwise fine-tune DONE, {len(ft["hist"])} epochs, best epoch {ft["best_ep"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=os.environ.get('SEARCH_DATA_DIR'),
                    help='folder with the three *.search.parquet files (common.holdout)')
    ap.add_argument('--source', type=Path, required=True, help='full-data fit of the richer variant (has net_lp.pt)')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.data is None:
        ap.error('set --data or SEARCH_DATA_DIR')
    run(Path(a.data), a.source, a.out)


if __name__ == '__main__':
    main()
