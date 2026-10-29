"""Full-data fit of the sequence network on the 8,000 search series (one CUDA device).

    python -m common.holdout --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/search_split"
    D="$OUT_DIR/search_split"
    python -m networks.train_full --variant pointwise --data "$D" --out "$OUT_DIR/broken_or_not"
    python -m networks.train_full --variant richer --data "$D" --out "$OUT_DIR/which_one_broke_base"
    python -m networks.train_full --variant richer_scale_blind --data "$D" --out "$OUT_DIR/blind_to_size"

All 8,000 series are training series; 10 % of them (rng [SEED, 9, 9, 11]) are
the inner set used to pick the best epoch of each stage.  Writes
``<out>/full/``: ``net_pre.pt``, ``net_lp.pt``, ``net_ft.pt``, ``info.json`` and
``preds_full.npz`` (float64 LP/FT logits at every step t % 16 == 0 and the
per-series offsets d_LP, d_FT, used by ``calibrate.py``).

``--smoke`` runs the whole path on the CPU with 120 series and 1-2 epochs
(plumbing test only).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import sequence_net as J
from . import variants


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', required=True, choices=variants.VARIANTS)
    ap.add_argument('--data', type=Path, default=os.environ.get('SEARCH_DATA_DIR'),
                    help='folder with the three *.search.parquet files (common.holdout)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.data is None:
        ap.error('set --data or SEARCH_DATA_DIR')
    variants.install(a.variant)
    if a.smoke:
        J.LP_EPOCHS, J.FT_EPOCHS = 1, 1
    torch, _ = J.torch_mod()
    out = a.out / 'full'
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cpu') if a.smoke else torch.device('cuda:0')
    amp = dev.type == 'cuda'
    fd = torch.float32
    s, k = 9, 9
    tr = np.arange(8000)
    rng = np.random.default_rng([J.SEED, s, k, 11])
    perm = rng.permutation(len(tr))
    n_iv = int(round(J.INNER_FRAC * len(tr)))
    iv = np.sort(tr[perm[:n_iv]])
    pt = np.sort(tr[perm[n_iv:]])
    if a.smoke:
        pt, iv = pt[:96], iv[:24]
    ids, hist, onl, lab = J.load_data(Path(a.data), want=set(np.r_[pt, iv].tolist()) if a.smoke else None)
    info = dict(kind='model_full', variant=a.variant, n_pt=int(len(pt)), n_iv=int(len(iv)), torch=torch.__version__,
                device=str(dev))
    trn = np.r_[pt, iv]
    desc = np.stack([J.series_desc(hist[i]) for i in trn])
    knn_pt = J.knn_within(desc[:len(pt)])
    knn_iv = J.knn_within(desc[len(pt):], k=min(J.KNN, len(iv) - 1))
    store = J.Store([hist[i] for i in trn], dev)
    torch.manual_seed(J.SEED + 10 * s + k)
    net = J.make_net().to(dev)
    info['pretrain'] = J.pretrain(net, store, np.arange(len(pt)), np.arange(len(pt), len(trn)), knn_pt, knn_iv, s, k,
                                  dev, amp, fd, 2 if a.smoke else J.EPOCHS_PRE, 1 if a.smoke else J.EX_PER_SERIES)
    del store
    torch.save({kk: v.cpu() for kk, v in net.state_dict().items()}, out / 'net_pre.pt')
    rd = J.RealData(hist, onl, lab, dev)
    tt_fit = np.concatenate([np.arange(len(onl[i])) for i in pt])
    y_fit = np.concatenate([lab[i] for i in pt]).astype(float)
    npos, nneg = J.pair_weights(tt_fit, y_fit)
    wbar = (np.dot(npos, nneg) * 2.0) / len(tt_fit)
    wpos, wneg = (nneg / wbar).astype(np.float32), (npos / wbar).astype(np.float32)
    info['lp'] = J.supervised(net, rd, pt, iv, wpos, wneg, s, k, dev, amp, fd, 'lp')
    lp_state = {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}
    torch.save(lp_state, out / 'net_lp.pt')
    info['ft'] = J.supervised(net, rd, pt, iv, wpos, wneg, s, k, dev, amp, fd, 'ft')
    ft_state = {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}
    torch.save(ft_state, out / 'net_ft.pt')
    allidx = trn if a.smoke else np.arange(8000)
    t0 = time.time()
    res = {}
    for tag, st in (('LP', lp_state), ('FT', ft_state)):
        net64 = J.make_net().double().to(dev)
        net64.load_state_dict({kk: v.double() for kk, v in st.items()})
        net64.eval()
        lo, _ = J.infer_all(net64, rd, allidx, want_pl=False)
        res[tag] = lo
        res['d_' + tag], _ = J.offsets(net64, rd, allidx)
    info['infer_secs'] = time.time() - t0
    n_on = np.array([len(o) for o in onl], np.int64)
    offs = np.r_[0, np.cumsum(n_on)]
    outz = dict(ids=ids, series=allidx)
    for tag in ('LP', 'FT'):
        full = np.full(int(offs[-1]), np.nan)
        for c_, i in enumerate(allidx):
            full[offs[i]:offs[i + 1]] = res[tag][c_]
        tt = np.concatenate([np.arange(v) for v in n_on])
        have = np.zeros(8000, bool)
        have[allidx] = True
        rs = np.repeat(np.arange(8000), n_on)
        rows16 = np.flatnonzero(have[rs] & (tt % 16 == 0))
        outz['train_rows'] = rows16
        outz[f'train_raw_{tag}'] = full[rows16]
        d = np.full(8000, np.nan)
        d[allidx] = res['d_' + tag]
        outz[f'd_{tag}'] = d
    np.savez_compressed(out / 'preds_full.npz', **outz)
    for f in ('net_pre.pt', 'net_lp.pt', 'net_ft.pt', 'preds_full.npz'):
        info[f + '_sha'] = J.sha_file(out / f)
    if variants.STATE['gen_stats']:
        gs = variants.STATE['gen_stats']
        info['generator'] = dict(n_calls=len(gs), rows=int(sum(g['n_rows'] for g in gs)),
                                 points=int(sum(g['n_points'] for g in gs)), clipped=int(sum(g['n_clip'] for g in gs)))
    (out / 'info.json').write_text(json.dumps(info, indent=1, default=float))
    J.log('full-data fit DONE')


if __name__ == '__main__':
    main()
