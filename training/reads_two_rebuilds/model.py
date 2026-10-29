"""Network of reads_two_rebuilds: a 2-layer GRU (hidden 128) + MLP head over 354 streaming features per online point.

Inputs per point (from ``features.extract``): surprises 182 | pattern_removed 110 | summary 36 | traces 22 |
context 4, float32.

Transform, with statistics computed on the training rows only:
    z = clip(asinh((x - median) / scale), -8, 8),  non-finite -> 0,
    scale = IQR / 1.349 (the sd if the IQR is ~0, else 1).
The transformed table is stored in float16 on the GPU and batches are cast back
to float32, so the network is trained on fp16-quantised inputs; inference
quantises its inputs the same way (``quantize_fp16``).

Training (``train_one``), one learner per bag index b:
    * target = the per-step label; weight of a training row at step t =
      n_pos(t) n_neg(t) / n(t), normalised to mean 1 over the training rows;
    * 10 % of the training series (``default_rng([20260923, s, k, b])``) are an
      inner validation set; the best epoch is the one with the lowest inner
      weighted loss over 20 epochs (cosine schedule, no patience);
    * AdamW (lr 2e-3, wd 1e-4), linear warm-up over the first epoch then
      cosine to 0, gradient clip 1.0, batches of 64 series sorted by length,
      BCE-with-logits times the row weight, normalised by (total training
      weight / number of batches);
    * seeds: ``torch.manual_seed(1000 (b + 1) + 10 s + k)`` and the batch
      order from ``default_rng`` with the same seed.
The member's logit is the mean of the logits of its 3 learners (b = 0, 1, 2),
before one per-step CDF map.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

SRC_ORDER = ("surprises", "pattern_removed", "summary", "traces", "context")
NCOL = {"surprises": 182, "pattern_removed": 110, "summary": 36, "traces": 22, "context": 4}
N_FEAT = 354
NAMES = [f"{nm}_{j}" for nm in SRC_ORDER for j in range(NCOL[nm])]
HDIM, HHEAD = 128, 64
BATCH, BATCH_EVAL = 64, 64
LR, WD, CLIP = 2e-3, 1e-4, 1.0
INNER_FRAC = 0.10
E_MAX = 20
N_T = 1000
SQRT1_2 = 0.7071067811865476
LEARNER_FORMAT = "reads_two_rebuilds/v1"
NET_KEYS = {'gru.weight_ih_l0': 'Wih0', 'gru.weight_hh_l0': 'Whh0', 'gru.bias_ih_l0': 'bih0', 'gru.bias_hh_l0': 'bhh0',
            'gru.weight_ih_l1': 'Wih1', 'gru.weight_hh_l1': 'Whh1', 'gru.bias_ih_l1': 'bih1', 'gru.bias_hh_l1': 'bhh1',
            'head.0.weight': 'W1', 'head.0.bias': 'b1', 'head.2.weight': 'W2', 'head.2.bias': 'b2'}


# ------------------------------------------------------------------------------------------------ transform
def col_stats(col: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Median and robust scale of one column over the rows ``idx`` (finite values only)."""
    c = col[idx].astype(np.float64)
    c = c[np.isfinite(c)]
    assert len(c) > 0, 'column without finite values in the training rows'
    q25, q50, q75 = np.quantile(c, [0.25, 0.5, 0.75])
    s = (q75 - q25) / 1.349
    if not s > 1e-12:
        s = c.std() if c.std() > 1e-12 else 1.0
    return q50, s


def transform(X: np.ndarray, med: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """float64 arithmetic: asinh, clip +-8, non-finite -> 0; returned as float32."""
    z = (X.astype(np.float64) - med) / sc
    z = np.clip(np.arcsinh(z), -8.0, 8.0)
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def quantize_fp16(z: np.ndarray) -> np.ndarray:
    """Input as seen by the network: float32(z) -> float16 (round to nearest even) -> float64."""
    return z.astype(np.float32).astype(np.float16).astype(np.float64)


def target_y(TT: np.ndarray, Y: np.ndarray, rows_of: list, tr_idx: np.ndarray):
    """Target (the label on training rows, 0 elsewhere) and per-row weights n_pos n_neg / n at the row's step."""
    rows_tr = np.zeros(len(Y), bool)
    tr_rows_ordered = np.concatenate([rows_of[i] for i in tr_idx])
    rows_tr[tr_rows_ordered] = True
    tgt = np.where(rows_tr, Y.astype(np.float64), 0.0)
    ttr, ytr = TT[rows_tr], Y[rows_tr]
    nt = np.bincount(ttr, minlength=N_T).astype(float)
    npos = np.bincount(ttr, weights=ytr, minlength=N_T)
    wt_t = np.where(nt > 0, npos * (nt - npos) / np.maximum(nt, 1), 0.0)
    wt_t = wt_t / wt_t[ttr].mean()
    w = np.where(rows_tr, wt_t[TT], 0.0)
    return tgt, w


# ------------------------------------------------------------------------------------------------ torch side
_NET = {}


def net_class():
    """The PyTorch module (created on first use so that numpy-only steps do not import torch)."""
    if 'Net' not in _NET:
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, f: int, hdim: int = HDIM, hh: int = HHEAD):
                super().__init__()
                self.gru = nn.GRU(f, hdim, num_layers=2, batch_first=True, dropout=0.1)
                self.head = nn.Sequential(nn.Linear(hdim + f, hh), nn.GELU(), nn.Linear(hh, 1))

            def forward(self, x):
                h, _ = self.gru(x)
                return self.head(torch.cat([h, x], -1)).squeeze(-1)

        _NET['Net'] = Net
    return _NET['Net']


def set_determinism():
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Data:
    """Transformed feature table (float16, on the GPU) + per-series row ranges; batches are gathered by index."""

    def __init__(self, n_on: np.ndarray, fo: np.ndarray, dev):
        import torch

        self.dev = dev
        self.non = torch.from_numpy(n_on.astype(np.int64)).to(dev)
        self.fo = torch.from_numpy(fo.astype(np.int64)).to(dev)
        self.lens = n_on.copy()
        self.PAD_FEAT = int(fo[-1] + n_on[-1])
        self.FEAT = None
        self.TGT = None
        self.W = None

    def set_table(self, feat, tgt, w):
        import torch

        self.FEAT = feat
        self.TGT = torch.from_numpy(np.r_[tgt, 0.0].astype(np.float32)).to(self.dev)
        self.W = torch.from_numpy(np.r_[w, 0.0].astype(np.float32)).to(self.dev)

    def clear(self):
        import torch

        self.FEAT = None
        self.TGT = None
        self.W = None
        torch.cuda.empty_cache()

    def batch(self, bi: np.ndarray):
        import torch

        b = torch.from_numpy(bi.astype(np.int64)).to(self.dev)
        L = int(self.lens[bi].max())
        pos = torch.arange(L, device=self.dev)[None, :]
        on = pos < self.non[b][:, None]
        fidx = torch.where(on, self.fo[b][:, None] + pos, torch.full_like(pos, self.PAD_FEAT))
        x = self.FEAT[fidx]
        if x.dtype != torch.float32:
            x = x.float()
        return x, self.TGT[fidx], self.W[fidx], on, fidx


def batches(idx, lens, bs, rng=None):
    order = idx[np.argsort(lens[idx], kind='stable')]
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if rng is not None:
        rng.shuffle(chunks)
    return chunks


def inner_loss(net, data: Data, idx):
    """Weighted inner-validation loss (the early-stopping criterion)."""
    import torch
    import torch.nn as nn

    lossf = nn.BCEWithLogitsLoss(reduction='none')
    num = den = 0.0
    net.eval()
    with torch.no_grad():
        for bi in batches(idx, data.lens, BATCH_EVAL):
            x, y, w, _, _ = data.batch(bi)
            lo = net(x)
            num += float((lossf(lo, y) * w).sum())
            den += float(w.sum())
    return num / den


def train_one(data: Data, s, k, b, tr_idx, rows_of, e_max, dev, log):
    import torch
    import torch.nn as nn

    rng_split = np.random.default_rng([20260923, s, k, b])
    perm = rng_split.permutation(len(tr_idx))
    n_iv = int(round(INNER_FRAC * len(tr_idx)))
    iv_idx = np.sort(tr_idx[perm[:n_iv]])
    it_idx = np.sort(tr_idx[perm[n_iv:]])
    seed = 1000 * (b + 1) + 10 * s + k
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = net_class()(N_FEAT).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    nb = math.ceil(len(it_idx) / BATCH)
    total = e_max * nb
    warm = nb

    def lr_lambda(step):
        return min(1.0, (step + 1) / warm) * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    wv = data.W[:-1].cpu().numpy()
    it_rows = np.concatenate([rows_of[i] for i in it_idx])
    norm = float(wv[it_rows].astype(np.float64).sum()) / nb
    lossf = nn.BCEWithLogitsLoss(reduction='none')
    best, best_ep, best_state = np.inf, -1, None
    t0 = time.time()
    for ep in range(e_max):
        net.train()
        tot = 0.0
        for bi in batches(it_idx, data.lens, BATCH, rng):
            x, y, w, _, _ = data.batch(bi)
            lo = net(x)
            loss = (lossf(lo, y) * w).sum() / norm
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            sched.step()
            tot += float(loss)
        vl = inner_loss(net, data, iv_idx)
        torch.cuda.empty_cache()
        log(f'  k{k} b{b} ep{ep:2d} train {tot / nb:.5f} inner {vl:.5f} ({time.time() - t0:.0f}s)')
        if not np.isfinite(vl):
            log(f'  k{k} b{b}: non-finite inner loss at epoch {ep}; keeping the best previous epoch')
            break
        if vl < best - 1e-6:
            best, best_ep = vl, ep
            best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
    assert best_state is not None, f'k{k} b{b}: no epoch with a finite inner loss'
    net.load_state_dict(best_state)
    return net, dict(best_ep=best_ep, best_inner=best, secs=time.time() - t0, n_it=int(len(it_idx)),
                     n_iv=int(len(iv_idx)), seed=seed)


def state_dict_np(net) -> dict:
    return {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}


# ------------------------------------------------------------------------------------------------ learner files
def save_learner(path: Path, sd_np: dict, med: np.ndarray, sc: np.ndarray, meta: dict) -> None:
    """One learner: float32 weights (named as in ``NET_KEYS``), transform statistics and metadata."""
    assert set(sd_np) == set(NET_KEYS), sorted(sd_np)
    W = {NET_KEYS[k]: np.ascontiguousarray(v) for k, v in sd_np.items()}
    assert W['Wih0'].shape == (384, N_FEAT) and W['W1'].shape == (64, 128 + N_FEAT)
    m = dict(meta, format=LEARNER_FORMAT, input_set='reads_two_rebuilds', sources=list(SRC_ORDER), n_features=N_FEAT,
             quant='fp16', weight_dtype=str(W['Wih0'].dtype))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.npz')
    np.savez(tmp, med=np.asarray(med, np.float64), sc=np.asarray(sc, np.float64), names=np.array(NAMES),
             meta_json=np.array(json.dumps(m, sort_keys=True)), **W)
    os.replace(tmp, path)


def load_learner(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta_json']))
    assert meta['format'] == LEARNER_FORMAT, meta.get('format')
    P = {v: np.ascontiguousarray(z[v], dtype=np.float64) for v in NET_KEYS.values()}
    return dict(P=P, med=np.asarray(z['med'], np.float64), sc=np.asarray(z['sc'], np.float64), meta=meta)


# ------------------------------------------------------------------------------------------------ numpy float64 forward
def np_net_step(x, h1, h2, P):
    """One online step of the GRU + head in float64 (same equations as the inference kernels)."""
    H = HDIM
    gi = P['Wih0'] @ x + P['bih0']
    gh = P['Whh0'] @ h1 + P['bhh0']
    r = 1.0 / (1.0 + np.exp(-(gi[:H] + gh[:H])))
    z = 1.0 / (1.0 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    h1 = (h1 - n) * z + n
    gi = P['Wih1'] @ h1 + P['bih1']
    gh = P['Whh1'] @ h2 + P['bhh1']
    r = 1.0 / (1.0 + np.exp(-(gi[:H] + gh[:H])))
    z = 1.0 / (1.0 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    h2 = (h2 - n) * z + n
    a = P['W1'] @ np.concatenate([h2, x]) + P['b1']
    g = a * 0.5 * (1.0 + np.array([math.erf(v) for v in (a * SQRT1_2).tolist()]))
    return float(P['W2'][0] @ g + P['b2'][0]), h1, h2


def series_logit(X32: np.ndarray, learners: list) -> np.ndarray:
    """Bag-mean logit of one series: transform + fp16 quantisation + float64 GRU per learner, sequential mean."""
    med, sc = learners[0]['med'], learners[0]['sc']
    z = (X32.astype(np.float64) - med) / sc
    z = np.clip(np.arcsinh(z), -8.0, 8.0)
    z[~np.isfinite(z)] = 0.0
    X = quantize_fp16(z)
    acc = np.zeros(X.shape[0])
    for L in learners:
        h1, h2 = np.zeros(HDIM), np.zeros(HDIM)
        out = np.empty(X.shape[0])
        for j in range(X.shape[0]):
            out[j], h1, h2 = np_net_step(np.ascontiguousarray(X[j]), h1, h2, L['P'])
        acc += out
    return acc / len(learners)
