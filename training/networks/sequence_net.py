"""Sequence network shared by the broken_or_not, which_one_broke and blind_to_size members.

A causal GRU detector: self-supervised pretraining on history splices and parametric changes, then a linear
probe (LP) and a full fine-tune (FT) on the real per-step labels under pair-count weights; float64 inference for
every online point; per-series pseudo-online offset (own-history null, matched t window).

Recipe:
 R1 Data: the 8,000 search series only (ids checked by sha, see ``common/holdout.py``); order = ascending id.
 R2 Pretraining store = histories of the fit's training series only (90 % gradient / 10 % inner, rng
    [SEED, s, k, 11]); LP/FT = online rows + labels of the same series (same 90/10 inner split for epoch
    selection).
 R3 Inputs per step (9, float64): AR(5)-whitened innovation (normal equations on the context) clip +-8, z-PIT midrank
    of the innovation vs context innovations, |innovation| clip 8, online flag, log1p(t)/7, log(n_c)/8, log se,
    standardized value clip +-8, z-PIT midrank of the value vs the context values. Sequence = last min(1000, n_c - 5)
    context steps + online.
 R4 Pretraining examples (history only): continuation x_i[n_c : n_c + m], break w.p. 0.5 at tau ~ U{0..m-1}; types
    E1 scaled random splice, E2 raw random splice, H1 neighbour splice, H2 neighbour splice matched in local moments,
    H3 H2 with ramp, P1 own-continuation variance change exp(+-U(.15,.6)), P2 own mean shift +-U(.2,1.0) local sd,
    P3 own MA(1) filter theta +-U(.3,.9) (variance normalised), P4 own tail power |r|^gamma, gamma in U(.45,.8) or
    U(1.3,2.2) (local second moment kept). Curriculum MIX0 -> MIX1. Neighbours: kNN (k=20) on 8 history descriptors.
 R5 Net: GRU(9 -> HDIM, 2 layers, dropout .1) -> emb Linear(HDIM -> 16) -> splice head MLP([emb|x] -> 32 -> 1);
    downstream head: LayerNorm + MLP([h_top | emb | splice logit | x] -> 64 GELU -> 1).
 R6 Pretrain: EPOCHS_PRE epochs x EX_PER_SERIES examples/series, batch 128, AdamW LR_PRE/1e-4, warmup 1 epoch +
    cosine, clip 1, AMP fp16 on CUDA, best epoch by inner pseudo loss.
 R7 LP: encoder frozen, downstream head trained LP_EPOCHS epochs (AdamW 2e-3, cosine). FT: everything, FT_EPOCHS epochs,
    AdamW (head 5e-4, encoder 1.5e-4), warmup + cosine, clip 1. Loss = per-step BCE with pair-count weights
    (positive rows n_neg(t), negative rows n_pos(t); counts over the fit's train rows; mean row weight 1; loss
    normalised by a global constant = rows per batch). Best epoch of each stage by inner TS-AUC (the 10 % inner
    train series).
 R8 Outputs (float64 inference, exact GRU recurrence): LP and FT logits for every online point; per-series offsets
    d_LP, d_FT = mean logit over up to 3 pseudo-online windows of 250 history points (context = history before the
    window) and t in [25, 250).  The released score is FT - b * d_FT (see ``calibrate.py``).

Variants (``variants.py``) replace some of these module-level functions before a fit; ``train_full.py`` runs the
full-data fit.
"""
from __future__ import annotations

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import hashlib  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

SEED = 20260927
P = 5
H_KEEP = 1000
N_C_MIN = 600
CONTEXT_SPAN = 1000
M_MIN, M_MAX = 10, 999
W_LOC = 200
RAMP = (20, 200)
SCALE_LOG = (0.3, 0.9)
KNN = 20
P_BREAK = 0.5
NTYPES = 9
MIX0 = np.array([.15, .15, .10, .08, .07, .15, .12, .09, .09])
MIX1 = np.array([.05, .05, .12, .15, .13, .15, .12, .12, .11])
MIX_VAL = np.full(NTYPES, 1.0 / NTYPES)
F_IN, EMB, HH, DH = 9, 16, 32, 64
HDIM = 192
EPOCHS_PRE = 36
EX_PER_SERIES = 6
VAL_EX_PER_SERIES = 3
BATCH, MICRO, L_BUCKET = 128, 64, 256
LR_PRE, WD, CLIP = 2e-3, 1e-4, 1.0
INNER_FRAC = 0.10
LP_EPOCHS, FT_EPOCHS = 6, 7
LP_LR, FT_LR_HEAD, FT_LR_ENC = 2e-3, 5e-4, 1.5e-4
FT_BATCH = 64
N_WIN, WIN, T_CAL = 3, 250, 25
EVAL_CHUNK = 256
INFER_CHUNK = 128
IDS_SHA = 'c06d4fc1515165211df478a69b5ec792725e8ea659404b85f668dc90504285c5'
T0 = time.time()


def log(*a):
    msg = ' '.join(str(v) for v in a)
    print(f'[{time.time() - T0:8.1f}s {time.strftime("%H:%M:%S", time.gmtime())}Z] {msg}', flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------------- data
def load_data(data_dir: Path, want=None):
    """8000 search series in ascending-id order. want = optional set of positions to load (smoke); the others
    get inert placeholders that no code path of the smoke touches."""
    import pyarrow.parquet as pq
    yi = pq.read_table(data_dir / 'y_train_index.search.parquet', columns=['id']).column('id').to_numpy()
    ids = np.sort(yi.astype(np.int64))
    assert len(ids) == 8000 and len(np.unique(ids)) == 8000
    assert hashlib.sha256(','.join(str(int(v)) for v in ids).encode('ascii')).hexdigest() == IDS_SHA, 'ids != search'
    flt = None if want is None else [('id', 'in', [int(ids[r]) for r in sorted(want)])]
    X = pq.read_table(data_dir / 'X_train.search.parquet', columns=['id', 'time', 'period', 'value'], filters=flt)
    Y = pq.read_table(data_dir / 'y_train.search.parquet', columns=['id', 'time', 'target'], filters=flt)
    xid = X.column('id').to_numpy().astype(np.int64)
    xt = X.column('time').to_numpy().astype(np.int64)
    o = np.lexsort((xt, xid))
    xid, xt = xid[o], xt[o]
    per = X.column('period').to_numpy()[o]
    val = X.column('value').to_numpy()[o].astype(np.float64)
    del X
    yid = Y.column('id').to_numpy().astype(np.int64)
    ytt = Y.column('time').to_numpy().astype(np.int64)
    o2 = np.lexsort((ytt, yid))
    yid = yid[o2]
    yt = Y.column('target').to_numpy()[o2].astype(np.int8)
    del Y
    xb = np.flatnonzero(np.r_[True, xid[1:] != xid[:-1], True])
    yb = np.flatnonzero(np.r_[True, yid[1:] != yid[:-1], True])
    xpos = {int(v): c for c, v in enumerate(xid[xb[:-1]])}
    ypos = {int(v): c for c, v in enumerate(yid[yb[:-1]])}
    hist, onl, lab = [], [], []
    for r in range(8000):
        if want is not None and r not in want:
            hist.append(np.zeros(1000))
            onl.append(np.zeros(10))
            lab.append(np.zeros(10, np.int8))
            continue
        c = xpos[int(ids[r])]
        a, b = xb[c], xb[c + 1]
        p = per[a:b]
        assert np.all(np.diff(p.astype(np.int64)) >= 0), 'period order'
        h = val[a:b][p == 1]
        on = val[a:b][p == 2]
        cy = ypos[int(ids[r])]
        yy = yt[yb[cy]:yb[cy + 1]]
        assert len(yy) == len(on) and len(h) >= 1000
        hist.append(h)
        onl.append(on)
        lab.append(yy)
    return ids, hist, onl, lab


# --------------------------------------------------------------------------------------------------- torch parts
def torch_mod():
    import torch
    import torch.nn as nn
    return torch, nn


def build_features(C, nc, O, m, feat_dtype, hkeep=H_KEEP):
    """C [B, Cm] context left-aligned (0 pad), nc [B]; O [B, Mm] online, m [B]. float64 in, feat_dtype out.
    Returns X [B, L, F_IN], on [B, L] (valid online steps), hl [B] (context steps in the sequence)."""
    torch, _ = torch_mod()
    dev = C.device
    B, Cm = C.shape
    Mm = O.shape[1]
    ncf = nc.to(torch.float64)
    cmask = torch.arange(Cm, device=dev)[None, :] < nc[:, None]
    mu = (C * cmask).sum(1) / ncf
    sd = torch.sqrt((((C - mu[:, None]) * cmask) ** 2).sum(1) / ncf)
    C = torch.where(cmask, (C - mu[:, None]) / sd[:, None], torch.zeros((), dtype=C.dtype, device=dev))
    omask = torch.arange(Mm, device=dev)[None, :] < m[:, None]
    O = torch.where(omask, (O - mu[:, None]) / sd[:, None], torch.zeros((), dtype=O.dtype, device=dev))
    W = C.unfold(1, P + 1, 1)
    y = W[:, :, P]
    X = W[:, :, :P].flip(-1)
    N = nc - P
    Nf = N.to(torch.float64)
    smask = torch.arange(Cm - P, device=dev)[None, :] < N[:, None]
    Xm = X * smask[..., None]
    XtX = torch.einsum('bsk,bsl->bkl', Xm, X)
    Xty = torch.einsum('bsk,bs->bk', Xm, y)
    phi = torch.linalg.solve(XtX, Xty)
    eh = y - torch.einsum('bsk,bk->bs', X, phi)
    em = (eh * smask).sum(1) / Nf
    se = torch.sqrt((((eh - em[:, None]) * smask) ** 2).sum(1) / Nf) + 1e-12
    uh = eh / se[:, None]
    inf = torch.full((), float('inf'), dtype=uh.dtype, device=dev)
    srt = torch.sort(torch.where(smask, uh, inf), dim=1).values
    srtx = torch.sort(torch.where(cmask, C, inf), dim=1).values
    tail = torch.gather(C, 1, N[:, None] + torch.arange(P, device=dev)[None, :])
    ZZ = torch.cat([tail, O], 1)
    WZ = ZZ.unfold(1, P + 1, 1)
    u = (WZ[:, :, P] - torch.einsum('bsk,bk->bs', WZ[:, :, :P].flip(-1), phi)) / se[:, None]
    hl = torch.clamp(N, max=hkeep)
    pos = torch.arange(hkeep, device=dev)[None, :]
    hidx = torch.clamp(N[:, None] - hl[:, None] + pos, max=Cm - P - 1)
    hu = torch.gather(uh, 1, hidx)
    hz = torch.gather(C, 1, hidx + P)

    def pit(tab, cnt, v):
        v = v.contiguous()
        lo = torch.searchsorted(tab, v, side='left')
        hi = torch.searchsorted(tab, v, side='right')
        return torch.special.ndtri((lo + hi + 1).to(torch.float64) / (2.0 * (cnt[:, None] + 1.0)))
    lognc = (torch.log(ncf) / 8.0)[:, None]
    logse = torch.log(se)[:, None]
    z_h = torch.zeros_like(hu)
    ctxF = torch.stack([hu.clamp(-8, 8), pit(srt, Nf, hu), hu.abs().clamp(max=8), z_h, z_h,
                        lognc.expand_as(hu), logse.expand_as(hu), hz.clamp(-8, 8), pit(srtx, ncf, hz)], -1)
    tpos = torch.log1p(torch.arange(Mm, device=dev, dtype=torch.float64))[None, :].expand(B, Mm) / 7.0
    onF = torch.stack([u.clamp(-8, 8), pit(srt, Nf, u), u.abs().clamp(max=8), torch.ones_like(u), tpos,
                       lognc.expand_as(u), logse.expand_as(u), O.clamp(-8, 8), pit(srtx, ncf, O)], -1)
    allF = torch.cat([ctxF, onF, torch.zeros((B, 1, F_IN), dtype=torch.float64, device=dev)], 1)
    L = hl + m
    Lmax = int(math.ceil(int(L.max()) / L_BUCKET) * L_BUCKET)
    q = torch.arange(Lmax, device=dev)[None, :]
    valid = q < L[:, None]
    src = torch.where(q < hl[:, None], q, hkeep + (q - hl[:, None]))
    src = torch.where(valid, src, torch.full_like(src, hkeep + Mm))
    Xs = torch.gather(allF, 1, src[..., None].expand(-1, -1, F_IN)).to(feat_dtype)
    on = valid & (q >= hl[:, None])
    return Xs, on, hl


class Store:
    def __init__(self, arrays, dev):
        torch, _ = torch_mod()
        self.len = np.array([len(a) for a in arrays], np.int64)
        self.off = np.r_[0, np.cumsum(self.len)[:-1]].astype(np.int64)
        self.V = torch.from_numpy(np.concatenate(arrays).astype(np.float64)).to(dev)
        self.len_t = torch.from_numpy(self.len).to(dev)
        self.off_t = torch.from_numpy(self.off).to(dev)
        self.dev = dev

    def gather(self, sidx, start, length, width):
        torch, _ = torch_mod()
        p = torch.arange(width, device=self.dev)[None, :]
        mask = p < length[:, None]
        idx = self.off_t[sidx][:, None] + start[:, None] + p
        return torch.where(mask, self.V[torch.where(mask, idx, torch.zeros_like(idx))],
                           torch.zeros((), dtype=self.V.dtype, device=self.dev)), mask


# --------------------------------------------------------------------------------------------------- neighbours
def acf(x, lag):
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[lag:], x[:-lag]) / d) if d > 0 else 0.0


def series_desc(h):
    X = np.column_stack([h[P - q - 1:len(h) - q - 1] for q in range(P)])
    phi = np.linalg.lstsq(X, h[P:], rcond=None)[0]
    e = h[P:] - X @ phi
    se = e.std() + 1e-12
    u = (e - e.mean()) / se
    u2 = u * u
    return np.array([acf(h, 1), acf(h, 2), acf(h, 5), np.log(se), float(np.mean(u ** 3)), np.log(np.mean(u2 ** 2)),
                     acf(u2, 1), float(np.mean([acf(u2, g) for g in range(1, 6)]))])


def knn_within(desc, k=KNN):
    from scipy.special import ndtri
    from scipy.stats import rankdata
    n = len(desc)
    Z = np.column_stack([ndtri((rankdata(desc[:, c]) - 0.5) / n) for c in range(desc.shape[1])])
    out = np.empty((n, k), np.int64)
    sq = (Z * Z).sum(1)
    for a in range(0, n, 1024):
        d = sq[a:a + 1024, None] + sq[None, :] - 2.0 * Z[a:a + 1024] @ Z.T
        d[np.arange(d.shape[0]), np.arange(a, a + d.shape[0])] = np.inf
        out[a:a + 1024] = np.argsort(d, axis=1, kind='stable')[:, :k]
    return out


# --------------------------------------------------------------------------------------------------- examples
def gen_specs(rng, group, hlen, knn_local, reps, mix):
    g = np.repeat(np.arange(len(group)), reps)
    n = len(g)
    i = group[g]
    n_i = hlen[i]
    mmax = np.minimum(M_MAX, n_i - N_C_MIN)
    assert (mmax >= M_MIN).all()
    m = rng.integers(M_MIN, mmax + 1)
    lo = np.maximum(N_C_MIN, n_i - m - CONTEXT_SPAN)
    nc = rng.integers(lo, n_i - m + 1)
    brk = rng.random(n) < P_BREAK
    tau = np.where(brk, rng.integers(0, m), -1)
    typ = np.where(brk, rng.choice(NTYPES, size=n, p=mix), -1)
    rj = rng.integers(0, len(group) - 1, size=n)
    rj = rj + (rj >= g)
    kj = knn_local[g, rng.integers(0, KNN, size=n)]
    splice = brk & (typ <= 4)
    jg = np.where(np.isin(typ, (2, 3, 4)), kj, rj)
    jg = np.where(splice, jg, g)
    j = group[jg]
    seg = np.where(splice, m - tau, 1)
    a = rng.integers(0, hlen[j] - seg + 1)
    ramp = rng.integers(RAMP[0], RAMP[1] + 1, size=n)
    sgn = rng.choice([-1.0, 1.0], size=n)
    lsc = rng.uniform(SCALE_LOG[0], SCALE_LOG[1], size=n) * sgn
    scale = np.where(typ == 0, np.exp(lsc), 1.0)
    u1 = rng.random(n)
    gam = np.where(rng.random(n) < 0.5, 0.45 + 0.35 * u1, 1.3 + 0.9 * u1)
    pv = np.select([typ == 5, typ == 6, typ == 7, typ == 8],
                   [np.exp(sgn * (0.15 + 0.45 * u1)), sgn * (0.2 + 0.8 * u1), sgn * (0.3 + 0.6 * u1), gam], 0.0)
    sp = dict(i=i, j=j, m=m, nc=nc, tau=tau, typ=typ, a=a, ramp=ramp, scale=scale, pv=pv)
    assert (nc + m <= n_i).all() and (nc >= N_C_MIN).all() and (a + seg <= hlen[j]).all()
    assert not np.any(splice & (i == j))
    return sp


def build_pseudo(store, sp, sel):
    torch, _ = torch_mod()
    dev = store.dev
    t = {k: torch.from_numpy(np.ascontiguousarray(v[sel])).to(dev) for k, v in sp.items()}
    i, j, m, nc, tau, typ, pv = t['i'], t['j'], t['m'], t['nc'], t['tau'], t['typ'], t['pv']
    Cm, Mm = int(nc.max()), int(m.max())
    zero = torch.zeros_like(nc)
    C, _ = store.gather(i, zero, nc, Cm)
    cont, omask = store.gather(i, nc, m, Mm)
    tt = torch.arange(Mm, device=dev)[None, :]
    brk = tau >= 0
    after = brk[:, None] & (tt >= tau[:, None]) & omask
    splice = brk & (typ <= 4)
    segl = torch.where(splice, m - tau, torch.ones_like(m))
    segr, _ = store.gather(j, t['a'], segl, Mm)
    sidx = torch.clamp(tt - tau[:, None], min=0)
    seg = torch.gather(segr, 1, torch.clamp(sidx, max=Mm - 1)) * t['scale'][:, None]
    wl = torch.full_like(nc, W_LOC)
    iw, _ = store.gather(i, nc + torch.clamp(tau, min=0) - W_LOC, wl, W_LOC)
    a0 = torch.minimum(t['a'], store.len_t[j] - W_LOC)
    jw, _ = store.gather(j, a0, wl, W_LOC)
    mu_i, sd_i = iw.mean(1), iw.std(1, unbiased=False) + 1e-12
    mu_j, sd_j = jw.mean(1), jw.std(1, unbiased=False) + 1e-12
    matched = (typ == 3) | (typ == 4)
    seg = torch.where(matched[:, None], (seg - mu_j[:, None]) / sd_j[:, None] * sd_i[:, None] + mu_i[:, None], seg)
    one = torch.ones((), dtype=torch.float64, device=dev)
    lam = torch.where((typ == 4)[:, None], torch.clamp((tt - tau[:, None] + 1).to(torch.float64)
                                                      / t['ramp'][:, None].to(torch.float64), 0.0, 1.0), one)
    mix = mu_i[:, None] + ((1 - lam) * (cont - mu_i[:, None]) + lam * (seg - mu_i[:, None])) / torch.sqrt(
        (1 - lam) ** 2 + lam ** 2)
    O = torch.where(after & (typ == 4)[:, None], mix, torch.where(after & splice[:, None], seg, cont))
    # parametric changes of the own continuation (P1..P4)
    d = cont - mu_i[:, None]
    pvv = pv[:, None]
    o1 = mu_i[:, None] + d * pvv
    o2 = cont + pvv * sd_i[:, None]
    prev0 = C[torch.arange(len(nc), device=dev), nc - 1][:, None]
    dprev = torch.cat([prev0 - mu_i[:, None], d[:, :-1]], 1)
    o3 = mu_i[:, None] + (d + pvv * dprev) / torch.sqrt(1.0 + pvv * pvv)
    r = d / sd_i[:, None]
    rw = (iw - mu_i[:, None]) / sd_i[:, None]
    cg = torch.sqrt((rw.abs() ** (2.0 * pvv)).mean(1, keepdim=True).clamp(min=1e-12))
    o4 = mu_i[:, None] + sd_i[:, None] * torch.sign(r) * r.abs() ** pvv / cg
    tp = typ[:, None]
    par = torch.where(tp == 5, o1, torch.where(tp == 6, o2, torch.where(tp == 7, o3, o4)))
    O = torch.where(after & (tp >= 5), par, O)
    O = torch.where(omask, O, torch.zeros((), dtype=O.dtype, device=dev))
    return C, nc, O, m, after.to(torch.float32), typ


def pseudo_batch(store, sp, sel, feat_dtype):
    torch, _ = torch_mod()
    C, nc, O, m, Yp, typ = build_pseudo(store, sp, sel)
    X, on, hl = build_features(C, nc, O, m, feat_dtype)
    q = torch.arange(X.shape[1], device=X.device)[None, :]
    ti = torch.clamp(q - hl[:, None], min=0, max=Yp.shape[1] - 1)
    Yseq = torch.where(on, torch.gather(Yp, 1, ti), torch.zeros((), device=X.device))
    return X, Yseq, on, hl, typ


def batches_by_len(L, bs, rng=None):
    order = np.argsort(L, kind='stable')
    chunks = [order[a:a + bs] for a in range(0, len(order), bs)]
    if rng is not None:
        rng.shuffle(chunks)
    return chunks


def spec_len(sp):
    return np.minimum(H_KEEP, sp['nc'] - P) + sp['m']


# --------------------------------------------------------------------------------------------------- model
def make_net():
    torch, nn = torch_mod()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(F_IN, HDIM, num_layers=2, batch_first=True, dropout=0.1)
            self.emb = nn.Linear(HDIM, EMB)
            self.head = nn.Sequential(nn.Linear(EMB + F_IN, HH), nn.GELU(), nn.Linear(HH, 1))
            self.dnorm = nn.LayerNorm(HDIM + EMB + 1 + F_IN)
            self.dhead = nn.Sequential(nn.Linear(HDIM + EMB + 1 + F_IN, DH), nn.GELU(), nn.Linear(DH, 1))

        def forward(self, x, down=False):
            h, _ = self.gru(x)
            e = self.emb(h)
            pl = self.head(torch.cat([e, x], -1))
            if not down:
                return pl.squeeze(-1), e
            z = torch.cat([h, e, pl, x], -1)
            return self.dhead(self.dnorm(z)).squeeze(-1), pl.squeeze(-1)

        def downstream(self, h, x):
            e = self.emb(h)
            pl = self.head(torch.cat([e, x], -1))
            z = torch.cat([h, e, pl, x], -1)
            return self.dhead(self.dnorm(z)).squeeze(-1), pl.squeeze(-1)
    return Net()


# --------------------------------------------------------------------------------------------------- metric
def ts_auc_flat(t, s, y):
    t = np.asarray(t, np.int64)
    s = np.asarray(s, np.float64)
    y = np.asarray(y, np.int64)
    order = np.lexsort((s, t))
    ts_, ss, ls = t[order], s[order], y[order].astype(np.float64)
    n = len(ts_)
    new_grp = np.empty(n, bool)
    new_grp[0] = True
    np.not_equal(ts_[1:], ts_[:-1], out=new_grp[1:])
    grp_first = np.flatnonzero(new_grp)
    grp_id = np.cumsum(new_grp) - 1
    pos_in_grp = np.arange(n) - grp_first[grp_id]
    new_blk = new_grp.copy()
    new_blk[1:] |= ss[1:] != ss[:-1]
    blk_first = np.flatnonzero(new_blk)
    blk_id = np.cumsum(new_blk) - 1
    blk_sizes = np.diff(np.append(blk_first, n))
    ranks = (pos_in_grp[blk_first] + 1.0 + (blk_sizes - 1) / 2.0)[blk_id]
    ng = int(grp_id[-1]) + 1
    n_all = np.bincount(grp_id, minlength=ng).astype(float)
    n_pos = np.bincount(grp_id, weights=ls, minlength=ng)
    n_neg = n_all - n_pos
    prs = np.bincount(grp_id, weights=ranks * ls, minlength=ng)
    w = n_pos * n_neg
    mk = w > 0
    auc = (prs[mk] - n_pos[mk] * (n_pos[mk] + 1.0) / 2.0) / w[mk]
    return float(np.dot(w[mk], auc) / w[mk].sum())


# --------------------------------------------------------------------------------------------------- stages
def pretrain(net, store, grp_pt, grp_iv, knn_pt, knn_iv, s, k, dev, amp, feat_dtype, epochs, reps):
    torch, nn = torch_mod()
    opt = torch.optim.AdamW([p for n_, p in net.named_parameters() if not n_.startswith('d')], lr=LR_PRE,
                            weight_decay=WD)
    nb = math.ceil(len(grp_pt) * reps / BATCH)
    total, warm = epochs * nb, nb

    def lr_lambda(step):
        return min(1.0, (step + 1) / warm) * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    lossf = nn.BCEWithLogitsLoss(reduction='none')
    sp_iv = gen_specs(np.random.default_rng([SEED, s, k, 999]), grp_iv, store.len, knn_iv, VAL_EX_PER_SERIES, MIX_VAL)
    chunks_iv = batches_by_len(spec_len(sp_iv), BATCH)
    best, best_ep, best_state, hist = np.inf, -1, None, []
    t0 = time.time()
    for ep in range(epochs):
        torch.manual_seed(SEED + 10 * s + k + 7919 * (ep + 1))
        mix = MIX0 + (MIX1 - MIX0) * ep / max(1, epochs - 1)
        rng = np.random.default_rng([SEED, s, k, ep])
        sp = gen_specs(rng, grp_pt, store.len, knn_pt, reps, mix)
        norm = float(sp['m'].sum()) / nb
        net.train()
        tot, te = 0.0, time.time()
        for ch in batches_by_len(spec_len(sp), BATCH, rng):
            opt.zero_grad(set_to_none=True)
            for mb in np.array_split(ch, math.ceil(len(ch) / MICRO)):
                X, Yq, on, _, _ = pseudo_batch(store, sp, mb, feat_dtype)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    lo, _ = net(X)
                loss = (lossf(lo.float(), Yq) * on).sum() / norm
                scaler.scale(loss).backward()
                tot += float(loss)
                del X, Yq, on, lo, loss
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
        net.eval()
        num = den = 0.0
        with torch.no_grad():
            for ch in chunks_iv:
                X, Yq, on, _, _ = pseudo_batch(store, sp_iv, ch, feat_dtype)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    lo, _ = net(X)
                num += float((lossf(lo.float(), Yq) * on).sum())
                den += float(on.sum())
        vl = num / den
        hist.append(dict(ep=ep, train=tot / nb, inner=vl, secs=round(time.time() - te, 1)))
        log(f'  pre s{s}k{k} ep{ep:2d} train {tot / nb:.5f} inner {vl:.5f} ({time.time() - te:.1f}s; '
            f'total {time.time() - t0:.0f}s)')
        if np.isfinite(vl) and vl < best - 1e-7:
            best, best_ep = vl, ep
            best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
    assert best_state is not None
    net.load_state_dict(best_state)
    return dict(best_ep=best_ep, best_inner=best, hist=hist, secs=time.time() - t0)


def pair_weights(tt_rows, y_rows):
    nt = np.bincount(tt_rows, minlength=1000).astype(float)
    npos = np.bincount(tt_rows, weights=y_rows, minlength=1000)
    nneg = nt - npos
    return npos, nneg


class RealData:
    """Real series on the device: histories, onlines, labels (float32)."""

    def __init__(self, hist, onl, lab, dev):
        self.HS = Store(hist, dev)
        self.OS = Store(onl, dev)
        self.YS = Store([v.astype(np.float64) for v in lab], dev)
        self.dev = dev

    def batch(self, b, feat_dtype, wpos=None, wneg=None):
        torch, _ = torch_mod()
        bt = torch.from_numpy(np.asarray(b, np.int64)).to(self.dev)
        zero = torch.zeros(len(b), dtype=torch.int64, device=self.dev)
        nc = self.HS.len_t[bt]
        m = self.OS.len_t[bt]
        C, _ = self.HS.gather(bt, zero, nc, int(nc.max()))
        O, _ = self.OS.gather(bt, zero, m, int(m.max()))
        X, on, hl = build_features(C, nc, O, m, feat_dtype)
        q = torch.arange(X.shape[1], device=X.device)[None, :]
        Mm = int(m.max())
        ti = torch.clamp(q - hl[:, None], min=0, max=Mm - 1)
        Yo, _ = self.YS.gather(bt, zero, m, Mm)
        Yseq = torch.where(on, torch.gather(Yo, 1, ti), torch.zeros((), dtype=Yo.dtype, device=X.device)).float()
        W = None
        if wpos is not None:
            wp = torch.from_numpy(wpos).to(self.dev)[ti]
            wn = torch.from_numpy(wneg).to(self.dev)[ti]
            W = torch.where(on, torch.where(Yseq > 0.5, wp, wn), torch.zeros((), dtype=wp.dtype,
                                                                               device=X.device)).float()
        return X, Yseq, on, hl, W, ti


def eval_inner(net, rd, idx, feat_dtype, amp):
    """Downstream logit (float32/AMP) on the online steps of idx; TS-AUC against the labels."""
    torch, _ = torch_mod()
    net.eval()
    L = np.minimum(H_KEEP, rd.HS.len[idx] - P) + rd.OS.len[idx]
    ss, tt, yy = [], [], []
    with torch.no_grad():
        for ch in batches_by_len(L, EVAL_CHUNK):
            b = idx[ch]
            X, Yseq, on, hl, _, ti = rd.batch(b, feat_dtype)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                lo, _ = net(X, down=True)
            onn = on.cpu().numpy()
            ss.append(lo.float().cpu().numpy()[onn])
            tt.append(ti.cpu().numpy()[onn])
            yy.append(Yseq.cpu().numpy()[onn])
    return ts_auc_flat(np.concatenate(tt), np.concatenate(ss), np.concatenate(yy).astype(np.int64))


def supervised(net, rd, rows_fit_series, iv_series, wpos, wneg, s, k, dev, amp, feat_dtype, stage):
    torch, nn = torch_mod()
    if stage == 'lp':
        for n_, p in net.named_parameters():
            p.requires_grad_(n_.startswith('d'))
        groups = [dict(params=[p for n_, p in net.named_parameters() if n_.startswith('d')], lr=LP_LR)]
        epochs = LP_EPOCHS
    else:
        for p in net.parameters():
            p.requires_grad_(True)
        groups = [dict(params=[p for n_, p in net.named_parameters() if n_.startswith('d')], lr=FT_LR_HEAD),
                  dict(params=[p for n_, p in net.named_parameters() if not n_.startswith('d')], lr=FT_LR_ENC)]
        epochs = FT_EPOCHS
    opt = torch.optim.AdamW(groups, weight_decay=WD)
    L = np.minimum(H_KEEP, rd.HS.len[rows_fit_series] - P) + rd.OS.len[rows_fit_series]
    nb = math.ceil(len(rows_fit_series) / FT_BATCH)
    norm = float(rd.OS.len[rows_fit_series].sum()) / nb      # global constant (mean row weight = 1)
    total, warm = epochs * nb, max(1, nb // 2)

    def lr_lambda(step):
        w = min(1.0, (step + 1) / warm) if stage == 'ft' else 1.0
        return w * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    lossf = nn.BCEWithLogitsLoss(reduction='none')
    best, best_ep, best_state, hist = -np.inf, -1, None, []
    t0 = time.time()
    for ep in range(epochs):
        torch.manual_seed(SEED + 300 + 10 * s + k + 7919 * (ep + 1) + (0 if stage == 'lp' else 55))
        rng = np.random.default_rng([SEED, s, k, 400 + ep, 0 if stage == 'lp' else 1])
        if stage == 'lp':
            net.eval()
        else:
            net.train()
        tot, te = 0.0, time.time()
        for ch in batches_by_len(L, FT_BATCH, rng):
            b = rows_fit_series[ch]
            X, Yseq, on, hl, W, _ = rd.batch(b, feat_dtype, wpos, wneg)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                if stage == 'lp':
                    with torch.no_grad():
                        h, _ = net.gru(X)
                    lo, _ = net.downstream(h, X)
                else:
                    lo, _ = net(X, down=True)
            loss = (lossf(lo.float(), Yseq) * W).sum() / norm
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], CLIP)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
        auc = eval_inner(net, rd, iv_series, feat_dtype, amp)
        hist.append(dict(ep=ep, train=tot / nb, inner_tsauc=auc, secs=round(time.time() - te, 1)))
        log(f'  {stage} s{s}k{k} ep{ep} train {tot / nb:.5f} inner TS-AUC {auc:.5f} ({time.time() - te:.1f}s)')
        if auc > best + 1e-9:
            best, best_ep = auc, ep
            best_state = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    for p in net.parameters():
        p.requires_grad_(True)
    return dict(best_ep=best_ep, best_inner_tsauc=best, hist=hist, secs=time.time() - t0)


# --------------------------------------------------------------------------------------------------- f64 inference
def gru64(net, X):
    """Exact float64 GRU recurrence (same equations as torch.nn.GRU), batch-first. Returns top-layer h [B, L, H]."""
    torch, _ = torch_mod()
    g = net.gru
    inp = X
    for lyr in range(2):
        wi = getattr(g, f'weight_ih_l{lyr}')
        wh = getattr(g, f'weight_hh_l{lyr}')
        bi = getattr(g, f'bias_ih_l{lyr}')
        bh = getattr(g, f'bias_hh_l{lyr}')
        B, L, _ = inp.shape
        H = wh.shape[1]
        gi = torch.matmul(inp, wi.T) + bi
        h = torch.zeros(B, H, dtype=torch.float64, device=inp.device)
        out = torch.empty(B, L, H, dtype=torch.float64, device=inp.device)
        for st in range(L):
            gh = torch.matmul(h, wh.T) + bh
            r = torch.sigmoid(gi[:, st, :H] + gh[:, :H])
            z = torch.sigmoid(gi[:, st, H:2 * H] + gh[:, H:2 * H])
            nn_ = torch.tanh(gi[:, st, 2 * H:] + r * gh[:, 2 * H:])
            h = (1.0 - z) * nn_ + z * h
            out[:, st] = h
        inp = out
    return inp


def infer64(net64, C, nc, O, m):
    """float64 logits (downstream, splice) for the online steps; returns numpy lists per row."""
    torch, _ = torch_mod()
    X, on, hl = build_features(C, nc, O, m, torch.float64)
    Lv = int((hl + m).max())
    X = X[:, :Lv]
    on = on[:, :Lv]
    with torch.no_grad():
        h = gru64(net64, X)
        lo, pl = net64.downstream(h, X)
    return lo, pl, on, hl


def infer_all(net64, rd, idx, want_pl):
    """Downstream (and splice) float64 logits for the real online of series idx (canonical row layout)."""
    torch, _ = torch_mod()
    dev = rd.dev
    res_lo = [None] * len(idx)
    res_pl = [None] * len(idx) if want_pl else None
    L = np.minimum(H_KEEP, rd.HS.len[idx] - P) + rd.OS.len[idx]
    for ch in batches_by_len(L, INFER_CHUNK):
        b = idx[ch]
        bt = torch.from_numpy(b).to(dev)
        zero = torch.zeros(len(b), dtype=torch.int64, device=dev)
        nc = rd.HS.len_t[bt]
        m = rd.OS.len_t[bt]
        C, _ = rd.HS.gather(bt, zero, nc, int(nc.max()))
        O, _ = rd.OS.gather(bt, zero, m, int(m.max()))
        lo, pl, on, hl = infer64(net64, C, nc, O, m)
        lo_np, pl_np, hl_np = lo.cpu().numpy(), pl.cpu().numpy(), hl.cpu().numpy()
        for r, c in enumerate(ch):
            mm = int(rd.OS.len[idx[c]])
            sl = slice(int(hl_np[r]), int(hl_np[r]) + mm)
            res_lo[c] = lo_np[r, sl].copy()
            if want_pl:
                res_pl[c] = pl_np[r, sl].copy()
    return res_lo, res_pl


def offsets(net64, rd, idx):
    """Per-series offset: mean downstream logit over up to N_WIN pseudo-online windows of WIN history points,
    t in [T_CAL, WIN). Context of window j = history[: n - WIN*j]. Uses the history only."""
    torch, _ = torch_mod()
    dev = rd.dev
    n = rd.HS.len[idx]
    K = np.minimum(N_WIN, (n - N_C_MIN) // WIN)
    assert (K >= 1).all()
    ser = np.repeat(idx, K)
    jj = np.concatenate([np.arange(1, kk + 1) for kk in K])
    ncs = rd.HS.len[ser] - WIN * jj
    acc = np.zeros(len(idx))
    pos_of = np.repeat(np.arange(len(idx)), K)
    Lp = np.minimum(H_KEEP, ncs - P) + WIN
    for ch in batches_by_len(Lp, INFER_CHUNK):
        bt = torch.from_numpy(ser[ch]).to(dev)
        nc = torch.from_numpy(ncs[ch]).to(dev)
        zero = torch.zeros(len(ch), dtype=torch.int64, device=dev)
        C, _ = rd.HS.gather(bt, zero, nc, int(nc.max()))
        w = torch.full_like(nc, WIN)
        O, _ = rd.HS.gather(bt, nc, w, WIN)
        lo, _, on, hl = infer64(net64, C, nc, O, w)
        lo_np, hl_np = lo.cpu().numpy(), hl.cpu().numpy()
        for r, c in enumerate(ch):
            v = lo_np[r, int(hl_np[r]) + T_CAL:int(hl_np[r]) + WIN]
            acc[pos_of[c]] += float(v.mean())
    return acc / K, K
