"""Scale-blind input (blind_to_size): the series is de-volatilised by a causal local scale before ``build_features``.

The network cannot lean on the variance level and spends its capacity on shape/tail, dependence, trend and mean.

Per row, on x = [C[0:nc], O[0:m]] (context then online):
  mu    = population mean of the context;  d_t = x_t - mu
  v_{-1}= mean over the context of d_t^2 (population variance of the context)
  v_t   = LAM * v_{t-1} + (1 - LAM) * d_t^2,  LAM = 2 ** (-1/50) (half-life 50), context then online (state continues)
  xt_t  = d_t / sqrt(max(v_{t-1}, FLOOR * v_{-1}))  (scale from the strict past; t = 0 uses v_{-1})
The original build_features is then called on (C~, nc, O~, m) unchanged. Online xt at t depends only on the context and
on online values <= t. Implementation: blocked scan (blocks of K steps: closed form inside the block, sequential
chaining of the state between blocks), float64; relative error vs the scalar recursion <= 1e-12.
"""
from __future__ import annotations

HALF_LIFE = 50.0
LAM = 2.0 ** (-1.0 / HALF_LIFE)
FLOOR = 1e-24
BLOCK = 64


def ewma_prev(e, v0, lam=LAM, block=BLOCK):
    """e [B, T] float64 (squared deviations, 0 on padding), v0 [B]. Returns vprev [B, T] with vprev[:, t] = v_{t-1}
    of the recursion v_t = lam v_{t-1} + (1-lam) e_t, v_{-1} = v0."""
    import torch
    B, T = e.shape
    nb = -(-T // block)
    pad = nb * block - T
    if pad:
        e = torch.cat([e, torch.zeros((B, pad), dtype=e.dtype, device=e.device)], 1)
    eb = e.view(B, nb, block)
    j = torch.arange(block, dtype=e.dtype, device=e.device)
    lam_t = torch.tensor(lam, dtype=e.dtype, device=e.device)
    pw = lam_t ** j                     # lam^j
    ipw = lam_t ** (-j)                 # lam^-j
    part = (1.0 - lam) * pw * torch.cumsum(eb * ipw, dim=2)     # contribution of the block's own e's
    lamj1 = pw * lam                    # lam^(j+1)
    v = torch.empty_like(eb)
    s = v0.to(e.dtype)
    for n in range(nb):
        vn = lamj1[None, :] * s[:, None] + part[:, n, :]
        v[:, n, :] = vn
        s = vn[:, -1]
    v = v.reshape(B, nb * block)[:, :T]
    return torch.cat([v0.to(e.dtype)[:, None], v[:, :-1]], 1)


def transform(C, nc, O, m):
    """C [B, Cm] (0 padded beyond nc), O [B, Mm] (0 padded beyond m), float64 tensors. Returns (C~, O~) same shapes."""
    import torch
    dev, dt = C.device, C.dtype
    B, Cm = C.shape
    Mm = O.shape[1]
    T = Cm + Mm
    p = torch.arange(T, device=dev)[None, :].expand(B, T)
    ncl, ml = nc[:, None], m[:, None]
    in_c = p < ncl
    in_o = (p >= ncl) & (p < ncl + ml)
    ic = torch.clamp(p, max=Cm - 1)
    io = torch.clamp(p - ncl, min=0, max=Mm - 1)
    zero = torch.zeros((), dtype=dt, device=dev)
    x = torch.where(in_c, torch.gather(C, 1, ic), torch.where(in_o, torch.gather(O, 1, io), zero))
    ncf = nc.to(dt)
    cmask = torch.arange(Cm, device=dev)[None, :] < ncl
    mu = (C * cmask).sum(1) / ncf
    valid = in_c | in_o
    d = torch.where(valid, x - mu[:, None], zero)
    v0 = (torch.where(cmask, C - mu[:, None], zero) ** 2).sum(1) / ncf
    vprev = ewma_prev(d * d, v0)
    xt = torch.where(valid, d / torch.sqrt(torch.maximum(vprev, (FLOOR * v0)[:, None])), zero)
    Ct = torch.where(cmask, xt[:, :Cm], zero)
    q = torch.arange(Mm, device=dev)[None, :]
    omask = q < ml
    Ot = torch.where(omask, torch.gather(xt, 1, torch.clamp(ncl + q, max=T - 1)), zero)
    return Ct, Ot


def patch(J):
    """Idempotent replacement of ``J.build_features`` (every path: pretrain, LP/FT, f64 inference, offsets)."""
    cur = J.build_features
    if getattr(cur, '_scale_blind', False):
        assert getattr(J, '_build_features_orig', None) is not None
        return J
    assert cur.__name__ == 'build_features' and cur.__module__ == J.__name__, 'build_features is not the original one'
    orig = cur

    def build_features_sb(C, nc, O, m, feat_dtype, hkeep=None):
        Ct, Ot = transform(C, nc, O, m)
        return orig(Ct, nc, Ot, m, feat_dtype, J.H_KEEP if hkeep is None else hkeep)

    build_features_sb._scale_blind = True
    J._build_features_orig = orig
    J.build_features = build_features_sb
    return J


def unpatch(J):
    orig = getattr(J, '_build_features_orig', None)
    if orig is not None:
        J.build_features = orig
        J._build_features_orig = None
    return J
