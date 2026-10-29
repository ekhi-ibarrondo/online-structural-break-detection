"""Pretraining variants of the sequence network.

A variant replaces module-level functions of ``sequence_net`` before a fit
(the recipe calls them through their module names, so every stage sees the
replacement):

* ``pointwise`` (broken_or_not): the recipe of ``sequence_net`` unchanged.
* ``richer`` (first stage of which_one_broke): richer self-supervised pretext.
    - Example mix: the splice slots get 87.5 % of the break mass and the
      parametric slots 12.5 %; then half of the no-break examples are promoted
      to generated breaks (type drawn from the splice-slot weights, donor
      drawn like the default recipe: random for E1/E2, kNN neighbour for H1-H3).
    - For every splice-slot break (original or promoted) the continuation after
      tau is no longer a pasted segment of another series: the series' own
      AR(5)-GARCH(1,1) process continues through tau with new parameters taken
      from the donor (level matched to the own local window for H2/H3, linear
      parameter ramp for H3), driven by the series' own innovations mapped
      onto the donor's innovation shape.  See ``fit_models`` / ``gen_continuations``.
* ``richer_scale_blind`` (blind_to_size): ``richer`` + the input series is
  de-volatilised by a causal local scale before the feature builder
  (``scale_blind.py``).
"""
from __future__ import annotations

import numpy as np

from . import scale_blind
from . import sequence_net as J

VARIANTS = ("pointwise", "richer", "richer_scale_blind")

# Originals, captured before any replacement.
_ORIG_SPECS = J.gen_specs
_ORIG_BUILD_PSEUDO = J.build_pseudo
_ORIG_PRETRAIN = J.pretrain

SPLICE_SHARE = 0.875
PROMOTE_NULL_SHARE = 0.5
NQ = 256
BURN = 20
ALPHAS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30)
BETAS = (0.0, 0.30, 0.50, 0.60, 0.70, 0.75, 0.80, 0.84, 0.87, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99)
PERS_MAX = 0.995
RHO_MAX = 0.99
PSI_TERMS = 3000
CLIP_SD = 30.0
ORIG_KEYS = ('i', 'j', 'm', 'nc', 'tau', 'typ', 'a', 'ramp', 'scale', 'pv')
STATE = {'gen_stats': [], 'store': None, 'model': None}


def install(variant: str) -> None:
    """Apply a variant to ``sequence_net`` (call once, before the fit)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if variant == "pointwise":
        return
    J.gen_specs = gen_specs_richer
    J.build_pseudo = build_pseudo_richer
    J.pretrain = pretrain_richer
    if variant == "richer_scale_blind":
        scale_blind.patch(J)


# ------------------------------------------------------------------------------------------ example specs
def richer_specs(rng, group, hlen, knn_local, reps, mix):
    """Default example draws with the splice/parametric re-weighting, then promotion of half of the nulls."""
    mix = np.array(mix, copy=True)
    mix[:5] *= SPLICE_SHARE / mix[:5].sum()
    mix[5:] *= (1 - SPLICE_SHARE) / mix[5:].sum()
    sp = _ORIG_SPECS(rng, group, hlen, knn_local, reps, mix)
    generated = (sp['tau'] < 0) & (rng.random(len(sp['i'])) < PROMOTE_NULL_SHARE)
    rows = np.flatnonzero(generated)
    n = len(rows)
    sp['tau'][rows] = rng.integers(0, sp['m'][rows])
    sp['typ'][rows] = rng.choice(5, n, p=mix[:5] / mix[:5].sum())
    g = np.repeat(np.arange(len(group)), reps)[rows]
    rj = rng.integers(0, len(group) - 1, n)
    rj += rj >= g
    kj = knn_local[g, rng.integers(0, J.KNN, n)]
    sp['j'][rows] = group[np.where(sp['typ'][rows] >= 2, kj, rj)]
    sp['a'][rows] = rng.integers(0, hlen[sp['j'][rows]] - (sp['m'][rows] - sp['tau'][rows]) + 1)
    sp['scale'][rows] = np.where(sp['typ'][rows] == 0,
                                 np.exp(rng.uniform(*J.SCALE_LOG, n) * rng.choice([-1., 1.], n)), 1.)
    sp['generated'] = generated
    return sp


def gen_specs_richer(rng, group, hlen, knn_local, reps, mix):
    sp = richer_specs(rng, group, hlen, knn_local, reps, mix)
    rows = np.flatnonzero(sp['generated'])
    sp['gen_row'] = np.full(len(sp['i']), -1, np.int64)
    sp['gen_row'][rows] = np.arange(len(rows))
    sp['_gen_seg'], stats = gen_continuations(sp, rows)
    STATE['gen_stats'].append(stats)
    assert stats['n_nonfinite'] == 0, 'nonfinite generated continuation'
    return sp


def pretrain_richer(*args):
    """Fit the per-series AR-GARCH models on the pretraining store, then run the default pretraining."""
    store = args[1]
    STATE['store'] = store
    STATE['model'], stats = fit_models(store)
    J.log(f'per-series AR-GARCH models: {stats}')
    try:
        return _ORIG_PRETRAIN(*args)
    finally:
        STATE['store'] = STATE['model'] = None


# ------------------------------------------------------------------------------------------ per-series models
def fit_models(store):
    """AR(5)-GARCH(1,1) per store row on its full history (float64, device of the store).

    Mean mu, AR(5) by least squares on the demeaned history (companion spectral
    radius capped at 0.99 by phi_k *= c^k), GARCH(1,1) on the AR residuals by
    Gaussian QMLE on a fixed (alpha, beta) grid with variance targeting
    (omega = v (1 - alpha - beta)), the filtered variance path H_t, the
    standardised innovations Z_t = E_t / sqrt(H_t) and a 256-point quantile
    grid of Z (t >= 25).  Returns tensors (flat paths in the store layout) and
    summary statistics.
    """
    torch, _ = J.torch_mod()
    dev = store.dev
    f64 = torch.float64
    p = J.P
    S = len(store.len)
    lens = store.len_t
    Lm = int(store.len.max())
    pos = torch.arange(Lm, device=dev)[None, :]
    mask = pos < lens[:, None]
    idx = torch.where(mask, store.off_t[:, None] + pos, torch.zeros_like(pos))
    zero = torch.zeros((), dtype=f64, device=dev)
    X = torch.where(mask, store.V[idx], zero)
    del idx
    n = lens.to(f64)
    mu = X.sum(1) / n
    D = torch.where(mask, X - mu[:, None], zero)
    del X
    tmask = mask[:, p:]
    tm = tmask.to(f64)
    lag = [D[:, p - q:Lm - q] for q in range(p + 1)]
    XtX = torch.empty((S, p, p), dtype=f64, device=dev)
    Xty = torch.empty((S, p), dtype=f64, device=dev)
    for a in range(1, p + 1):
        Xty[:, a - 1] = (lag[0] * lag[a] * tm).sum(1)
        for b in range(a, p + 1):
            v = (lag[a] * lag[b] * tm).sum(1)
            XtX[:, a - 1, b - 1] = v
            XtX[:, b - 1, a - 1] = v
    phi = torch.linalg.solve(XtX, Xty)
    # stationarity cap on the host (small, deterministic): companion spectral radius <= RHO_MAX
    ph = phi.cpu().numpy()
    comp = np.zeros((S, p, p))
    comp[:, 0, :] = ph
    comp[:, np.arange(1, p), np.arange(p - 1)] = 1.0
    rho = np.abs(np.linalg.eigvals(comp)).max(1)
    c = np.where(rho > RHO_MAX, RHO_MAX / np.maximum(rho, 1e-300), 1.0)
    ph = ph * c[:, None] ** np.arange(1, p + 1)[None, :]
    phi = torch.from_numpy(ph).to(dev)
    pred = sum(phi[:, q - 1:q] * lag[q] for q in range(1, p + 1))
    E = torch.zeros_like(D)
    E[:, p:] = torch.where(tmask, lag[0] - pred, zero)
    del pred, lag, D, tm
    v = (E * E).sum(1) / (n - p)
    v = torch.clamp(v, min=1e-12)
    grid = [(a, b) for a in ALPHAS for b in BETAS if a + b <= PERS_MAX]
    ga = torch.tensor([g[0] for g in grid], dtype=f64, device=dev)[None, :]
    gb = torch.tensor([g[1] for g in grid], dtype=f64, device=dev)[None, :]
    om = v[:, None] * (1.0 - ga - gb)
    h = v[:, None].expand(S, len(grid)).clone()
    nll = torch.zeros((S, len(grid)), dtype=f64, device=dev)
    for t in range(p, Lm):
        if t > p:
            e1 = E[:, t - 1:t]
            h = om + ga * e1 * e1 + gb * h
        if t >= p + BURN:
            et = E[:, t:t + 1]
            nll = nll + torch.where(mask[:, t:t + 1], torch.log(h) + et * et / h, zero)
    best = torch.argmin(nll, dim=1)
    al = ga[0, best]
    be = gb[0, best]
    omg = v * (1.0 - al - be)
    H = v[:, None].expand(S, Lm).clone()
    hh = v.clone()
    for t in range(p + 1, Lm):
        hh = omg + al * E[:, t - 1] ** 2 + be * hh
        H[:, t] = hh
    H = torch.where(mask, H, v[:, None])
    Z = torch.where(mask & (pos >= p), E / torch.sqrt(H), zero)
    Hf, Ef, Zf = H[mask], E[mask], Z[mask]
    assert Hf.numel() == store.V.numel()
    # innovation quantile grid on t in [p + BURN, len)
    valid = mask & (pos >= p + BURN)
    inf = torch.full((), float('inf'), dtype=f64, device=dev)
    srt = torch.sort(torch.where(valid, Z, inf), dim=1).values
    nv = (lens - p - BURN).to(f64)
    pk = (torch.arange(NQ, device=dev, dtype=f64)[None, :] + 0.5) / NQ * nv[:, None] - 0.5
    pk = torch.clamp(pk, min=0.0)
    pk = torch.minimum(pk, (nv - 1.0)[:, None])
    lo = torch.floor(pk).to(torch.int64)
    hi = torch.minimum(lo + 1, (lens - p - BURN - 1)[:, None])
    w = pk - lo.to(f64)
    Qz = srt.gather(1, lo) + w * (srt.gather(1, hi) - srt.gather(1, lo))
    del srt, H, E, Z, nll
    # sum of squared MA(inf) weights of the AR part
    buf = torch.zeros((S, p), dtype=f64, device=dev)
    buf[:, 0] = 1.0
    G = torch.ones(S, dtype=f64, device=dev)
    for _ in range(PSI_TERMS):
        pj = (phi * buf).sum(1)
        G = G + pj * pj
        buf = torch.cat([pj[:, None], buf[:, :-1]], 1)
    model = dict(mu=mu, phi=phi, om=omg, al=al, be=be, v=v, G=G, Hf=Hf, Ef=Ef, Zf=Zf, Qz=Qz)
    alc, bec = al.cpu().numpy(), be.cpu().numpy()
    stats = dict(n_series=int(S), grid_size=len(grid), frac_rho_capped=float((rho > RHO_MAX).mean()),
                 alpha_q=np.quantile(alc, [.1, .5, .9]).tolist(), beta_q=np.quantile(bec, [.1, .5, .9]).tolist(),
                 pers_q=np.quantile(alc + bec, [.1, .5, .9]).tolist(), frac_alpha0=float((alc == 0).mean()),
                 G_q=np.quantile(G.cpu().numpy(), [.1, .5, .9]).tolist(),
                 phi1_q=np.quantile(ph[:, 0], [.1, .5, .9]).tolist())
    return model, stats


# ------------------------------------------------------------------------------------------ generator
def qmap(Zc, Gi, Gj):
    """Rank-preserving map of own innovations onto the donor shape: Q_j(F_i(z)) by linear interpolation between the
    two 256-point grids; beyond the own grid, proportional extension."""
    torch, _ = J.torch_mod()
    nq = Gi.shape[1]
    idx = torch.searchsorted(Gi.contiguous(), Zc.contiguous())
    lo = torch.clamp(idx - 1, 0, nq - 2)
    hi = lo + 1
    gl, gh = Gi.gather(1, lo), Gi.gather(1, hi)
    dg = gh - gl
    w = torch.where(dg > 0, (Zc - gl) / torch.where(dg > 0, dg, torch.ones_like(dg)), torch.zeros_like(dg))
    w = torch.clamp(w, 0.0, 1.0)
    jl, jh = Gj.gather(1, lo), Gj.gather(1, hi)
    mid = jl + w * (jh - jl)
    g0, g1 = Gi[:, :1], Gi[:, -1:]
    rb = torch.where(g0 < -1e-6, Gj[:, :1] / torch.where(g0 < -1e-6, g0, -torch.ones_like(g0)), torch.ones_like(g0))
    ra = torch.where(g1 > 1e-6, Gj[:, -1:] / torch.where(g1 > 1e-6, g1, torch.ones_like(g1)), torch.ones_like(g1))
    return torch.where(Zc < g0, Zc * rb, torch.where(Zc > g1, Zc * ra, mid))


def recursion(V, own, tgt, ramp_rows, ramp, p0, L, Zc, Zq):
    """AR-GARCH continuation from flat position p0 (first generated point) for L steps. own/tgt: dicts of per-row
    tensors mu [R], phi [R, P], om, al, be; lambda = 1 except ramp rows ((s + 1) / ramp, capped at 1).
    State at p0: the true last P values of the series, its last residual and its filtered variance (from own['Ef'],
    own['Hf'] flat paths). Returns out [R, Lmax] and the number of clipped points."""
    torch, _ = J.torch_mod()
    f64 = torch.float64
    R = len(L)
    Lmax = int(L.max())
    p = J.P
    lagx = torch.stack([V[p0 - q] for q in range(1, p + 1)], 1)
    eps = own['Ef'][p0 - 1].clone()
    h = own['Hf'][p0 - 1].clone()
    sd_t = torch.sqrt(tgt['om'] / (1.0 - tgt['al'] - tgt['be']) * tgt['G'])
    lo_c, hi_c = tgt['mu'] - CLIP_SD * sd_t, tgt['mu'] + CLIP_SD * sd_t
    out = torch.zeros((R, Lmax), dtype=f64, device=V.device)
    rampf = ramp.to(f64)
    one = torch.ones(R, dtype=f64, device=V.device)
    nclip = 0
    for s in range(Lmax):
        act = s < L
        lam = torch.where(ramp_rows, torch.clamp((s + 1.0) / rampf, max=1.0), one)
        mu_s = own['mu'] + lam * (tgt['mu'] - own['mu'])
        phi_s = own['phi'] + lam[:, None] * (tgt['phi'] - own['phi'])
        om_s = own['om'] + lam * (tgt['om'] - own['om'])
        al_s = own['al'] + lam * (tgt['al'] - own['al'])
        be_s = own['be'] + lam * (tgt['be'] - own['be'])
        mean = mu_s + ((lagx - mu_s[:, None]) * phi_s).sum(1)
        h = om_s + al_s * eps * eps + be_s * h
        z = Zc[:, s] + lam * (Zq[:, s] - Zc[:, s])
        x = mean + torch.sqrt(h) * z
        xc = torch.maximum(torch.minimum(x, hi_c), lo_c)
        nclip += int(((xc != x) & act).sum())
        eps = xc - mean
        lagx = torch.cat([xc[:, None], lagx[:, :-1]], 1)
        out[:, s] = xc
    return out, nclip


def gen_continuations(sp, rows):
    """Generated continuations for the splice-slot break examples ``rows`` of spec dict sp. Returns F
    [len(rows), M_MAX] (value at column t = generated continuation at online step t for t in [tau, m), 0 elsewhere)
    and statistics.

    Target parameters per slot: E1 = donor parameters with the level scaled like E1 (mu' = mu_j * scale,
    omega' = omega_j * scale^2); E2, H1 = donor parameters as fitted; H2 = donor dynamics with level and
    unconditional variance matched to the own window of W_LOC points before tau; H3 = H2 reached through a linear
    parameter ramp.  Generated values are clipped to mu' +- 30 unconditional sd.
    """
    torch, _ = J.torch_mod()
    store, M = STATE['store'], STATE['model']
    dev = store.dev
    f64 = torch.float64
    F = torch.zeros((len(rows), J.M_MAX), dtype=f64, device=dev)
    st = dict(n_rows=int(len(rows)), n_points=0, n_clip=0, n_nonfinite=0)
    if len(rows) == 0:
        return F, st

    def tt(key):
        return torch.from_numpy(np.ascontiguousarray(sp[key][rows])).to(dev)
    i, j, m, nc, tau, typ, ramp, scale = (tt(k_) for k_ in ('i', 'j', 'm', 'nc', 'tau', 'typ', 'ramp', 'scale'))
    assert bool((typ <= 4).all()) and bool((tau >= 0).all()) and bool((i != j).all())
    L = m - tau
    Lmax = int(L.max())
    p0 = store.off_t[i] + nc + tau
    sidx = torch.arange(Lmax, device=dev)[None, :]
    act = sidx < L[:, None]
    zi = torch.where(act, p0[:, None] + sidx, p0[:, None])
    Zc = torch.where(act, M['Zf'][zi], torch.zeros((), dtype=f64, device=dev))
    Zq = qmap(Zc, M['Qz'][i], M['Qz'][j])
    own = dict(mu=M['mu'][i], phi=M['phi'][i], om=M['om'][i], al=M['al'][i], be=M['be'][i], Ef=M['Ef'], Hf=M['Hf'])
    wpos = p0[:, None] - J.W_LOC + torch.arange(J.W_LOC, device=dev)[None, :]
    iw = store.V[wpos]
    mu_loc = iw.mean(1)
    sd_loc = iw.std(1, unbiased=False) + 1e-12
    matched = (typ == 3) | (typ == 4)
    sc = torch.where(typ == 0, scale, torch.ones_like(scale))
    al_j, be_j, G_j = M['al'][j], M['be'][j], M['G'][j]
    tgt = dict(mu=torch.where(matched, mu_loc, M['mu'][j] * sc),
               phi=M['phi'][j], al=al_j, be=be_j, G=G_j,
               om=torch.where(matched, sd_loc ** 2 * (1.0 - al_j - be_j) / G_j, M['om'][j] * sc ** 2))
    out, nclip = recursion(store.V, own, tgt, typ == 4, ramp, p0, L, Zc, Zq)
    fin = torch.isfinite(out) | ~act
    nonfin_rows = ~fin.all(1)
    if bool(nonfin_rows.any()):                      # never expected; fall back to the real continuation of the row
        real = torch.where(act, store.V[zi], torch.zeros((), dtype=f64, device=dev))
        out = torch.where(nonfin_rows[:, None], real, out)
    rr = torch.arange(len(rows), device=dev)[:, None].expand(-1, Lmax)
    cols = tau[:, None] + sidx
    F[rr[act], cols[act]] = out[act]
    st.update(n_points=int(act.sum()), n_clip=int(nclip), n_nonfinite=int(nonfin_rows.sum()))
    return F, st


def build_pseudo_richer(store, sp, sel):
    """Default pseudo-online batch, with the post-tau part of generated rows replaced by their continuation."""
    torch, _ = J.torch_mod()
    C, nc, O, m, after, typ = _ORIG_BUILD_PSEUDO(store, {k_: sp[k_] for k_ in ORIG_KEYS}, sel)
    if '_gen_seg' not in sp:
        return C, nc, O, m, after, typ
    fr = torch.from_numpy(np.ascontiguousarray(sp['gen_row'][sel])).to(store.dev)
    use = fr >= 0
    if bool(use.any()):
        G = torch.zeros_like(O)
        G[use] = sp['_gen_seg'][fr[use], :O.shape[1]]
        O = torch.where((after > 0.5) & use[:, None], G, O)
    return C, nc, O, m, after, typ
