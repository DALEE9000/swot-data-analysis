"""SWOTxAI evolve candidate.

CONTRACT — the harness calls exactly this function; keep the signature:

    train_and_predict(X_train, Y_train, X_test, params) -> np.ndarray

  X_train : float32 (n_train, d) feature matrix; may contain NaN (stencil
            padding at swath edges).
  Y_train : float32 (n_train, 2) targets (u, v) in m/s; NaN marks an invalid
            component (train on the valid entries only).
  X_test  : float32 (n_test, d).
  params  : dict with keys:
              seed          - int, seed all RNGs with this
              device        - "cuda" or "cpu"
              max_epochs    - int, do not train longer than this many epochs
              time_budget_s - float, soft wall-clock budget for training

  Returns predictions for X_test, shape (n_test, 2), finite everywhere.

Everything below the contract may be rewritten freely. Allowed imports:
numpy, torch, math, random, time, copy, typing, dataclasses, collections,
itertools, functools, warnings, sklearn, scipy. No file, network, or OS access.
"""
import copy
import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge


def _cells(d):
    # Columns are k*k spatial-stencil copies of base features, feature-major.
    # Infer the stencil cell count so the appended blocks stay layout-correct.
    for c in (9, 25, 49):
        if d % c == 0:
            return c
    return 1


def _featurize(X, mean, scale, n_cells):
    """Standardize, gradient-fill missing stencil cells, append missingness,
    spatial std, explicit per-feature spatial derivatives (first AND second
    order), and NONLINEAR flow-regime features.

    Proven plumbing (unchanged from the lineage):
      * per-cell missing fraction (n, C);
      * per-feature stencil std over VALID cells (n, F) — front/eddy signal;
      * gradient-preserving mirror imputation (2*center - mirror, clamped to
        +/-4 sigma, center fallback) so swath-edge truncation keeps the
        first-order gradient instead of flattening the neighborhood;
      * per-feature central differences gx, gy (n, 2F);
      * per-feature 5-point Laplacian and the two diagonal central
        differences (n, 3F);
      * |grad f| per feature (n, F);
      * flow-regime block when F == 8 (speeds, vorticity, strains,
      Okubo-Weiss, geostrophic advection).

    The WHOLE flat vector feeds the MoE core; the regime block near the end
    (speeds, vorticity, strains, Okubo-Weiss) plus the missingness block is
    exactly the coordinate system the gate needs to partition rows into
    dynamical regimes."""
    Xs = (X.astype(np.float32) - mean) / scale
    n, d = Xs.shape
    F = d // n_cells
    miss = np.isnan(X).reshape(n, F, n_cells).mean(axis=1).astype(np.float32)
    Z = Xs.reshape(n, F, n_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spat = np.nanstd(Z, axis=2).astype(np.float32)
    spat = np.nan_to_num(spat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    # Fill AFTER the std block (which must see true valid cells). Snapshot the
    # point-reflected cells before any in-place filling so late cells never
    # read an already-imputed mirror as if it were data.
    center = Z[:, :, n_cells // 2]
    mirror = Z[:, :, ::-1].copy()
    for c in range(n_cells):
        if c == n_cells // 2:
            continue
        col = Z[:, :, c]
        hole = np.isnan(col)
        if hole.any():
            lin = 2.0 * center - mirror[:, :, c]
            fill = np.where(np.isfinite(lin), np.clip(lin, -4.0, 4.0), center)
            col[hole] = fill[hole]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    blocks = [Xs, miss, spat]
    k = int(round(math.sqrt(n_cells)))
    if k >= 2 and k * k == n_cells:
        Zf = Xs[:, :F * n_cells].reshape(n, F, n_cells)
        ci = n_cells // 2
        gx = (0.5 * (Zf[:, :, ci + 1] - Zf[:, :, ci - 1])).astype(np.float32)
        gy = (0.5 * (Zf[:, :, ci + k] - Zf[:, :, ci - k])).astype(np.float32)
        blocks += [gx, gy]
        if k >= 3:
            lap = (Zf[:, :, ci + 1] + Zf[:, :, ci - 1] + Zf[:, :, ci + k]
                   + Zf[:, :, ci - k] - 4.0 * Zf[:, :, ci]).astype(np.float32)
            inv2 = np.float32(0.5 / math.sqrt(2.0))
            gp = (inv2 * (Zf[:, :, ci + k + 1] - Zf[:, :, ci - k - 1])).astype(np.float32)
            gq = (inv2 * (Zf[:, :, ci - k + 1] - Zf[:, :, ci + k - 1])).astype(np.float32)
            blocks += [lap, gp, gq]
        gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        blocks.append(gmag)
        if F == 8:
            # Pooled layout: [mdt, ssha, ugos, vgos, ugosa, vgosa,
            # era5_u, era5_v]; center-cell standardized values.
            cu, cv = Zf[:, 2, ci], Zf[:, 3, ci]
            au, av = Zf[:, 4, ci], Zf[:, 5, ci]
            wu, wv = Zf[:, 6, ci], Zf[:, 7, ci]
            spd = np.sqrt(cu * cu + cv * cv)
            spda = np.sqrt(au * au + av * av)
            spdw = np.sqrt(wu * wu + wv * wv)
            vort = gx[:, 3] - gy[:, 2]
            sn = gx[:, 2] - gy[:, 3]
            ss = gx[:, 3] + gy[:, 2]
            ow = np.clip(sn * sn + ss * ss - vort * vort, -16.0, 16.0)
            reg = np.stack([spd, spda, spdw, vort, sn, ss, ow],
                           axis=1).astype(np.float32)
            adv = np.clip(cu[:, None] * gx + cv[:, None] * gy,
                          -16.0, 16.0).astype(np.float32)
            blocks += [reg, adv]
    return np.hstack(blocks)


class RegimeMoE(nn.Module):
    """WILDCARD CORE — soft-gated mixture-of-experts over flow regimes.

    Motivation: the lineage's fast-regime error (rmse 0.227) dwarfs the slow
    bin (0.066), and its late-test u bin lags. A single shared trunk must fit
    quiescent interior rows, front/eddy rows, wind-driven rows, and truncated
    swath-edge rows with the SAME parameters — the fit is a compromise that
    the majority-slow population dominates. Here a shared encoder produces a
    representation z; a linear gate maps z to softmax weights over E=4 small
    expert heads, and the prediction is the gate-weighted sum of expert
    outputs. Each expert can specialize its input-output mapping to one
    dynamical population while the encoder shares low-level features across
    all of them. The engineered input already contains explicit regime
    coordinates (speeds, vorticity, strain, Okubo-Weiss, missing fraction),
    so the gate does not need to discover the partition from raw pixels.

    Gate weight AND bias are zero-initialized: at step 0 the gate is exactly
    uniform, the model behaves as an average of 4 same-shaped heads (i.e. a
    plain MLP with a slightly wider effective head), and gradient descent
    only sharpens the mixture where specialization actually lowers the loss.
    Soft (dense) gating — every expert runs on every row — keeps training
    stable on this small scale; no load-balancing loss is needed because a
    collapsed gate is just the parent-style single trunk, an acceptable
    floor, not a failure mode.

    Heads: mixture mu (ridge-residual predictions) and a shared logvar from
    the encoder (anchored-NLL weighting), preserving the lineage's loss
    interface unchanged."""

    def __init__(self, d_total, n_experts=4, dropout=0.1, logvar_init=-3.5):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_total, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(192, n_experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(192, 128), nn.SiLU(),
                          nn.Linear(128, 2))
            for _ in range(n_experts)
        ])
        self.logvar = nn.Linear(192, 2)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, logvar_init)

    def forward(self, x):
        z = self.enc(x)
        g = torch.softmax(self.gate(z), dim=1)          # (b, E)
        mus = torch.stack([e(z) for e in self.experts], dim=1)  # (b, E, 2)
        mu = (g.unsqueeze(-1) * mus).sum(dim=1)         # (b, 2)
        return mu, self.logvar(z)


def _make_net(d_total, n_feat, n_cells, device, arch):
    """All members are RegimeMoE in this wildcard; arch is kept in the
    signature so the member loop's plumbing is unchanged, and member seeds
    alone differentiate the two ensemble members (init + data order)."""
    return RegimeMoE(d_total, n_experts=4, dropout=0.1).to(device)


def _masked_nll_anchored(mu, logvar, target, mask, row_w, comp_w):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component,
    scaled by a per-row recency weight AND a per-component inverse-variance
    weight (unchanged from the lineage). The score is mean per-component
    R^2 = mean_c(1 - MSE_c/Var_c), so a unit of squared error on the
    lower-variance component (u) buys more R^2 than the same error on v;
    comp_w = (1/Var_u, 1/Var_v) mean-1 normalized makes the loss's exchange
    rate between u and v errors match the metric's. The anchor keeps a
    gradient floor on variance-downweighted rows."""
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    w = mask * row_w.unsqueeze(1) * comp_w.unsqueeze(0)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _masked_mse(pred, target, mask, comp_w):
    """Component-weighted masked MSE. With comp_w proportional to the
    inverse target variances this is (up to a constant) the mean per-
    component R^2 deficit — the scored quantity — so early stopping,
    raw-vs-EMA selection, the LR schedule, and the member guard all
    optimize the metric directly instead of pooled MSE."""
    w = mask * comp_w.unsqueeze(0)
    diff2 = (pred - target).pow(2) * w
    return diff2.sum() / w.sum().clamp(min=1e-8)


class _EMA:
    """Exponential moving average of a net's weights along the SGD
    trajectory (SWA-family); shadow sits in the flatter basin center,
    transfers better across the temporal train->test shift."""

    def __init__(self, net, decay):
        self.decay = decay
        self.net = copy.deepcopy(net)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net):
        for pe, pn in zip(self.net.parameters(), net.parameters()):
            pe.lerp_(pn, 1.0 - self.decay)
        for be, bn in zip(self.net.buffers(), net.buffers()):
            be.copy_(bn)


def _val_mse(net, X_v, Y_v, M_v, bs, comp_w):
    # X_v may be stored fp16 on-device; cast per batch (no-op for fp32).
    net.eval()
    with torch.no_grad():
        losses = []
        for i in range(0, len(X_v), bs):
            mu, _ = net(X_v[i:i + bs].float())
            losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs],
                                      comp_w).item())
    return float(np.mean(losses))


def _train_member(member_seed, arch, deadline, device, max_epochs,
                  n_feat, n_cells, comp_w,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v, W_v):
    """Train one RegimeMoE member with the recency- and component-weighted
    anchored heteroscedastic NLL; selection over raw-vs-EMA on component-
    weighted temporal-val masked MSE (the scored quantity); then the proven
    recency fine-tune ending on a short-horizon EMA. Returns (net, best_val).

    Training loop plumbing is unchanged from the reference: matrices may
    arrive GPU-resident in float16 (see train_and_predict) and are indexed
    on-device with a per-batch fp32 cast; the CPU fallback path makes all
    the `.to(device)` / `.float()` calls no-ops. The fine-tune gathers
    train/val batches from the two tensors directly (virtual concat), no
    full concatenated copy on the 4 GB card."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = _make_net(X_t.shape[1], n_feat, n_cells, device, arch)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)

    ema = _EMA(net, decay=0.999)

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train)).to(X_t.device)
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb = X_t[idx].to(device).float()
            yb = Y_t[idx].to(device)
            mb = M_t[idx].to(device)
            wb = W_t[idx].to(device)
            optimizer.zero_grad()
            mu, lv = net(xb)
            loss = _masked_nll_anchored(mu, lv, yb, mb, wb, comp_w)
            loss.backward()
            optimizer.step()
            ema.update(net)

        val_raw = _val_mse(net, X_v, Y_v, M_v, bs, comp_w)
        val_ema = _val_mse(ema.net, X_v, Y_v, M_v, bs, comp_w)
        scheduler.step(val_raw)
        val_loss = min(val_raw, val_ema)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            src = net if val_raw <= val_ema else ema.net
            best_state = copy.deepcopy(src.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() > deadline:
            break

    net.load_state_dict(best_state)

    # Recency fine-tune (proven +~0.01 in lineage): absorb the held-out
    # most-recent 10% with a short low-LR pass over the full window, ending
    # on a short-horizon EMA of the fine-tune trajectory. Batches are
    # gathered from the train and val tensors separately (virtual concat).
    if time.time() < deadline and np.isfinite(best_val):
        n_t = len(X_t)
        n_all = n_t + len(X_v)
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        ft_ema = _EMA(net, decay=0.998)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                a = idx < n_t
                i1 = idx[a].to(X_t.device)
                i2 = (idx[~a] - n_t).to(X_v.device)
                xb = torch.cat([X_t[i1].to(device).float(),
                                X_v[i2].to(device).float()])
                yb = torch.cat([Y_t[i1].to(device), Y_v[i2].to(device)])
                mb = torch.cat([M_t[i1].to(device), M_v[i2].to(device)])
                wb = torch.cat([W_t[i1].to(device), W_v[i2].to(device)])
                ft_opt.zero_grad()
                mu, lv = net(xb)
                loss = _masked_nll_anchored(mu, lv, yb, mb, wb, comp_w)
                loss.backward()
                ft_opt.step()
                ft_ema.update(net)
                if bi % 100 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break
        net.load_state_dict(ft_ema.net.state_dict())

    net.eval()
    return net, best_val


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    torch.manual_seed(seed)
    device = torch.device(params["device"])
    max_epochs = int(params["max_epochs"])
    time_budget_s = float(params["time_budget_s"])
    t0 = time.time()

    # Per-column standardization ignoring NaNs; NaN -> 0 (the feature mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    n_cells = _cells(X_train.shape[1])
    n_feat = X_train.shape[1] // n_cells
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)
    n = len(Xs)

    # Per-component inverse-variance weights (unchanged from the lineage).
    cw = np.ones(2, dtype=np.float32)
    for c in range(2):
        vc = float(np.var(Y_train[mask[:, c] > 0, c])) if mask[:, c].sum() > 1000 else 0.0
        cw[c] = 1.0 / vc if vc > 1e-8 else 1.0
    cw = np.clip(cw / cw.mean(), 0.5, 2.0).astype(np.float32)
    cw /= cw.mean()
    comp_w = torch.from_numpy(cw).to(device)

    # ---- Linear baseline + residual learning (unchanged) ----
    # Ridge from the full engineered block to (u, v) on a seeded subsample;
    # the nets fit the residual. The linear map extrapolates the dominant
    # geostrophic signal into the fast tail where bounded activations
    # saturate; baseline clipped to +/-3 m/s. Additive per-row baseline
    # means residual val MSE == final-prediction val MSE, so all selection
    # still optimizes the scored quantity.
    rng0 = np.random.default_rng(seed)
    sub = rng0.choice(n, size=min(n, 1_500_000), replace=False)
    W_lin = np.zeros((Xs.shape[1], 2), dtype=np.float32)
    b_lin = np.zeros(2, dtype=np.float32)
    for c in range(2):
        rows = sub[mask[sub, c] > 0]
        if len(rows) > 1000:
            ridge = Ridge(alpha=1.0)
            ridge.fit(Xs[rows], Ys[rows, c])
            W_lin[:, c] = ridge.coef_.astype(np.float32)
            b_lin[c] = np.float32(ridge.intercept_)

    def _baseline(Xf):
        return np.clip(Xf @ W_lin + b_lin, -3.0, 3.0).astype(np.float32)

    base_tr = np.empty((n, 2), dtype=np.float32)
    for i in range(0, n, 1_000_000):
        base_tr[i:i + 1_000_000] = _baseline(Xs[i:i + 1_000_000])
    Ys = np.where(mask > 0, Ys - base_tr, 0.0).astype(np.float32)
    del base_tr

    # Temporal validation split: last 10% of the time-ordered window, so
    # early stopping and the LR schedule optimize forward-in-time
    # generalization — what the temporal test holdout measures.
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights: latest rows weigh 3x the earliest (exponential ramp,
    # mean-1 normalized so the tuned loss scale is unchanged).
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
    w_all = np.exp(np.log(3.0) * pos).astype(np.float32)
    w_all /= w_all.mean()

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    W_t = torch.from_numpy(w_all[train_idx])
    X_v = torch.from_numpy(Xs[val_idx])
    Y_v = torch.from_numpy(Ys[val_idx])
    M_v = torch.from_numpy(mask[val_idx])
    W_v = torch.from_numpy(w_all[val_idx])
    del Xs

    # fp16 GPU staging (unchanged from the reference): full featurized train
    # matrix on-device in float16 (~1.1 GB on the 4 GB card), on-device
    # indexing + per-batch fp32 cast; CPU fallback on OOM reproduces the
    # host-resident pipeline exactly.
    if device.type == "cuda":
        try:
            X_t = X_t.to(device, dtype=torch.float16)
            Y_t = Y_t.to(device)
            M_t = M_t.to(device)
            W_t = W_t.to(device)
            X_v = X_v.to(device, dtype=torch.float16)
        except RuntimeError:
            torch.cuda.empty_cache()
            X_t = X_t.cpu()
            Y_t = Y_t.cpu()
            M_t = M_t.cpu()
            W_t = W_t.cpu()
            X_v = X_v.cpu().to(device)  # fallback: fp32 val on device
        Y_v = Y_v.to(device)
        M_v = M_v.to(device)
        W_v = W_v.to(device)

    # Two-member ensemble of differently-seeded RegimeMoE nets with
    # per-member wall-clock slices. Seeds differ, so init and data order
    # differ; the gate partitions they discover can also differ, which
    # decorrelates errors on the regime boundaries where soft assignments
    # are least certain.
    archs = ["moe", "moe"]
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, archs[m % len(archs)], deadline, device, max_epochs,
            n_feat, n_cells, comp_w,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v, W_v)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse
    # than the best, in the component-weighted metric) so the worst case
    # degenerates to a single MoE instead of averaging in an undertrained net.
    finite = [(net, v) for net, v in members if np.isfinite(v)]
    if finite:
        v_best = min(v for _, v in finite)
        keep = [net for net, v in finite if v <= v_best * 1.15]
    else:
        keep = [members[0][0]]

    # Predict: ridge baseline + averaged residual mu over accepted members.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            Xf = _featurize(X_test[i:i + 65536], mean, scale, n_cells)
            base = _baseline(Xf)
            xb = torch.from_numpy(Xf).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net in keep:
                mu, _ = net(xb)
                acc += mu
            out[i:i + 65536] = base + (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
