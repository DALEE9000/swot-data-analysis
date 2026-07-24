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
    order), and nonlinear flow-regime features (proven plumbing, unchanged):

      * per-cell missing fraction (n, C);
      * per-feature stencil std over VALID cells (n, F) — front/eddy signal;
      * gradient-preserving mirror imputation (2*center - mirror, clamped to
        +/-4 sigma, center fallback) so swath-edge truncation keeps the
        first-order gradient instead of flattening the neighborhood;
      * per-feature central differences gx, gy (n, 2F);
      * per-feature 5-point Laplacian and the two diagonal central
        differences (n, 3F);
      * |grad f| per feature (n, F) — front intensity;
      * when F == 8 (pooled layout [mdt, ssha, ugos, vgos, ugosa, vgosa,
        era5_u, era5_v]): center-cell flow speeds |u_g|, |u_ga|, |u_wind|,
        relative vorticity, normal/shear strain, Okubo-Weiss parameter, and
        geostrophic advection of every feature. These are exactly the regime
        descriptors the MoE gate needs to route slow vs fast rows.
    Quadratic terms clipped to +/-16 (inputs sigma-scaled, fills clamped to
    +/-4) so rare outliers do not blow up the first Linear layer.
    """
    Xs = (X.astype(np.float32) - mean) / scale
    n, d = Xs.shape
    F = d // n_cells
    miss = np.isnan(X).reshape(n, F, n_cells).mean(axis=1).astype(np.float32)
    Z = Xs.reshape(n, F, n_cells)
    with warnings.catch_warnings():
        # nanstd emits RuntimeWarning on all-NaN neighborhoods; those rows
        # are handled by the nan_to_num below.
        warnings.simplefilter("ignore")
        spat = np.nanstd(Z, axis=2).astype(np.float32)
    spat = np.nan_to_num(spat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    # Fill AFTER the std block (which must see true valid cells). Snapshot the
    # point-reflected cells before any in-place filling so late cells never
    # read an already-imputed mirror as if it were data.
    center = Z[:, :, n_cells // 2]
    mirror = Z[:, :, ::-1].copy()  # cell c's mirror across the center is C-1-c
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
    # Derivatives on the filled, finite stencil. Cells are flattened
    # row-major over the k x k window: the center's horizontal neighbors sit
    # at +/-1, vertical neighbors at +/-k, diagonals at +/-(k+1) and
    # +/-(k-1). Orientation/sign conventions only need to be consistent.
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


class MoENet(nn.Module):
    """Soft-gated mixture-of-experts over flow regimes (the wildcard core).

    Shared trunk -> softmax gate over E small expert heads. Each expert
    predicts its own (mu, logvar); the model output is the gate-weighted
    mixture mean of the mus (what the harness scores). Rationale: the fast
    regime (fronts/eddies) obeys different local dynamics than the slow
    quiescent background — a single MLP head must average them, and the
    diagnostics show it sacrifices the fast tail (rmse 0.228 fast vs 0.069
    slow). The trunk input already contains explicit regime descriptors
    (speeds, vorticity, strain, Okubo-Weiss), so the gate can partition
    rows by dynamical regime and let one expert specialize on the fast tail
    without dragging the slow-regime fit. Soft gating keeps everything
    differentiable end-to-end and degrades gracefully to the parent's
    single-head behavior if the gate finds no useful partition.
    """

    def __init__(self, n_inputs, trunk=(256, 256), n_experts=4,
                 expert_hidden=128, dropout=0.1, logvar_init=-3.5):
        super().__init__()
        layers = []
        d = n_inputs
        for h in trunk:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.gate = nn.Linear(d, n_experts)
        nn.init.zeros_(self.gate.weight)  # start at uniform gating
        nn.init.zeros_(self.gate.bias)
        self.bodies = nn.ModuleList()
        self.mu_heads = nn.ModuleList()
        self.lv_heads = nn.ModuleList()
        for _ in range(n_experts):
            self.bodies.append(nn.Sequential(
                nn.Linear(d, expert_hidden), nn.LayerNorm(expert_hidden), nn.SiLU()))
            self.mu_heads.append(nn.Linear(expert_hidden, 2))
            lv = nn.Linear(expert_hidden, 2)
            # Start near the marginal target log-variance so the NLL is well
            # scaled from step 1 (u, v std ~0.15-0.2 m/s -> log var ~ -3.5).
            nn.init.zeros_(lv.weight)
            nn.init.constant_(lv.bias, logvar_init)
            self.lv_heads.append(lv)

    def forward(self, x):
        z = self.trunk(x)
        g = torch.softmax(self.gate(z), dim=1)                     # (B, E)
        mus, lvs = [], []
        for body, mh, lh in zip(self.bodies, self.mu_heads, self.lv_heads):
            h = body(z)
            mus.append(mh(h))
            lvs.append(lh(h))
        mus = torch.stack(mus, dim=1)                              # (B, E, 2)
        lvs = torch.stack(lvs, dim=1)                              # (B, E, 2)
        mu_mix = (g.unsqueeze(-1) * mus).sum(dim=1)                # (B, 2)
        return mu_mix, (mus, lvs, g)


def _moe_loss(mu_mix, mus, lvs, g, target, mask, row_w):
    """Gate-routed heteroscedastic NLL + MSE anchor on the mixture mean,
    per valid component, scaled by the per-row recency weight.

    The per-expert Gaussian NLL is weighted by the gate: experts only pay
    for rows routed to them, which is the specialization pressure. The
    plain-MSE anchor is computed on the MIXTURE MEAN — the exact quantity
    the harness scores — so the scored output keeps a direct, unweighted
    gradient on every row (the lesson from the lineage's anchored loss, and
    the trap gen019_00 fell into by realigning the wrong term). A small
    load-balance penalty (squared deviation of mean gate usage from
    uniform, scaled by E) prevents the classic MoE failure of the gate
    collapsing onto one expert in early training.
    """
    lvs = lvs.clamp(-7.0, 2.0)
    err2_e = (mus - target.unsqueeze(1)).pow(2)                    # (B, E, 2)
    nll_e = 0.5 * (lvs + err2_e * torch.exp(-lvs))
    nll = (g.unsqueeze(-1) * nll_e).sum(dim=1)                     # (B, 2)
    anchor = (mu_mix - target).pow(2)
    per = nll + anchor
    w = mask * row_w.unsqueeze(1)
    loss = (per * w).sum() / w.sum().clamp(min=1e-8)
    n_e = g.shape[1]
    balance = n_e * ((g.mean(dim=0) - 1.0 / n_e).pow(2)).sum()
    return loss + 0.02 * balance


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


class _EMA:
    """Exponential moving average of a net's weights along the SGD
    trajectory (SWA-family). Near convergence, SGD with a finite LR bounces
    around a minimum; the running average sits in the flatter center of
    that basin, which transfers better across the temporal train->test
    shift. Weight-space averaging is complementary to the prediction-space
    two-seed ensemble.
    """

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


def _val_mse(net, X_v, Y_v, M_v, bs):
    net.eval()
    with torch.no_grad():
        losses = []
        for i in range(0, len(X_v), bs):
            mu, _ = net(X_v[i:i + bs])
            losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs]).item())
    return float(np.mean(losses))


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train one MoE ensemble member with the recency-weighted routed loss.
    Each epoch, BOTH the raw net and its weight-EMA shadow are scored on the
    UNWEIGHTED temporal-val masked MSE of the mixture mean — the scored
    quantity — and model selection keeps the better of the two; the LR
    schedule still steps on the raw net's val loss. Then the proven recency
    fine-tune runs, ending on a short-horizon EMA of the fine-tune
    trajectory. Returns (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = MoENet(X_t.shape[1], trunk=(256, 256), n_experts=4,
                 expert_hidden=128, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)

    # EMA horizon ~1000 steps (decay 0.999) ~= 1.2 epochs at bs 4096.
    ema = _EMA(net, decay=0.999)

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            wb = W_t[idx].to(device)
            optimizer.zero_grad()
            mu, (mus, lvs, g) = net(xb)
            loss = _moe_loss(mu, mus, lvs, g, yb, mb, wb)
            loss.backward()
            optimizer.step()
            ema.update(net)

        # Model selection on UNWEIGHTED masked MSE of the mixture mean —
        # the scored quantity — NOT on the training loss, whose value mixes
        # in the variance and balance terms and the recency weighting.
        val_raw = _val_mse(net, X_v, Y_v, M_v, bs)
        val_ema = _val_mse(ema.net, X_v, Y_v, M_v, bs)
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

    # Recency fine-tune (proven +~0.01 in lineage): the validation tail is the
    # most recent — and hence most test-like — 10% of the training window, and
    # the model above never trained on it. After model selection, absorb it
    # with a short low-LR pass over the FULL window (tail included), using the
    # same recency-weighted routed loss so relative row weighting stays
    # consistent. The pass has no validation-based selection, so it returns a
    # short-horizon EMA (decay 0.998); if the deadline cuts the pass short,
    # the EMA has barely moved, so the worst case degenerates to no fine-tune.
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        W_f = torch.cat([W_t, W_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        ft_ema = _EMA(net, decay=0.998)
        n_all = len(X_f)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                xb, yb, mb = X_f[idx].to(device), Y_f[idx].to(device), M_f[idx].to(device)
                wb = W_f[idx].to(device)
                ft_opt.zero_grad()
                mu, (mus, lvs, g) = net(xb)
                loss = _moe_loss(mu, mus, lvs, g, yb, mb, wb)
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
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split: rows are time-ordered (later rows = later in
    # the mission window, regions interleaved), and the test set is a temporal
    # holdout. Validating on the last 10% of the training window makes early
    # stopping and the LR schedule optimize forward-in-time generalization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights over the time-ordered window: latest rows weigh 3x the
    # earliest (exponential ramp, normalized to mean 1 so the loss scale — and
    # hence the tuned LR/scheduler behavior — is unchanged).
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
    w_all = np.exp(np.log(3.0) * pos).astype(np.float32)
    w_all /= w_all.mean()

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    W_t = torch.from_numpy(w_all[train_idx])
    # Keep CPU copies of the validation tail for the post-selection fine-tune.
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    W_v_cpu = torch.from_numpy(w_all[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    # Two-seed deep ensemble with per-member wall-clock slices. Independently
    # seeded MoE nets can also learn DIFFERENT regime partitions, so their
    # mixture means decorrelate more than two identical-architecture MLPs.
    # Member m must finish by 0.95*(m+1)/M of the budget, leaving 5% for
    # test prediction.
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse than
    # the best member) so the worst case degenerates to a single MoE net
    # instead of averaging in an undertrained one.
    finite = [(net, v) for net, v in members if np.isfinite(v)]
    if finite:
        v_best = min(v for _, v in finite)
        keep = [net for net, v in finite if v <= v_best * 1.15]
    else:
        keep = [members[0][0]]

    # Predict on the test set in batches, averaging the mixture mean over
    # accepted members.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net in keep:
                mu, _ = net(xb)
                acc += mu
            out[i:i + 65536] = (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
