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
    for c in (9, 25, 49):
        if d % c == 0:
            return c
    return 1


def _featurize(X, mean, scale, n_cells):
    """Standardize, gradient-fill missing stencil cells, append missingness,
    spatial std, explicit per-feature spatial derivatives (first AND second
    order), and nonlinear flow-regime features.

    Grafted verbatim from the proven gen017_02 lineage:
      * per-cell missing fraction (n, C);
      * per-feature stencil std over VALID cells (n, F) — front/eddy signal
        and a natural routing coordinate for the MoE gate;
      * gradient-preserving mirror imputation (2*center - mirror, clamped to
        +/-4 sigma, center fallback) so swath-edge truncation keeps the
        first-order gradient instead of flattening the neighborhood — this
        is the direct attack on the parent's edge-stencil deficit;
      * per-feature central differences gx, gy (n, 2F);
      * per-feature 5-point Laplacian and the two diagonal central
        differences (n, 3F);
      * |grad f| per feature (n, F) — front intensity;
      * when F == 8 (pooled layout [mdt, ssha, ugos, vgos, ugosa, vgosa,
        era5_u, era5_v]): center-cell flow speeds, relative vorticity,
        normal/shear strain, Okubo-Weiss W = sn^2+ss^2-zeta^2 (the classic
        eddy-core vs strain-front discriminant), and geostrophic advection
        u_g·grad(f) of every feature. These give the softmax gate an
        explicit regime coordinate to specialize experts on, instead of
        forcing it to synthesize regime from raw stencil values.
    Quadratic terms are clipped to +/-16 (inputs sigma-scaled, fills clamped
    to +/-4) so outliers do not blow up the first Linear layer.
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
        # ---- nonlinear flow-regime block ----
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


class MoE(nn.Module):
    """Shared trunk + softmax-gated mixture of small expert heads.

    A single MLP head is one function fit to a flow distribution dominated
    by slow, quiet ocean; the fast regime is systematically under-fit. Here
    the trunk builds a shared representation, a gate assigns each row a
    soft mixture over experts, and each expert is free to specialize (e.g.
    large-amplitude front/eddy dynamics vs. quiet background) without its
    weights being pulled toward the majority regime's mean. Dense soft
    gating keeps everything differentiable and cheap at this width. With
    the new regime features (vorticity, strain, Okubo-Weiss, speeds) in the
    input, the gate sees the eddy/front discriminant directly.
    """

    def __init__(self, n_inputs, n_experts=4, trunk=(256, 256), expert_hidden=128, dropout=0.1):
        super().__init__()
        layers = []
        d = n_inputs
        for h in trunk:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.gate = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, expert_hidden), nn.SiLU(), nn.Linear(expert_hidden, 2))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        h = self.trunk(x)
        g = torch.softmax(self.gate(h), dim=-1)                    # (n, E)
        outs = torch.stack([e(h) for e in self.experts], dim=1)    # (n, E, 2)
        y = (g.unsqueeze(-1) * outs).sum(dim=1)
        return y, g


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _loss(net, xb, yb, mb, lb_coef=1e-2):
    pred, g = net(xb)
    mse = _masked_mse(pred, yb, mb)
    # Load-balance penalty: E * sum(mean_gate^2) is minimized (=1) at uniform
    # expert usage; without it soft gates tend to collapse onto one expert
    # early, degenerating back to the monolithic parent.
    imp = g.mean(dim=0)
    lb = (imp.pow(2).sum() * g.shape[1] - 1.0)
    return mse + lb_coef * lb


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(params["device"])
    max_epochs = int(params["max_epochs"])
    time_budget_s = float(params["time_budget_s"])
    t0 = time.time()
    deadline = t0 + time_budget_s * 0.95

    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    n_cells = _cells(X_train.shape[1])
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split (proven): last 10% of the time-ordered window
    # so early stopping optimizes forward-in-time generalization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    net = MoE(X_t.shape[1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            loss = _loss(net, xb, yb, mb)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                pred, _ = net(X_v[i:i + bs])
                # Model selection on pure masked MSE — the quantity the test
                # scores — not the regularized training objective.
                val_losses.append(_masked_mse(pred, Y_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() > deadline:
            break

    net.load_state_dict(best_state)

    # Recency fine-tune (proven in gen002_00): absorb the held-out validation
    # tail — the most test-like data — with a short low-LR pass over the full
    # window after model selection.
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        n_all = len(X_f)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                xb, yb, mb = X_f[idx].to(device), Y_f[idx].to(device), M_f[idx].to(device)
                ft_opt.zero_grad()
                loss = _loss(net, xb, yb, mb)
                loss.backward()
                ft_opt.step()
                if bi % 100 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break

    net.eval()
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            pred, _ = net(xb)
            out[i:i + 65536] = pred.cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
