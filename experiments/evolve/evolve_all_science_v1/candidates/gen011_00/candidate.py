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
    """Standardize, gradient-fill missing stencil cells, append missingness + spatial std.

    Feature-major layout: column f * n_cells + c is feature f at stencil
    cell c, so reshaping to (n, F, C) groups each feature's spatial
    neighborhood. Three ingredients:
      * per-cell missing fraction (n, C) — lets the net distinguish
        swath-edge padding from a genuinely calm/average ocean state (edge
        rows are the widest R^2 gap in the diagnostics);
      * per-feature stencil standard deviation (n, F), computed on the
        standardized values over VALID cells only — an explicit front/eddy
        intensity signal the flat feature vector otherwise hides. Rows with
        a single valid cell get std 0 ("no local gradient information");
      * gradient-preserving imputation: a missing stencil cell at offset
        (di, dj) from the center is filled with 2*center - mirror, where
        mirror is the cell at (-di, -dj) — the first-order Taylor (constant
        local gradient) continuation of the field. Swath-edge padding
        truncates one SIDE of the neighborhood, so the mirror cell is
        usually valid; mean-fill (the previous scheme) replaced edge
        neighborhoods with "globally average ocean", erasing exactly the
        gradient signal the geostrophic features carry. Extrapolated fills
        are clamped to +/-4 standardized sigma so edge noise is not
        amplified; where the mirror is itself missing the fill falls back
        to the center value, and rows whose center is NaN fall back to the
        mean (0 after standardization) via the final nan_to_num. The
        missingness block still marks which cells were padded, so the model
        can discount imputed structure.
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
    return np.hstack([Xs, miss, spat])


class MLP(nn.Module):
    """Joint (u, v) MLP: shared trunk, 2-unit head."""

    def __init__(self, n_inputs, hidden=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        d = n_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


class _EMA:
    """Exponential moving average of model weights.

    SGD's final iterates orbit the loss-basin floor; their running average
    sits closer to the basin center and generalizes better — especially
    forward in time, where the training-window optimum is slightly offset
    from the test-window optimum. Decay ramps up from 0 so early epochs are
    not biased toward the random init (~1-epoch effective horizon at 0.999
    with ~800 steps/epoch).
    """

    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in net.state_dict().items()}

    def update(self, net):
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v, alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
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
    # stopping and the LR schedule optimize forward-in-time generalization —
    # the quantity the test actually measures — instead of random-split
    # interpolation, which stops too late and overfits the training window.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    # Keep CPU copies of the validation tail for the post-selection fine-tune.
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    net = MLP(X_t.shape[1], hidden=(256, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    ema = _EMA(net, decay=0.999)
    # Separate instance for evaluating/serving the averaged weights without
    # disturbing the live training weights.
    eval_net = MLP(X_t.shape[1], hidden=(256, 256, 128), dropout=0.1).to(device)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(ema.state_dict())
    best_epoch = 0
    n_train = len(X_t)

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            loss = _masked_mse(net(xb), yb, mb)
            loss.backward()
            optimizer.step()
            ema.update(net)

        # Validate — and therefore early-stop, schedule the LR, and checkpoint
        # — on the EMA weights: model selection picks the averaged solution,
        # which is the one that gets served.
        eval_net.load_state_dict(ema.state_dict())
        eval_net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                val_losses.append(_masked_mse(eval_net(X_v[i:i + bs]), Y_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(ema.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() - t0 > time_budget_s:
            break

    net.load_state_dict(best_state)

    # Recency fine-tune (proven in gen002_00, +0.010 fitness): the validation
    # tail is the most recent — and hence most test-like — 10% of the training
    # window, and the model above never trained on it. After model selection,
    # absorb it with a short low-LR pass over the FULL window (tail included).
    # Low LR + few epochs keeps the selected solution basin intact while
    # adapting to the freshest ocean state; there is no validation signal
    # during this phase by construction, so it is deliberately conservative
    # and budget-checked in-loop. The EMA keeps running here (re-anchored on
    # the selected weights) so the served model is the smoothed fine-tune
    # trajectory rather than whatever the last noisy mini-batches dictate.
    if time.time() - t0 < time_budget_s:
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        ft_ema = _EMA(net, decay=0.999)
        ft_ema.step = 1000  # start at full decay: anchor is the selected model, not a random init
        n_all = len(X_f)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                xb, yb, mb = X_f[idx].to(device), Y_f[idx].to(device), M_f[idx].to(device)
                ft_opt.zero_grad()
                loss = _masked_mse(net(xb), yb, mb)
                loss.backward()
                ft_opt.step()
                ft_ema.update(net)
                if bi % 100 == 0 and time.time() - t0 > time_budget_s:
                    out_of_time = True
                    break
            if out_of_time:
                break
        net.load_state_dict(ft_ema.state_dict())

    net.eval()

    # Predict on the test set in batches.
    out = np.empty((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            out[i:i + 65536] = net(xb).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
