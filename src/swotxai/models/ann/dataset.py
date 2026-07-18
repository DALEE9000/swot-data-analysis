from __future__ import annotations

import numpy as np


def concat_for_ann(flattened: dict, training_percentage: float = 0.8, held_out: bool = False):
    """Build a joint (X, Y) training set from the shared flattened dict.

    Unlike ``concat_flattened`` (which returns separate X_u / X_v matrices
    for the two independent RF models), the ANN trains one network on both
    components, so rows are kept whenever *either* target is valid and
    Y has shape (n, 2) with NaN marking the invalid component — the loss
    masks those entries.

    Relies on the stencil flattening preserving the DataFrame row index in
    entry["y_u"] / entry["y_v"] so targets can be re-aligned to entry["df"].
    """
    keys = list(flattened.keys())
    n = max(1, int(len(keys) * training_percentage))
    selected_keys = keys[n:] if held_out else keys[:n]

    X_list, Y_list = [], []
    for t in selected_keys:
        for entry in flattened[t]:
            df = entry["df"]
            y_u = entry["y_u"].reindex(df.index)
            y_v = entry["y_v"].reindex(df.index)
            valid = y_u.notna().to_numpy() | y_v.notna().to_numpy()
            if not valid.any():
                continue
            X_list.append(df.to_numpy(dtype=np.float32)[valid])
            Y_list.append(np.column_stack([
                y_u.to_numpy(dtype=np.float32)[valid],
                y_v.to_numpy(dtype=np.float32)[valid],
            ]))

    if not X_list:
        raise ValueError("No valid training rows found in flattened data.")
    return np.concatenate(X_list, axis=0), np.concatenate(Y_list, axis=0)


def fit_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/scale ignoring NaNs (stencil padding)."""
    mean = np.nanmean(X, axis=0)
    scale = np.nanstd(X, axis=0)
    mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    return mean, scale
