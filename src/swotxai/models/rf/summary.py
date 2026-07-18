"""RF-specific run summaries (hyperparameters, metrics, feature importances)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from swotxai.data_utils import concat_flattened


def summarize_results(
    rf_u=None,
    rf_v=None,
    flattened: dict | None = None,
    feature_names: list[str] | None = None,
    stencil_k: int | None = None,
    training_percentage: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
    meta_path: str | Path | None = None,
    weights_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Print hyperparameters, RMSE/R², and feature importances for one or all runs.

    Can be driven three ways:

    1. Pass ``meta_path`` — loads rf_u/rf_v, feature_names, and stencil_k
       automatically from the corresponding meta pkl and joblib files, then
       summarises that single run.

    2. Pass ``weights_dir`` — scans the directory for every ``rf_meta_*.pkl``
       and summarises all runs, returning a combined hyperparameter + importance
       DataFrame (performance metrics are skipped since no flattened data is
       available).

    3. Pass ``rf_u`` and ``rf_v`` directly (original behaviour) along with
       optional ``flattened``, ``feature_names``, and ``stencil_k``.

    Parameters
    ----------
    rf_u, rf_v : fitted model objects or paths to .joblib files
    flattened : flattened dict — used to compute RMSE/R²; omit to skip
    feature_names : base feature names before stencil expansion
    stencil_k : stencil window size used during training
    test_size : fraction held out for evaluation (default 0.2)
    random_state : random seed for train/test split
    meta_path : path to a single ``rf_meta_*.pkl`` file
    weights_dir : directory containing ``rf_meta_*.pkl`` files (all-runs mode)

    Returns
    -------
    pd.DataFrame of per-feature importances (single run) or per-run
    hyperparameters + importances (all-runs mode)
    """
    import re
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # ------------------------------------------------------------------
    # All-runs mode: scan weights_dir for every meta pkl
    # ------------------------------------------------------------------
    if weights_dir is not None:
        weights_dir = Path(weights_dir)
        rows = []
        for mp in sorted(weights_dir.glob("rf_meta_*.pkl")):
            m = re.search(r'\d+_\d+_(.*)', mp.stem)
            run_id = m.group(1) if m else mp.stem
            row_df = summarize_results(meta_path=mp)
            row_df.insert(0, "run_id", run_id)
            rows.append(row_df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Single-run mode: load everything from meta pkl + joblib
    # ------------------------------------------------------------------
    if meta_path is not None:
        meta_path = Path(meta_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        feature_names = feature_names or meta.get("features")
        stencil_k     = stencil_k     or meta.get("stencil_k")

        rf_u_path = meta_path.parent / meta_path.name.replace("rf_meta_", "rf_u_").replace(".pkl", ".joblib")
        rf_v_path = meta_path.parent / meta_path.name.replace("rf_meta_", "rf_v_").replace(".pkl", ".joblib")
        if rf_u is None and rf_u_path.exists():
            rf_u = joblib.load(rf_u_path)
        if rf_v is None and rf_v_path.exists():
            rf_v = joblib.load(rf_v_path)

        print(f"=== Hyperparameters ({meta_path.stem}) ===")
        print(f"  features:     {feature_names}")
        print(f"  stencil_k:    {stencil_k}")
        if rf_u is not None:
            print(f"  n_estimators: {rf_u.n_estimators}")
            print(f"  max_depth:    {rf_u.max_depth}")
            print(f"  random_state: {rf_u.random_state}")
        print()

    # ------------------------------------------------------------------
    # Load models from paths if strings/Paths were passed directly
    # ------------------------------------------------------------------
    if isinstance(rf_u, (str, Path)):
        rf_u = joblib.load(rf_u)
    if isinstance(rf_v, (str, Path)):
        rf_v = joblib.load(rf_v)

    if rf_u is None or rf_v is None:
        raise ValueError("Provide rf_u/rf_v directly, or supply meta_path / weights_dir.")

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------
    if flattened is not None:
        X_u, X_v, y_u, y_v = concat_flattened(flattened, training_percentage=training_percentage)
        _, X_test_u, _, y_test_u = train_test_split(X_u, y_u, test_size=test_size, random_state=random_state)
        _, X_test_v, _, y_test_v = train_test_split(X_v, y_v, test_size=test_size, random_state=random_state)
        pred_u = rf_u.predict(X_test_u)
        pred_v = rf_v.predict(X_test_v)
        perf = pd.DataFrame(
            {"u": [float(np.sqrt(mean_squared_error(y_test_u, pred_u))), float(r2_score(y_test_u, pred_u))],
             "v": [float(np.sqrt(mean_squared_error(y_test_v, pred_v))), float(r2_score(y_test_v, pred_v))]},
            index=["RMSE", "R²"],
        )
        print("=== Performance ===")
        print(perf.to_string())
        print()

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------
    fi_u = getattr(rf_u, "feature_importances_", None)
    fi_v = getattr(rf_v, "feature_importances_", None)

    if fi_u is None or fi_v is None:
        print("Models have no feature_importances_ attribute.")
        return pd.DataFrame()

    fi_u = np.asarray(fi_u)
    fi_v = np.asarray(fi_v)
    n_total = len(fi_u)

    if feature_names is not None and stencil_k is not None:
        k2 = stencil_k ** 2
        fi_u = fi_u.reshape(len(feature_names), k2).mean(axis=1) * k2
        fi_v = fi_v.reshape(len(feature_names), k2).mean(axis=1) * k2
        names = feature_names
    elif feature_names is not None and n_total % len(feature_names) == 0:
        k2 = n_total // len(feature_names)
        fi_u = fi_u.reshape(len(feature_names), k2).mean(axis=1) * k2
        fi_v = fi_v.reshape(len(feature_names), k2).mean(axis=1) * k2
        names = feature_names
    else:
        names = [f"feature_{i}" for i in range(n_total)]

    fi_df = (
        pd.DataFrame({"feature": names, "importance_u": fi_u, "importance_v": fi_v})
        .sort_values("importance_u", ascending=False)
        .reset_index(drop=True)
    )

    print("=== Feature Importances ===")
    print(fi_df.to_string(index=False))
    return fi_df
