import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cached FIL kwargs that work on this machine; None = not yet probed
_cuml_predict_kw: dict | None = None


def train(X, y, n_estimators: int, max_depth: int, random_state: int):
    from cuml.ensemble import RandomForestRegressor as cuRF

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train_f32 = np.asarray(X_train, dtype="float32")
    y_train_f32 = np.asarray(y_train, dtype="float32")

    rf = cuRF(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, max_features=1.0, n_bins=256,
    )
    rf.fit(X_train_f32, y_train_f32)

    # Capture importances before pickling — cuML loses them on reload
    try:
        rf._feature_importances_saved = np.asarray(rf.feature_importances_)
    except Exception:
        rf._feature_importances_saved = None

    y_pred = np.asarray(rf.predict(np.asarray(X_test, dtype="float32")))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return rf


def predict(rf, X) -> np.ndarray:
    global _cuml_predict_kw
    X = np.asarray(X, dtype="float32")
    _attempts = (
        [_cuml_predict_kw] if _cuml_predict_kw is not None
        else [{}, {"layout": "sparse8"}, {"layout": "sparse"}, {"default_chunk_size": 1}]
    )
    for _kw in _attempts:
        try:
            result = np.asarray(rf.predict(X, **_kw))
            _cuml_predict_kw = _kw
            return result
        except (RuntimeError, TypeError):
            continue
    raise RuntimeError(
        "All cuML FIL configurations failed (invalid configuration argument). "
        "Retrain with sklearn or a compatible cuML version."
    )
