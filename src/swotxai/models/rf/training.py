import numpy as np


def train(X, y, config):
    """Dispatch RF training to the appropriate backend based on config flags."""
    if config.rf_use_lgbm:
        from swotxai.models.rf.train_gpu_lgbm import train as _train
        return _train(X, y, config.rf_n_estimators, config.rf_max_depth, config.random_state, config.rf_n_jobs)

    if config.rf_use_gpu:
        try:
            from swotxai.models.rf.train_gpu_cuml import train as _train
            return _train(X, y, config.rf_n_estimators, config.rf_max_depth, config.random_state)
        except ImportError:
            print("cuML not available — falling back to sklearn.")

    from swotxai.models.rf.train_cpu import train as _train
    return _train(X, y, config.rf_n_estimators, config.rf_max_depth, config.random_state, config.rf_n_jobs)


def predict(rf, X) -> np.ndarray:
    """Dispatch prediction to the appropriate backend based on model type."""
    try:
        from cuml.ensemble import RandomForestRegressor as cuRF
        if isinstance(rf, cuRF):
            from swotxai.models.rf.train_gpu_cuml import predict as _predict
            return _predict(rf, X)
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        if isinstance(rf, lgb.LGBMRegressor):
            from swotxai.models.rf.train_gpu_lgbm import predict as _predict
            return _predict(rf, X)
    except ImportError:
        pass

    from swotxai.models.rf.train_cpu import predict as _predict
    return _predict(rf, X)
