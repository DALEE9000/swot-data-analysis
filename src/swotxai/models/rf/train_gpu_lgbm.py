import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def train(X, y, n_estimators: int, max_depth: int, random_state: int, n_jobs: int):
    import lightgbm as lgb

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    rf = lgb.LGBMRegressor(
        device="cuda", n_estimators=n_estimators, max_depth=max_depth,
        random_state=random_state, n_jobs=n_jobs, verbose=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return rf


def predict(rf, X) -> np.ndarray:
    return np.asarray(rf.predict(X))
