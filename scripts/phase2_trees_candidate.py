"""Gradient-boosted trees learner for the holdout protocol (the pipeline's `rf` backend is
LightGBM, so this is the project's tree model; this build has no CUDA tree learner, so it runs on the CPU). Same interface as an evolve
candidate: train_and_predict(X_train, Y_train, X_test, params) -> (n_test, 2).

One LightGBM regressor per component on the raw columns (NaN handled natively, no
featurization), rows time-ordered so the last 10 % is the early-stopping split, as in the
champion. Run through swotxai.evolve.run_candidate by scripts/holdout_eval.py --learner trees.
"""
from __future__ import annotations

import numpy as np


def train_and_predict(X_train, Y_train, X_test, params):
    import lightgbm as lgb
    seed = int(params.get("seed", 42))
    n = len(X_train); n_val = max(1, n // 10)
    pred = np.zeros((len(X_test), 2), np.float32)
    for c in range(2):
        ok = np.isfinite(Y_train[:, c])
        tr = np.flatnonzero(ok[: n - n_val]); va = n - n_val + np.flatnonzero(ok[n - n_val:])
        m = lgb.LGBMRegressor(n_estimators=int(params.get("n_estimators", 600)), learning_rate=0.05,
                              num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed + c, n_jobs=4, verbose=-1)
        m.fit(X_train[tr], Y_train[tr, c], eval_set=[(X_train[va], Y_train[va, c])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        for i in range(0, len(X_test), 2_000_000):
            pred[i:i + 2_000_000, c] = m.predict(X_test[i:i + 2_000_000])
    return pred


if __name__ == "__main__":   # self-check: recovers a planted linear signal
    rng = np.random.default_rng(0); X = rng.normal(size=(20000, 10)).astype(np.float32)
    Y = np.column_stack([X[:, 1] + 0.1 * rng.normal(size=20000), -X[:, 2]]).astype(np.float32)
    p = train_and_predict(X[:16000], Y[:16000], X[16000:], {"seed": 1, "n_estimators": 100})
    r2 = 1 - ((Y[16000:, 0] - p[:, 0]) ** 2).mean() / Y[16000:, 0].var()
    assert r2 > 0.9, r2; print("ok", round(float(r2), 3))
