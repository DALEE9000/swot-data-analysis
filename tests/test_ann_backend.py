"""Smoke tests for the ANN backend and the model-partitioned config."""
import numpy as np
import pandas as pd
import pytest
import yaml

from swotxai.config import SWOTConfig, load_config


def _progress(step, frac, msg):
    pass


def make_config(tmp_path, **overrides):
    kwargs = dict(
        model="ann",
        features=["mdt", "ssha_filtered"],
        stencil_k=3,
        cache_dir=str(tmp_path / "cache"),
        run_id="smoke",
        ann_hidden_layers=[32, 32],
        ann_dropout=0.0,
        ann_lr=3e-3,
        ann_max_epochs=80,
        ann_patience=25,
        ann_batch_size=128,
    )
    kwargs.update(overrides)
    return SWOTConfig(**kwargs)


def make_flattened(n_cycles=5, rows_per_cycle=400, n_features=2, k=3, seed=0):
    """Synthetic dict shaped like rf_flattening_stencil output, with the
    row-index alignment between df and y_u/y_v that concat_for_ann relies on."""
    rng = np.random.default_rng(seed)
    n_cols = n_features * k * k
    flattened = {}
    for c in range(n_cycles):
        X = rng.normal(size=(rows_per_cycle, n_cols)).astype(np.float32)
        # deterministic targets: linear maps of the inputs + small noise
        w_u = np.linspace(-1, 1, n_cols)
        w_v = np.linspace(1, -1, n_cols) ** 2
        y_u = X @ w_u + 0.01 * rng.normal(size=rows_per_cycle)
        y_v = X @ w_v + 0.01 * rng.normal(size=rows_per_cycle)
        # knock out some targets (independently for u and v) and some features
        y_u[rng.random(rows_per_cycle) < 0.3] = np.nan
        y_v[rng.random(rows_per_cycle) < 0.3] = np.nan
        X[rng.random(X.shape) < 0.02] = np.nan

        df = pd.DataFrame(X)
        y_u = pd.Series(y_u, name="u")
        y_v = pd.Series(y_v, name="v")
        mask_u, mask_v = ~y_u.isna(), ~y_v.isna()
        flattened[str(474 + c)] = [{
            "df": df,
            "X_u": df[mask_u], "X_v": df[mask_v],
            "y_u": y_u[mask_u], "y_v": y_v[mask_v],
        }]
    return flattened


def test_ann_train_evaluate_predict(tmp_path):
    from swotxai.models import get_backend, predict

    config = make_config(tmp_path)
    backend = get_backend("ann")
    flattened = make_flattened()

    model_u, model_v = backend.step_train(config, flattened, _progress, use_cache=True)
    assert config.cache_path("ann_model").exists()

    metrics = backend.step_evaluate(config, model_u, model_v, flattened, _progress)
    assert metrics["r2_u"] > 0.8, f"ANN failed to fit linear synthetic data: {metrics}"
    assert metrics["r2_v"] > 0.8, f"ANN failed to fit linear synthetic data: {metrics}"
    assert set(metrics["feature_importance_u"]) == {"mdt", "ssha_filtered"}
    fi_total = sum(metrics["feature_importance_u"].values())
    assert fi_total == pytest.approx(1.0, abs=1e-6) or fi_total == 0.0

    # backend-agnostic dispatch handles component views (NaNs included)
    X = flattened["474"][0]["df"].to_numpy()
    pred = predict(model_u, X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


def test_ann_cache_reload_and_staleness(tmp_path):
    from swotxai.models import get_backend

    config = make_config(tmp_path)
    backend = get_backend("ann")
    flattened = make_flattened()

    model_u, _ = backend.step_train(config, flattened, _progress, use_cache=True)
    first_meta = model_u.base.meta

    # same config → loaded from cache (identical training history object)
    model_u2, _ = backend.step_train(config, flattened, _progress, use_cache=True)
    assert model_u2.base.meta["history"] == first_meta["history"]

    # changed architecture → stale cache → retrained with new hidden sizes
    config2 = make_config(tmp_path, ann_hidden_layers=[16])
    model_u3, _ = backend.step_train(config2, flattened, _progress, use_cache=True)
    assert model_u3.base.meta["hidden"] == [16]

    # changed activation → stale cache → retrained; recorded and reloadable
    config3 = make_config(tmp_path, ann_hidden_layers=[16], ann_activation="relu")
    model_u4, _ = backend.step_train(config3, flattened, _progress, use_cache=True)
    assert model_u4.base.meta["activation"] == "relu"
    from swotxai.models.ann.model import ANNRegressor
    reloaded = ANNRegressor.load(config3.cache_path("ann_model"))
    assert reloaded.meta["activation"] == "relu"


def test_cache_paths_partitioned_by_model(tmp_path):
    rf_cfg = make_config(tmp_path, model="rf")
    ann_cfg = make_config(tmp_path, model="ann")

    assert "rf" in rf_cfg.cache_path("inference").parts
    assert "ann" in ann_cfg.cache_path("inference").parts
    assert rf_cfg.cache_path("inference") != ann_cfg.cache_path("inference")
    # shared flattened data is NOT partitioned
    assert rf_cfg.cache_path("flattened") == ann_cfg.cache_path("flattened")
    assert rf_cfg.cache_path("rf_u").suffix == ".joblib"
    assert ann_cfg.cache_path("ann_model").suffix == ".pt"
    # frames/animations are partitioned too, named after the experiment id
    rf_cfg.experiment_id = "rf_test_id"
    ann_cfg.experiment_id = "ann_test_id"
    rf_cfg.ensure_output_paths()
    ann_cfg.ensure_output_paths()
    assert rf_cfg.frame_dir.endswith("/rf/rf_test_id")
    assert ann_cfg.frame_dir.endswith("/ann/ann_test_id")
    assert ann_cfg.animation_output.endswith("/ann/ann_test_id")

    region_cfg = make_config(tmp_path, model="ann", region="uswc")
    assert region_cfg.cache_path("ann_model").parts[-4:-1] == ("uswc", "ann", "weights")

    # blank run_id: model artifacts are keyed by the experiment id, but shared
    # data stems stay id-free so they're reused across runs (and cleanup paths
    # match what the shared steps wrote before the id was minted)
    noid_cfg = make_config(tmp_path, model="ann", run_id="")
    shared_before = noid_cfg.cache_path("cycle_data")
    noid_cfg.experiment_id = "ann_20260710_000000_abc123"
    assert noid_cfg.cache_path("ann_model").stem.endswith("ann_20260710_000000_abc123")
    assert noid_cfg.cache_path("inference").stem.endswith("ann_20260710_000000_abc123")
    assert noid_cfg.cache_path("cycle_data") == shared_before
    # explicit run_id still wins over the experiment id
    named_cfg = make_config(tmp_path, model="ann")
    named_cfg.experiment_id = "ann_x"
    assert named_cfg.cache_path("ann_model").stem.endswith("smoke")


def test_legacy_config_migration(tmp_path):
    legacy = {
        "n_estimators": 99,
        "max_depth": 7,
        "use_gpu": False,
        "use_lgbm": True,
        "sklearn_n_jobs": 4,
        "run_id": "legacy",
    }
    p = tmp_path / "old.yaml"
    p.write_text(yaml.dump(legacy))
    cfg = load_config(p)
    assert cfg.rf_n_estimators == 99
    assert cfg.rf_max_depth == 7
    assert cfg.rf_use_gpu is False
    assert cfg.rf_use_lgbm is True
    assert cfg.rf_n_jobs == 4
    assert cfg.model == "rf"


def test_experiment_registry(tmp_path, monkeypatch):
    from swotxai import experiments

    monkeypatch.setattr(experiments, "REGISTRY_ROOT", tmp_path / "registry")
    metrics = {"rmse_u": 0.1, "rmse_v": 0.2, "r2_u": 0.9, "r2_v": 0.8,
               "feature_importance_u": {"mdt": 1.0}, "feature_importance_v": {"mdt": 1.0}}

    rec_ann = experiments.record_experiment(
        make_config(tmp_path), metrics, extra={"step_seconds": {"train": 1.0}})
    rec_rf = experiments.record_experiment(
        make_config(tmp_path, model="rf"), metrics)

    assert rec_ann["experiment_id"].startswith("ann_")
    assert rec_rf["experiment_id"].startswith("rf_")
    assert rec_ann["experiment_id"] != rec_rf["experiment_id"]
    # partitioned per model, with jsonl + csv in each
    assert (tmp_path / "registry" / "ann" / "experiments.jsonl").exists()
    assert (tmp_path / "registry" / "ann" / "experiments_summary.csv").exists()
    assert (tmp_path / "registry" / "rf" / "experiments.jsonl").exists()
    # record captures everything fed into the run
    assert rec_ann["hyperparameters"]["ann_hidden_layers"] == [32, 32]
    assert rec_ann["config"]["features"] == ["mdt", "ssha_filtered"]
    assert rec_ann["stencil_k"] == 3
    assert rec_ann["metrics"]["r2_u"] == 0.9
    assert rec_ann["step_seconds"] == {"train": 1.0}
    assert "python" in rec_ann["environment"]

    loaded = experiments.load_experiments()
    assert {r["experiment_id"] for r in loaded} == {rec_ann["experiment_id"], rec_rf["experiment_id"]}
    only_rf = experiments.load_experiments("rf")
    assert [r["model"] for r in only_rf] == ["rf"]

    # an id minted at run start (used for animation naming) is reused verbatim
    cfg = make_config(tmp_path)
    cfg.experiment_id = "ann_20260710_000000_fixed"
    rec = experiments.record_experiment(cfg, metrics)
    assert rec["experiment_id"] == "ann_20260710_000000_fixed"


def test_train_epoch_events(tmp_path):
    import json
    from swotxai.models.ann.training import train_ann

    events = []
    def cb(step, frac, msg):
        if step == "train_epoch":
            events.append(json.loads(msg))

    config = make_config(tmp_path, ann_max_epochs=3, ann_patience=10)
    X = np.random.rand(500, 18).astype(np.float32)
    Y = np.random.rand(500, 2).astype(np.float32)
    train_ann(X, Y, config, cb=cb)

    assert len(events) == 3
    required = {"epoch", "max_epochs", "train_loss", "val_loss", "lr",
                "epoch_s", "elapsed_s", "best_epoch", "best_val_loss", "device"}
    assert required <= set(events[0])
    assert [e["epoch"] for e in events] == [1, 2, 3]
    assert events[-1]["elapsed_s"] >= events[0]["elapsed_s"]
