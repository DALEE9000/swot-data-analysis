from __future__ import annotations

import pickle
from pathlib import Path


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _cached(path: Path, use_cache: bool) -> bool:
    return use_cache and path.exists()


def _save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".joblib":
        import joblib
        joblib.dump(model, path)
    else:
        _save(model, path)


def _load_model(path: Path):
    if path.suffix == ".joblib":
        import joblib
        return joblib.load(path)
    return _load(path)


def _load_s3_pkl(s3_path: str):
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(s3_path) as f:
        return pickle.load(f)


def _s3_exists(s3_path: str) -> bool:
    try:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        return fs.exists(s3_path)
    except Exception:
        return False


def _save_s3_pkl(obj, s3_path: str) -> None:
    import s3fs
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(s3_path, "wb") as f:
        pickle.dump(obj, f)
