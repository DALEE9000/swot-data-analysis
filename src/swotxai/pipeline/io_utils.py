from __future__ import annotations

import os
import pickle
import time
import uuid
from pathlib import Path


def _replace_into_place(tmp: Path, path: Path) -> None:
    """Atomically move a finished temp file over the cache path.

    On Windows the destination can be locked by a concurrent reader/writer of
    the same keyed cache; cache keys encode content, so losing that race is
    fine — the other party's bytes are equivalent.
    """
    try:
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        if not path.exists():
            raise


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    _replace_into_place(tmp, path)


def _load(path: Path):
    # Windows: a concurrent os.replace briefly holds the destination in a
    # delete-pending state where open() raises EACCES — retry through it.
    for attempt in range(5):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _cached(path: Path, use_cache: bool) -> bool:
    return use_cache and path.exists()


def _save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".joblib":
        import joblib
        tmp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex[:8]}")
        joblib.dump(model, tmp)
        _replace_into_place(tmp, path)
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
