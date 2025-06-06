from pathlib import Path
from functools import lru_cache
import lmdb, numpy as np

# ---------- 文本  ----------
@lru_cache(maxsize=None)
def _open_text_lmdb(root: Path):
    return lmdb.open(str(root / "clip_text.lmdb"), readonly=True, lock=False)

def get_text_feat(text: str, root: Path) -> np.ndarray:
    with _open_text_lmdb(root).begin() as txn:
        buf = txn.get(text.encode())
    if buf is None:
        raise KeyError(f"'{text}' not found in clip_text.lmdb")
    return np.frombuffer(buf, dtype=np.float16).astype(np.float32)

# ---------- 全局帧 ----------
@lru_cache(maxsize=None)
def _open_global_lmdb(log_dir: Path):
    return lmdb.open(str(log_dir / "clip_global.lmdb"), readonly=True, lock=False)

def get_global_feat(log_dir: Path, cam: str, ts: int):
    with _open_global_lmdb(log_dir).begin() as txn:
        buf = txn.get(f"{cam}/{ts}".encode())
    return None if buf is None else np.frombuffer(buf, dtype=np.float16).astype(np.float32)

# ---------- 对象框 ----------
@lru_cache(maxsize=None)
def _open_obj_lmdb(log_dir: Path):
    return lmdb.open(str(log_dir / "clip_object.lmdb"), readonly=True, lock=False)

def get_obj_feat(log_dir: Path, ts: int, det_key: str):
    with _open_obj_lmdb(log_dir).begin() as txn:
        buf = txn.get(f"{ts}/{det_key}".encode())
    return None if buf is None else np.frombuffer(buf, dtype=np.float16).astype(np.float32)
