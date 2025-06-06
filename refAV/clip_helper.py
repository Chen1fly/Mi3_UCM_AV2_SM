from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import numpy as np, lmdb, torch, open_clip, os

_DATA_ROOT: str | None = None
_TEXT_DB: str | None = None

_CAM = [
    "ring_front_center", "ring_front_left", "ring_front_right",
    "ring_rear_left", "ring_rear_right",
    "ring_side_left", "ring_side_right",
    "stereo_front_left", "stereo_front_right"
]


@lru_cache(maxsize=None)
def _open_env(path: Path) -> lmdb.Environment:
    return lmdb.open(str(path), readonly=True, lock=False,
                     readahead=False, meminit=False, subdir=True)


@lru_cache(maxsize=16384)
def _compute_txt(text: str) -> np.ndarray:
    model, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    with torch.inference_mode():
        token = open_clip.tokenize([text]).cuda()
        feat = model.cuda().eval().encode_text(token)[0].float().cpu().numpy()
    return feat / np.linalg.norm(feat)


@lru_cache(maxsize=16384)
def get_text_feat(text: str) -> np.ndarray:
    if _TEXT_DB and Path(_TEXT_DB).exists():
        try:
            env = _open_env(Path(_TEXT_DB))
            with env.begin() as txn:
                buf = txn.get(text.encode("utf-8"))
            if buf:
                arr = np.frombuffer(buf, np.float16).astype(np.float32)
                return arr / np.linalg.norm(arr)
        except lmdb.Error:
            pass
    return _compute_txt(text)


def _resolve_log(log_dir: Path) -> Path:
    if _DATA_ROOT is None:
        return log_dir
    return Path(_DATA_ROOT) / log_dir.name


def _fetch_feat(env: lmdb.Environment, key: str) -> np.ndarray | None:
    with env.begin() as txn:
        buf = txn.get(key.encode())
    if buf:
        arr = np.frombuffer(buf, np.float16).astype(np.float32)
        return arr / np.linalg.norm(arr)
    return None


def get_image_feat(cam: str, ts: int, log_dir: Path, obj_uuid: str | None = None
                   ) -> np.ndarray | None:
    root = _resolve_log(log_dir)
    env_name = "clip_object.lmdb" if obj_uuid else "clip_global.lmdb"
    try:
        env = _open_env(root / env_name)
    except lmdb.Error:
        return None
    key = f"{ts}/{cam}_{obj_uuid}" if obj_uuid else f"{cam}/{ts}"
    return _fetch_feat(env, key)


def clip_sim(uuid: str,
             log_dir: Path,
             text: str,
             *,
             w_g: float = .5,
             w_o: float = .5,
             stride: int = 4,
             thr: float = .28) -> bool:
    from refAV.utils import get_timestamps
    root = _resolve_log(log_dir)

    try:
        g_env = _open_env(root / "clip_global.lmdb")
        o_env = _open_env(root / "clip_object.lmdb")
    except lmdb.Error:
        return False

    txt = get_text_feat(text)
    ts_all = get_timestamps(uuid, log_dir)[::stride]

    best = 0.0
    for cam in _CAM:
        # Global
        g_vals = []
        with g_env.begin() as txn:
            for t in ts_all:
                buf = txn.get(f"{cam}/{t}".encode())
                if buf:
                    g_vals.append(np.frombuffer(buf, np.float16).astype(np.float32))
        g_sim = 0. if not g_vals else np.dot(np.stack(g_vals), txt).max()

        # Obj
        o_vals = []
        with o_env.begin() as txn:
            for t in ts_all:
                buf = txn.get(f"{t}/{cam}_{uuid}".encode())
                if buf:
                    o_vals.append(np.frombuffer(buf, np.float16).astype(np.float32))
        o_sim = 0. if not o_vals else np.dot(np.stack(o_vals), txt).max()

        best = max(best, w_g * g_sim + w_o * o_sim)

        if best >= thr:
            return True

    return best >= thr
