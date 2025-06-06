import numpy as np, lmdb, torch, open_clip
from pathlib import Path
from functools import lru_cache
from .utils import get_timestamps

_CAMERAS = ["ring_front_center","ring_front_left","ring_front_right",
            "ring_rear_left","ring_rear_right","ring_side_left",
            "ring_side_right","stereo_front_left","stereo_front_right"]

# ---------- 缓存 ---------- #
@lru_cache(maxsize=None)
def _open(db: Path):  # 读 LMDB
    return lmdb.open(str(db), readonly=True, lock=False, readahead=False)

@lru_cache(maxsize=None)
def _txt_feat(text: str):      # on-the-fly 编码，不硬写 prompt
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    with torch.inference_mode():
        tok = open_clip.tokenize([text]).cuda()
        f = model.cuda().eval().encode_text(tok).cpu().numpy()[0]
    return f / np.linalg.norm(f)

def clip_score(uuid: str,
               log_dir: Path,
               text: str,
               w_g: float = .5,
               w_o: float = .5,
               stride: int = 3) -> float:
    """九路相机取 max，全局+对象各占一半权重"""
    txt = _txt_feat(text)
    g_env = _open(log_dir/"clip_global.lmdb")
    o_env = _open(log_dir/"clip_object.lmdb")
    ts = get_timestamps(uuid, log_dir)[::stride]

    best = 0.
    for cam in _CAMERAS:
        # 全局
        g_feats = [g_env.begin().get(f"{cam}/{t}".encode()) for t in ts]
        g_feats = [np.frombuffer(b, np.float16).astype(np.float32) for b in g_feats if b]
        g_sim = 0. if not g_feats else np.dot(np.stack(g_feats), txt).max()

        # 对象
        o_feats = [o_env.begin().get(f"{t}/{cam}_{uuid}".encode()) for t in ts]
        o_feats = [np.frombuffer(b, np.float16).astype(np.float32) for b in o_feats if b]
        o_sim = 0. if not o_feats else np.dot(np.stack(o_feats), txt).max()

        best = max(best, w_g*g_sim + w_o*o_sim)
    return float(best)
