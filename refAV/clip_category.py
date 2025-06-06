# -*- coding: utf-8 -*-
"""
对象级 CLIP 语义匹配辅助
================================================
- 离线检测-跟踪-裁剪后的特征保存在
    features_obj/<track_uuid>/<direction>.h5
  内含:  features (N,D), timestamps (N,), bboxes (N,4), labels (N,)
- 这里按 track_uuid 汇总所有 direction→一个 T×D 数组
"""
from __future__ import annotations
import os, functools
from pathlib import Path
from typing import Dict, List

import h5py, numpy as np, torch
try:
    import open_clip
except ImportError as e:  # pragma: no cover
    raise RuntimeError("pip install open_clip_torch") from e


# ---------------------- 默认配置 ---------------------- #
_CLIP_CFG_DEFAULT = dict(
    model      = "ViT-B-32",
    pretrained = "laion2b_s34b_b79k",
    device     = "cuda",          # 自动 fallback 到 CPU
    feature_root = "features_obj",
    sim_threshold = 0.28,         # 经验阈值; 可自行调
)

_PROMPT_MAP = {
    # 可以按需补充 / 修改
    "VEHICLE"    : "a photo of a vehicle",
    "CAR"        : "a photo of a car",
    "TRUCK"      : "a photo of a truck",
    "BUS"        : "a photo of a bus",
    "PEDESTRIAN" : "a photo of a person walking",
    "BICYCLE"    : "a photo of a bicycle",
    # ...
}
# ----------------------------------------------------- #


# ---------- 单例加载 CLIP ---------- #
@functools.lru_cache(maxsize=None)
def _load_clip(model:str, pretrained:str, device:str):
    device = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model, pretrained=pretrained, device=device
    )
    clip_model.eval()
    return clip_model, preprocess, device


@functools.lru_cache(maxsize=None)
def _encode_prompt(prompt:str, model:str, pretrained:str, device:str) -> torch.Tensor:
    """取文本嵌入 (D,) -- 已经 L2-norm。"""
    clip_model, _, device = _load_clip(model, pretrained, device)
    with torch.no_grad():
        tokens = open_clip.tokenize(prompt).to(device)
        txt = clip_model.encode_text(tokens)[0]
        txt = txt / txt.norm()
    return txt.cpu()


# ---------- 读取对象级特征 ---------- #
def _load_track_features(track_uuid:str, feature_root:Path) -> np.ndarray:
    """
    将 track_uuid 旗下所有 direction.h5 拼成 (T,D) float16
    如果某些 direction 没有 .h5 会自动跳过。
    """
    feats: List[np.ndarray] = []
    track_dir = feature_root / track_uuid
    if not track_dir.exists():
        return np.empty((0, 512), dtype=np.float16)  # 512 for ViT-B-32
    for h5_path in track_dir.glob("*.h5"):
        with h5py.File(h5_path, "r") as f:
            feats.append(f["features"][...])         # (N,D)
    if not feats:
        return np.empty((0, 512), dtype=np.float16)
    return np.concatenate(feats, 0)                 # (T,D)


# ------------------ 对外主类 ------------------ #
class CLIPCategoryMatcher:
    """
    用法：
        matcher = CLIPCategoryMatcher(cfg)        # 可重复复用
        ok, sim = matcher.match(track_uuid, "CAR")
    """
    def __init__(self, cfg:Dict|None=None):
        self.cfg = {**_CLIP_CFG_DEFAULT, **(cfg or {})}
        # 预热加载模型
        _load_clip(self.cfg['model'], self.cfg['pretrained'], self.cfg['device'])

    # --------- 核心匹配函数 --------- #
    def match(self, track_uuid:str, category:str) -> tuple[bool, float]:
        """
        return: (是否属于该类, 最高相似度)
        """
        prompt = _PROMPT_MAP.get(category.upper(), f"a photo of {category.lower()}")
        txt_embed = _encode_prompt(
            prompt,
            self.cfg['model'], self.cfg['pretrained'], self.cfg['device']
        ).numpy()          # (D,)
        obj_feats = _load_track_features(track_uuid, Path(self.cfg['feature_root']))
        if obj_feats.size == 0:
            return False, 0.0
        # 余弦相似度 (N,)  -- 已经 L2-norm
        sims = obj_feats @ txt_embed
        best = float(sims.max())
        return best >= self.cfg['sim_threshold'], best

    # --------- 给 scenario 获取所有对象 --------- #
    def get_objects_of_category(self, log_dir:Path, category:str,
                                candidate_uuids:List[str]|None=None):
        """
        返回与原 `get_objects_of_category` 同结构的 scenario dict:
            { uuid: timestamps[] }
        没有 timestamps 信息时返回空列表，保持兼容。
        """
        from refAV.utils import to_scenario_dict, get_timestamps  # lazy import
        if candidate_uuids is None:
            # 粗取全部对象 uuid（任何类别）
            from refAV.utils import get_uuids_of_category
            candidate_uuids = get_uuids_of_category(log_dir, "ANY")

        scenario = {}
        for uuid in candidate_uuids:
            ok, _ = self.match(uuid, category)
            if ok:
                scenario[uuid] = get_timestamps(uuid, log_dir)  # 与原函数保持一致
        return scenario
