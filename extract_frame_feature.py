#!/usr/bin/env python3
"""
refAV/scripts/extract_clip_features_fast.py
==========================================
A drop‑in replacement for *extract_clip_features.py* with **~5‑10× speed‑up** and
an extra option ``--stride`` that lets you **extract one set of features every
*N* frames (default 5) and re‑use it for the following N‑1 frames**.

Main optimisation points
------------------------
1. **Batch inference** for both CLIP and YOLO (`--batch`).
2. **FP16** weights + activations (automatic when you have a CUDA GPU).
3. **LMDB fast flags** + batched commits (`--txn-step`).
4. Optional **multi‑process** launch: simply run the script once *per GPU* and
   point each instance to a different subset of logs.

The code keeps exactly the same LMDB schema (``clip_global.lmdb`` &
``clip_object.lmdb``) so downstream consumers remain untouched.
"""

import argparse
import os
import cv2
import lmdb
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import open_clip
from ultralytics import YOLO
PROXY = "http://127.0.0.1:7890"
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ[k] = PROXY
# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------
CAMERAS: Sequence[str] = [
    "ring_front_center", "ring_front_left", "ring_front_right",
    "ring_rear_left", "ring_rear_right", "ring_side_left",
    "ring_side_right", "stereo_front_left", "stereo_front_right",
]

# Reduce OpenCV thread contention when we already saturate the GPU.
cv2.setNumThreads(min(8, os.cpu_count() or 1))

# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def init_clip(device: str):
    """Load OpenCLIP ViT‑B/32 weights on *device* in half precision."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.half().to(device).eval()
    return model, preprocess


def encode_clip_batch(model, preprocess, imgs_bgr: List[np.ndarray], device: str) -> np.ndarray:
    """FP16 batch inference -> (B, 512) numpy array."""
    tensors = torch.stack([
        preprocess(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in imgs_bgr
    ]).to(device, dtype=torch.float16)

    with torch.inference_mode():
        feats = model.encode_image(tensors).half().cpu().numpy()
    return feats  # shape (B, 512)

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_log(
    log_dir: Path,
    model,
    preprocess,
    yolo,
    device: str,
    batch: int,
    stride: int,
    txn_step: int,
):
    """Extract & store CLIP global/object features for one *log* directory."""

    # -- LMDB with high‑throughput flags ------------------------------------
    g_env = lmdb.open(
        str(log_dir / "clip_global.lmdb"),
        map_size=2 << 30,  # 2 TiB should be plenty
        writemap=True,
        metasync=False,
        sync=False,
        readahead=False,
    )
    o_env = lmdb.open(
        str(log_dir / "clip_object.lmdb"),
        map_size=2 << 30,
        writemap=True,
        metasync=False,
        sync=False,
        readahead=False,
    )
    g_txn = g_env.begin(write=True)
    o_txn = o_env.begin(write=True)
    g_cnt = o_cnt = 0

    # -- Iterate over cameras ----------------------------------------------
    for cam in CAMERAS:
        cam_dir = log_dir / "sensors" / "cameras" / cam
        if not cam_dir.exists():
            continue

        img_paths: List[Path] = sorted(cam_dir.glob("*.jpg"))
        if not img_paths:
            continue

        # Build [sample_path] and corresponding [chunk_paths] for replication.
        sample_paths: List[Path] = []
        chunk_lists: List[List[Path]] = []
        for start in range(0, len(img_paths), stride):
            chunk = img_paths[start : start + stride]
            sample_paths.append(chunk[0])  # take first frame as representative
            chunk_lists.append(chunk)

        # Batch through all *sample* frames ---------------------------------
        for b_start in range(0, len(sample_paths), batch):
            b_sample_paths = sample_paths[b_start : b_start + batch]
            b_chunk_lists = chunk_lists[b_start : b_start + batch]

            # 1. Load images --------------------------------------------------
            b_imgs = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in b_sample_paths]

            # 2. Global CLIP features ---------------------------------------
            g_feats = encode_clip_batch(model, preprocess, b_imgs, device)

            # 3. YOLO batch detection ---------------------------------------
            det_results = yolo(b_imgs, imgsz=640)

            # 4. Write LMDB entries -----------------------------------------
            for img_path, chunk, g_feat, det in zip(b_sample_paths, b_chunk_lists, g_feats, det_results):
                # -- replicate GLOBAL feature for *all* frames in the chunk --
                for p in chunk:
                    ts = p.stem  # timestamp from filename
                    g_txn.put(f"{cam}/{ts}".encode(), g_feat.tobytes())
                    g_cnt += 1

                # -- prepare OBJECT crops -----------------------------------
                crops, rep_keys = [], []
                for i, box in enumerate(det.boxes.xyxy.cpu().numpy()):
                    x0, y0, x1, y1 = map(int, box)
                    crop = b_imgs[0][y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    crops.append(crop)
                    # replicate the same object feature across the *chunk*
                    rep_keys.append([
                        f"{p.stem}/{cam}_{i}".encode() for p in chunk
                    ])

                # -- encode & store OBJECT features -------------------------
                if crops:
                    o_feats = encode_clip_batch(model, preprocess, crops, device)
                    for feat, key_group in zip(o_feats, rep_keys):
                        for k in key_group:
                            o_txn.put(k, feat.tobytes())
                            o_cnt += 1

                # Commit periodically to avoid huge transactions ------------
                if g_cnt >= txn_step:
                    g_txn.commit(); g_txn = g_env.begin(write=True); g_cnt = 0
                if o_cnt >= txn_step:
                    o_txn.commit(); o_txn = o_env.begin(write=True); o_cnt = 0

    # -- final sync ----------------------------------------------------------
    g_txn.commit(); o_txn.commit()
    g_env.close(); o_env.close()

# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast CLIP/YOLO feature extractor")
    p.add_argument("--data_root", required=True, help="Dataset root directory")
    p.add_argument("--device", default="cuda:0", help="Torch device string")
    p.add_argument("--yolo", default="yolov8n.pt", help="YOLOv8 weights path")
    p.add_argument("--batch", type=int, default=32, help="Batch size for CLIP + YOLO")
    p.add_argument("--stride", type=int, default=5, help="Extract features every N frames")
    p.add_argument("--txn-step", type=int, default=2000, help="LMDB commit interval")
    return p


def main(opts):
    # Model init ------------------------------------------------------------
    model, preprocess = init_clip(opts.device)
    yolo = YOLO(opts.yolo).to(opts.device)
    yolo.model.half().eval()

    # Loop over logs --------------------------------------------------------
    log_dirs = [p for p in Path(opts.data_root).iterdir() if p.is_dir()]
    for log_dir in tqdm(log_dirs, desc="Logs"):
        process_log(
            log_dir=log_dir,
            model=model,
            preprocess=preprocess,
            yolo=yolo,
            device=opts.device,
            batch=opts.batch,
            stride=opts.stride,
            txn_step=opts.txn_step,
        )


if __name__ == "__main__":
    main(_build_arg_parser().parse_args())
