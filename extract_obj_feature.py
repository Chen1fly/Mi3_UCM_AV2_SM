# extract_object_clip.py
from __future__ import annotations
import os, argparse
from pathlib import Path
from typing import List, Tuple

import h5py, torch, numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.ops as ops

# ---------- 代理 ----------
PROXY = "http://127.0.0.1:7890"
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ[k] = PROXY
# ---------------------------

# ---------- 依赖 ----------
import open_clip                        # 图像编码
from ultralytics import YOLO           # 检测模型（v8）
# ---------------------------

################################################################################
# 参数
################################################################################
def parse_args():
    ap = argparse.ArgumentParser("Extract object-level CLIP features")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--yolo-weights", default="yolov8m.pt")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO score thresh")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()

################################################################################
# 核心函数
################################################################################
def detect_objects(detector:YOLO, img:np.ndarray, conf:float, device:str):
    """YOLO 返回 xyxy, conf, cls"""
    res = detector.predict(img, verbose=False, device=device, conf=conf, iou=0.5)[0]
    if res.boxes is None or len(res.boxes)==0:
        return np.empty((0,4),dtype=np.float32), np.empty((0,),dtype=np.int64), np.empty((0,),dtype=np.float32)
    boxes = res.boxes.xyxy.cpu().numpy()        # (N,4)
    scores= res.boxes.conf.cpu().numpy()
    labels= res.boxes.cls.cpu().numpy().astype(np.int64)
    return boxes, labels, scores

def clip_patches(
    clip_model, preprocess, device:str,
    img_pil:Image.Image,
    boxes:np.ndarray, batch_size:int
)->np.ndarray:
    """裁剪 bbox → CLIP features (N,D)"""
    if len(boxes)==0:
        return np.empty((0, clip_model.visual.output_dim), dtype=np.float16)

    # ROI 裁剪并预处理
    patches = []
    for xyxy in boxes:
        x1,y1,x2,y2 = map(int, xyxy)
        patch = img_pil.crop((x1,y1,x2,y2))
        patches.append(preprocess(patch))
    feats = []
    for i in range(0,len(patches), batch_size):
        batch = torch.stack(patches[i:i+batch_size]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16 if device=="cuda" else torch.float32):
            f = clip_model.encode_image(batch)
            f = (f / f.norm(dim=-1, keepdim=True)).cpu().half().numpy()
        feats.append(f)
    return np.concatenate(feats,0)   # (N,D)

################################################################################
# 主流程
################################################################################
def main():
    args = parse_args()
    device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"

    # 1. Load models
    yolo = YOLO(args.yolo_weights)
    if device=="cuda":
        yolo.to("cuda")
    clip_model,_, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    clip_model.eval()

    root, fout = Path(args.data_root), Path(args.feature_root)
    track_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    for track in tqdm(track_dirs, desc="Tracks"):
        cam_root = track / "sensors" / "cameras"
        if not cam_root.exists():
            continue
        for cam_dir in sorted(cam_root.iterdir()):
            if not cam_dir.is_dir(): continue
            direction = cam_dir.name
            img_paths = sorted(cam_dir.glob("*.jpg"))
            if not img_paths: continue

            out_h5 = fout / track.name / f"{direction}.h5"
            if out_h5.exists() and not args.overwrite:
                with h5py.File(out_h5,"r") as f:
                    if "features" in f and f["timestamps"].shape[0]==len(img_paths):
                        continue  # skip
            out_h5.parent.mkdir(parents=True, exist_ok=True)

            all_feats, all_ts, all_boxes, all_lbl = [],[],[],[]
            for p in tqdm(img_paths, leave=False):
                ts = int(p.stem)
                img_pil = Image.open(p).convert("RGB")
                img_np  = np.asarray(img_pil)
                boxes, labels, _ = detect_objects(yolo, img_np, args.conf, device)
                feats = clip_patches(clip_model, preprocess, device, img_pil, boxes, args.batch_size)
                # 追加
                all_ts   .extend([ts]*len(boxes))
                all_boxes.extend(boxes)
                all_lbl  .extend(labels)
                all_feats.append(feats)

            if len(all_feats)==0:          # 没检测到目标
                continue
            feats_np = np.concatenate(all_feats,0)
            ts_np    = np.asarray(all_ts ,dtype=np.int64)
            boxes_np = np.asarray(all_boxes,dtype=np.float32)
            lbl_np   = np.asarray(all_lbl,dtype=np.int64)

            # 保存
            with h5py.File(out_h5,"w") as f:
                f.create_dataset("features", data=feats_np, compression="gzip")
                f.create_dataset("timestamps", data=ts_np, compression="gzip")
                f.create_dataset("bboxes", data=boxes_np, compression="gzip")
                f.create_dataset("labels", data=lbl_np, compression="gzip")
            print(f"[saved] {out_h5.relative_to(fout)}")

if __name__=="__main__":
    main()
