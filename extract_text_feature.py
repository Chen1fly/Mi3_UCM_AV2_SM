#!/usr/bin/env python3
# refAV/scripts/extract_text_features.py
import argparse, json, lmdb, torch, open_clip
from pathlib import Path
from tqdm import tqdm
import os
PROXY = "http://127.0.0.1:7890"
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ[k] = PROXY
def init_clip(device):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    return model.to(device).eval()

@torch.inference_mode()
def encode(model, texts, device):
    toks = open_clip.tokenize(texts).to(device)
    feats = model.encode_text(toks).cpu().half().numpy()
    return dict(zip(texts, feats))

def main(opts):
    device = opts.device
    model = init_clip(device)

    desc_map = json.load(open(opts.desc_json))
    all_desc = sorted({d for lst in desc_map.values() for d in lst})
    print(f"Unique descriptions: {len(all_desc)}")

    env = lmdb.open(str(Path(opts.out_dir) / "clip_text.lmdb"),
                    map_size=1 << 34)   # 16 GB
    B = 256
    for i in tqdm(range(0, len(all_desc), B)):
        batch = all_desc[i:i+B]
        feats = encode(model, batch, device)
        with env.begin(write=True) as txn:
            for t, f in feats.items():
                txn.put(t.encode(), f.tobytes())
    env.sync(); env.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc_json", required=True, help="descriptions.json 路径")
    ap.add_argument("--out_dir", required=True, help="保存 LMDB 的目录")
    ap.add_argument("--device", default="cuda:0")
    main(ap.parse_args())
