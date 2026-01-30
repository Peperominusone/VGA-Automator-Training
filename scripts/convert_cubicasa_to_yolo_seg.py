#!/usr/bin/env python3

"""
CubiCasa5k -> YOLOv8-seg converter (skeleton)

This script is intentionally conservative: it creates a clean YOLO-seg dataset layout and
provides placeholders where CubiCasa-specific parsing logic should be implemented.

Expected output (YOLOv8-seg):
- images/<split>/*.png
- labels/<split>/*.txt  (each line: class_id x1 y1 x2 y2 ... ; normalized to [0,1])

Notes:
- CubiCasa5k annotations are commonly delivered as SVG polygons. You must parse SVG and map
  categories to our target classes (Wall/Door/Window/Column).
- If you use a different CubiCasa5k fork/version, adjust the parser accordingly.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm

def load_class_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def ensure_dirs(out_root: Path, split: str):
    (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

def write_data_yaml(out_root: Path, names: List[str]):
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)}
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def svg_to_yolo_polygons(svg_path: Path, img_w: int, img_h: int, class_cfg: Dict):
    """
    TODO: Implement SVG parsing for CubiCasa5k.

    Return:
      List[Tuple[class_id, List[float]]] where coords are normalized polygon points:
        [x1, y1, x2, y2, ...] in 0..1
    """
    # Placeholder: return no labels
    return []

def convert_one(sample_img: Path, sample_svg: Path, out_img: Path, out_label: Path, class_cfg: Dict):
    """
    TODO: Implement:
    - read image to get width/height (cv2 or PIL)
    - parse svg -> polygons
    - copy image to out_img
    - write polygons to out_label (one line per instance)
    """
    # Copy image (simple file copy)
    out_img.write_bytes(sample_img.read_bytes())

    # Placeholder label file
    out_label.write_text("", encoding="utf-8")

def find_samples(cubicasa_root: Path, img_ext: str) -> List[Tuple[Path, Path]]:
    """
    TODO: CubiCasa5k 구조에 맞춰 이미지-주석(SVG) 쌍을 찾도록 수정.
    아래는 '이미지와 동일한 stem의 .svg가 있다'는 가정으로 작성된 기본 검색기.
    """
    pairs = []
    for img_path in cubicasa_root.rglob(f"*.{img_ext}"):
        svg_path = img_path.with_suffix(".svg")
        if svg_path.exists():
            pairs.append((img_path, svg_path))
    return pairs

def split_pairs(pairs: List[Tuple[Path, Path]], train_ratio: float = 0.9):
    n = len(pairs)
    n_train = int(n * train_ratio)
    return pairs[:n_train], pairs[n_train:]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cubicasa_root", required=True, type=str)
    p.add_argument("--out_root", required=True, type=str)
    p.add_argument("--config", default="configs/classes.json", type=str)
    p.add_argument("--img_ext", default="png", type=str)
    p.add_argument("--train_ratio", default=0.9, type=float)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    args = p.parse_args()

    cubicasa_root = Path(args.cubicasa_root)
    out_root = Path(args.out_root)
    cfg = load_class_config(Path(args.config))
    names = [n for n in cfg["names"] if cfg.get("enabled", {}).get(n, True)]

    pairs = find_samples(cubicasa_root, args.img_ext)
    if len(pairs) == 0:
        raise RuntimeError(
            f"No image/svg pairs found under {cubicasa_root}. "
            f"Adjust find_samples() to match CubiCasa5k folder structure."
        )

    train_pairs, val_pairs = split_pairs(pairs, args.train_ratio)
    write_data_yaml(out_root, names)

    for split, split_pairs_list in [("train", train_pairs), ("val", val_pairs)]:
        ensure_dirs(out_root, split)
        for img_path, svg_path in tqdm(split_pairs_list, desc=f"Converting {split}"):
            out_img = out_root / "images" / split / img_path.name
            out_lbl = out_root / "labels" / split / (img_path.stem + ".txt")
            convert_one(img_path, svg_path, out_img, out_lbl, cfg)

    print(f"[OK] YOLO-seg dataset created at: {out_root}")
    print(f"     data.yaml: {out_root / 'data.yaml'}")

if __name__ == "__main__":
    main()
