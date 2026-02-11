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
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from lxml import etree
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
LINE_THICKNESS_PIXELS = 2  # Thickness for converting SVG lines to polygons

def load_class_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def ensure_dirs(out_root: Path, split: str):
    (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

def write_data_yaml(out_root: Path, names: List[str]):
    out_root.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)}
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def svg_to_yolo_polygons(svg_path: Path, img_w: int, img_h: int, class_cfg: Dict):
    """
    Parse CubiCasa5k SVG files to extract polygons.

    Return:
      List[Tuple[class_id, List[float]]] where coords are normalized polygon points:
        [x1, y1, x2, y2, ...] in 0..1
    """
    polygons = []
    
    try:
        # Parse SVG file
        tree = etree.parse(str(svg_path))
        root = tree.getroot()
        
        # Get SVG namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        if root.tag.startswith('{'):
            ns_url = root.tag.split('}')[0].strip('{')
            ns = {'svg': ns_url}
        
        # Build class name to ID mapping
        names = class_cfg.get("names", [])
        merge = class_cfg.get("merge", {})
        enabled = class_cfg.get("enabled", {})
        
        # Create mapping: class_name -> class_id
        class_to_id = {}
        for idx, name in enumerate(names):
            if enabled.get(name, True):
                class_to_id[name] = idx
        
        # Helper function to normalize class name
        def get_class_id(svg_class: str) -> int:
            """Map SVG class to YOLO class ID"""
            svg_class_lower = svg_class.lower()
            
            # Check for walls
            if "wall" in svg_class_lower or "space-boundary" in svg_class_lower:
                target = "Wall"
            # Check for doors
            elif "door" in svg_class_lower:
                target = "Door"
            # Check for windows
            elif "window" in svg_class_lower:
                target = "Window"
            # Check for columns
            elif "column" in svg_class_lower or "pillar" in svg_class_lower:
                target = "Column"
            else:
                return None
            
            # Return class ID if enabled
            return class_to_id.get(target, None)
        
        # Helper to parse points string "x1,y1 x2,y2 ..." into normalized coords
        def parse_points(points_str: str) -> List[float]:
            """Parse SVG points attribute and normalize"""
            coords = []
            # Split by whitespace and commas
            parts = points_str.replace(',', ' ').split()
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    x = float(parts[i]) / img_w
                    y = float(parts[i + 1]) / img_h
                    coords.extend([x, y])
            return coords
        
        # Extract elements with and without namespace
        for elem in root.iter():
            # Skip comments and other non-element nodes
            if not isinstance(elem.tag, str):
                continue
            
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            svg_class = elem.get('class')
            
            if not svg_class:
                continue
            
            class_id = get_class_id(svg_class)
            if class_id is None:
                continue
            
            coords = []
            
            # Parse <polygon> elements
            if tag == 'polygon':
                points_str = elem.get('points', '')
                if points_str:
                    coords = parse_points(points_str)
            
            # Parse <rect> elements
            elif tag == 'rect':
                x = float(elem.get('x', 0))
                y = float(elem.get('y', 0))
                w = float(elem.get('width', 0))
                h = float(elem.get('height', 0))
                # Create polygon from rectangle
                coords = [
                    x / img_w, y / img_h,
                    (x + w) / img_w, y / img_h,
                    (x + w) / img_w, (y + h) / img_h,
                    x / img_w, (y + h) / img_h
                ]
            
            # Parse <line> elements (convert to thin polygon)
            elif tag == 'line':
                x1 = float(elem.get('x1', 0))
                y1 = float(elem.get('y1', 0))
                x2 = float(elem.get('x2', 0))
                y2 = float(elem.get('y2', 0))
                
                # Create a thin rectangle around the line
                dx = x2 - x1
                dy = y2 - y1
                length = (dx**2 + dy**2)**0.5
                
                if length > 0:
                    # Perpendicular vector
                    px = -dy / length * LINE_THICKNESS_PIXELS / 2
                    py = dx / length * LINE_THICKNESS_PIXELS / 2
                    
                    # Four corners of the thin rectangle
                    coords = [
                        (x1 + px) / img_w, (y1 + py) / img_h,
                        (x2 + px) / img_w, (y2 + py) / img_h,
                        (x2 - px) / img_w, (y2 - py) / img_h,
                        (x1 - px) / img_w, (y1 - py) / img_h
                    ]
            
            # Parse <path> elements (basic support for simple paths)
            elif tag == 'path':
                d = elem.get('d', '')
                # Simple path parsing - only handle M and L commands
                if d:
                    path_coords = []
                    parts = d.replace(',', ' ').split()
                    i = 0
                    while i < len(parts):
                        cmd = parts[i]
                        if cmd in ['M', 'L'] and i + 1 < len(parts) - 1:
                            try:
                                x = float(parts[i + 1])
                                y = float(parts[i + 2])
                                path_coords.extend([x / img_w, y / img_h])
                                i += 3
                            except ValueError:
                                i += 1
                        else:
                            i += 1
                    
                    if len(path_coords) >= 6:  # At least 3 points
                        coords = path_coords
            
            # Add polygon if we have valid coordinates
            if coords and len(coords) >= 6:  # At least 3 points (x,y pairs)
                polygons.append((class_id, coords))
    
    except Exception as e:
        logging.warning(f"Error parsing SVG {svg_path}: {e}")
    
    return polygons

def convert_one(sample_img: Path, sample_svg: Path, out_img: Path, out_label: Path, class_cfg: Dict):
    """
    Convert one CubiCasa5k sample to YOLO-seg format.
    - read image to get width/height
    - parse svg -> polygons
    - copy image to out_img
    - write polygons to out_label (one line per instance)
    """
    try:
        # Read image to get dimensions
        img = Image.open(sample_img)
        img_w, img_h = img.size
        img.close()
        
        # Parse SVG to get polygons
        polygons = svg_to_yolo_polygons(sample_svg, img_w, img_h, class_cfg)
        
        # Copy image to output directory
        out_img.write_bytes(sample_img.read_bytes())
        
        # Write YOLO-seg format labels
        label_lines = []
        for class_id, coords in polygons:
            # Format: class_id x1 y1 x2 y2 x3 y3 ...
            coords_str = ' '.join(f"{c:.6f}" for c in coords)
            label_lines.append(f"{class_id} {coords_str}")
        
        out_label.write_text('\n'.join(label_lines) + '\n' if label_lines else '', encoding="utf-8")
        
    except Exception as e:
        logging.error(f"Error converting {sample_img}: {e}")
        # Create empty label file on error
        if not out_label.exists():
            out_label.write_text("", encoding="utf-8")

def find_samples(cubicasa_root: Path, img_ext: str) -> List[Tuple[Path, Path]]:
    """
    Find image-SVG pairs in CubiCasa5k directory structure.
    
    CubiCasa5k structure:
    - high_quality/N/F1_scaled.png + model.svg
    - high_quality_architectural/N/F1_scaled.png + model.svg
    - colorful/N/F1_scaled.png + model.svg
    
    Returns list of (image_path, svg_path) tuples.
    
    Note: When multiple images match a pattern in a directory with an SVG,
    only the first match is selected. This is by design to handle cases where
    multiple image versions exist in the same directory.
    """
    pairs = []
    
    # Look for model.svg files and find corresponding images
    for svg_path in cubicasa_root.rglob("model.svg"):
        # Look for F1_scaled.png in the same directory
        img_dir = svg_path.parent
        
        # Try exact name first
        img_path = img_dir / f"F1_scaled.{img_ext}"
        if img_path.exists():
            pairs.append((img_path, svg_path))
            continue
        
        # Try pattern matching for similar names (F1_*, etc.)
        # Note: Only the first match is selected when multiple images exist
        found = False
        for img_candidate in img_dir.glob(f"F*.{img_ext}"):
            pairs.append((img_candidate, svg_path))
            found = True
            break
        
        # If still not found, try any image in the directory
        if not found:
            for img_candidate in img_dir.glob(f"*.{img_ext}"):
                pairs.append((img_candidate, svg_path))
                break
    
    # Also check for SVG files with same stem as images (fallback)
    for img_path in cubicasa_root.rglob(f"*.{img_ext}"):
        svg_path = img_path.with_suffix(".svg")
        if svg_path.exists() and (img_path, svg_path) not in pairs:
            pairs.append((img_path, svg_path))
    
    logging.info(f"Found {len(pairs)} image/svg pairs")
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
            # Create unique filename using parent directory name
            unique_name = f"{img_path.parent.name}_{img_path.name}"
            out_img = out_root / "images" / split / unique_name
            out_lbl = out_root / "labels" / split / (img_path.parent.name + "_" + img_path.stem + ".txt")
            convert_one(img_path, svg_path, out_img, out_lbl, cfg)

    print(f"[OK] YOLO-seg dataset created at: {out_root}")
    print(f"     data.yaml: {out_root / 'data.yaml'}")

if __name__ == "__main__":
    main()
