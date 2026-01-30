#!/usr/bin/env python3
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--model", default="yolov8n-seg.pt", help="Base model checkpoint")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="e.g. 0 or 'cpu'")
    p.add_argument("--project", default="runs/segment")
    p.add_argument("--name", default="cubicasa_yolov8seg")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

if __name__ == "__main__":
    main()
