PY=python

.PHONY: install convert train

install:
	$(PY) -m pip install -r requirements.txt

convert:
	$(PY) scripts/convert_cubicasa_to_yolo_seg.py --cubicasa_root data/raw/CubiCasa5k --out_root data/yolo

train:
	$(PY) scripts/train_yolov8_seg.py --data data/yolo/data.yaml --model yolov8n-seg.pt --imgsz 1024 --epochs 100 --batch 8
