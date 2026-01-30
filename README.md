# VGA-Automator-Training

CubiCasa5k 기반 **YOLOv8-seg** 학습 레포입니다.  
목표: `best.pt`(segmentation weights) 생성 → 앱 레포(VGA-Automator)에 그대로 교체 적용.

## TL;DR
1) CubiCasa5k 다운로드/배치  
2) `scripts/convert_cubicasa_to_yolo_seg.py`로 YOLO-seg 라벨 생성  
3) `scripts/train_yolov8_seg.py`로 학습 → `runs/segment/.../weights/best.pt`

---

## 1. 환경 준비

### Python
- Python 3.10+ 권장

### 설치
```bash
pip install -r requirements.txt
```

---

## 2. 데이터 준비 (CubiCasa5k)

### 폴더 배치
아래 경로에 CubiCasa5k를 풀어주세요(원본 그대로).
```
data/raw/CubiCasa5k/
  ├─ (CubiCasa5k 원본 파일들...)
```

> 주의: 데이터셋은 용량이 커서 git에 올리지 않는 것을 권장합니다. `.gitignore`에 포함되어 있습니다.

---

## 3. CubiCasa5k → YOLOv8-seg 변환

### 기본 클래스(권장)
- Wall
- Door (Sliding Door 포함을 권장)
- Window
- Column (선택)

변환 옵션은 `configs/classes.json`에서 수정합니다.

### 변환 실행
```bash
python scripts/convert_cubicasa_to_yolo_seg.py \
  --cubicasa_root data/raw/CubiCasa5k \
  --out_root data/yolo \
  --splits train val \
  --img_ext png
```

생성 결과(예시):
```
data/yolo/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt   # YOLO-seg polygon labels
  labels/val/*.txt
  data.yaml
```

---

## 4. 학습 (YOLOv8-seg)

```bash
python scripts/train_yolov8_seg.py \
  --data data/yolo/data.yaml \
  --model yolov8n-seg.pt \
  --imgsz 1024 \
  --epochs 100 \
  --batch 8
```

학습 후:
- `runs/segment/<run_name>/weights/best.pt`

---

## 5. 앱 레포에 적용

앱 레포에서 기존 `best.pt`를 위에서 생성한 `best.pt`로 교체한 뒤,
`result.masks is not None`가 항상 True인지 디버그 로그로 확인하세요.

---

## 6. (중요) 변환 스크립트는 “골격”입니다

CubiCasa5k의 SVG/annotation 구조는 버전/샘플에 따라 차이가 있을 수 있습니다.
이 레포의 변환기는 아래를 빠르게 확장할 수 있도록 최소 골격으로 제공됩니다.

- SVG 파싱
- 폴리곤 좌표 정규화
- 클래스 매핑
- train/val 분리

필요하면 너의 도면 스타일/추론 파이프라인에 맞춰
- label smoothing
- 작은 객체 제거
- 방/공간 클래스 추가
까지 확장하면 됩니다.
