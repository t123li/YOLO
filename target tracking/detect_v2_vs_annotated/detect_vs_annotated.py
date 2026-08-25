"""对比：annotated_images 原版红框 vs 我的 100 轮模型检测绿框（同一帧左右并排）。"""
import glob
import os
import sys

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ultralytics import YOLO


def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_cn(p, img):
    ext = os.path.splitext(p)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(p)


def draw(img, r, color):
    out = img.copy()
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().tolist())
            conf = float(box.conf[0].cpu().numpy())
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"head {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


ANNOTATED = r"F:/车辆超员检测项目数据集/annotated_images/20260714151020365"
RAW = r"F:/车辆超员检测项目数据集/超员原图0729/20260714151020365"
MODEL = r"F:/vehicle_dataset/runs/head_v2/weights/best.pt"
OUT = r"F:/vehicle_dataset/detect_v2_vs_annotated"
os.makedirs(OUT, exist_ok=True)

model = YOLO(MODEL)
files = sorted(glob.glob(os.path.join(RAW, "*.jpg")))

for i, raw_path in enumerate(files):
    name = os.path.basename(raw_path)
    annotated_path = os.path.join(ANNOTATED, name)

    left = imread_cn(annotated_path)   # 原版红框(annotated_images)
    raw = imread_cn(raw_path)          # 原图
    right = draw(raw, model.predict(raw, conf=0.25, classes=[0], imgsz=640, verbose=False)[0], (0, 255, 0))

    # 统一高度
    H = max(left.shape[0], right.shape[0])
    def pad_h(img):
        h, w = img.shape[:2]
        if h < H:
            pad = np.full((H - h, w, 3), 255, np.uint8)
            return np.vstack([img, pad])
        return img

    side = np.hstack([pad_h(left), pad_h(right)])
    bar = np.full((40, side.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"frame {i}  |  LEFT: annotated(原版红框)  RIGHT: my head_v2(绿框)  conf>=0.25",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    side = np.vstack([bar, side])
    out_p = os.path.join(OUT, f"vs_f{i:02d}.jpg")
    imwrite_cn(out_p, side)
    print(f"已保存 {out_p}")

print(f"\n完成，对比图输出到: {OUT}")
