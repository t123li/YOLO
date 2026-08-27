"""对比：annotated_images(师兄模型检测框) vs 新模型 run_head_seqsplit 检测框（同一帧左右并排）。

- 左图：annotated_images/20260714151020365 原图（师兄模型检测结果，红框=head，绿框=car，黄框=car_window）
- 右图：run_head_seqsplit/best.pt（序列级划分训练的新模型）检测，只画 head 框（绿）
- 对比目的：看两个模型在 head 检测上的差异（漏检/误检/框位）
- 输出：target detection/detect_vs_annotated_2model/
"""
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


def draw_head(img, r, color):
    """在图上画 head 检测框（模型输出），只画 class 0。"""
    out = img.copy()
    if r.boxes is not None:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().tolist())
            conf = float(box.conf[0].cpu().numpy())
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"head {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


# 序列：20260714151020365（只对比这一个序列）
SEQ = "20260714151020365"

# 师兄模型结果：annotated_images（红框 head / 绿框 car / 黄框 car_window）
ANNOTATED = rf"F:/车辆超员检测项目数据集/annotated_images/{SEQ}"
# 原图（右图用原图画新模型检测框）
RAW = rf"F:/车辆超员检测项目数据集/超员原图0729/{SEQ}"
# 新模型：run_head_seqsplit/best.pt（序列级划分训练）
MODEL = r"D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target detection/run_head_seqsplit/weights/best.pt"
# 输出目录：target detection 下
OUT = r"D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target detection/detect_vs_annotated_2model"
os.makedirs(OUT, exist_ok=True)

print(f"[加载新模型] {MODEL}")
model = YOLO(MODEL)

# 用 annotated 目录里的文件名（0_original.jpg ~ 7_original.jpg）
files = sorted(glob.glob(os.path.join(RAW, "*.jpg")))
print(f"[序列] {SEQ} 共 {len(files)} 帧")

for i, raw_path in enumerate(files):
    name = os.path.basename(raw_path)
    annotated_path = os.path.join(ANNOTATED, name)

    left = imread_cn(annotated_path)   # 师兄模型检测框（红/绿/黄）
    raw = imread_cn(raw_path)          # 原图
    right = draw_head(raw, model.predict(raw, conf=0.25, classes=[0], imgsz=640, verbose=False)[0], (0, 255, 0))

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
    cv2.putText(bar, f"frame {i} | LEFT: 师兄模型(红=head 绿=car 黄=window)  RIGHT: 新模型 run_head_seqsplit(绿=head) conf>=0.25",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    side = np.vstack([bar, side])
    out_p = os.path.join(OUT, f"vs_f{i:02d}.jpg")
    imwrite_cn(out_p, side)
    print(f"已保存 {out_p}")

print(f"\n完成，对比图输出到: {OUT}")
