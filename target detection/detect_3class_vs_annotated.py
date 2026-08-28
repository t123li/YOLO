"""对比：annotated_images(师兄模型3类框) vs 新3类模型 run_3class 检测框（同一帧左右并排）。

- 左图：annotated_images/<SEQ> 原图（师兄模型检测结果：红=HEAD, 绿=CAR, 黄=WINDOW）
- 右图：run_3class/best.pt（3类检测）画框，每类独立颜色，标签带置信度
- 渲染：细框(1px)、细字(scale=0.4/line=1)、标签防遮挡、半透明底纹
- 输出：target detection/detect_3class_vs_annotated/

用法：
    D:/Anaconda/envs/course_torch/python.exe detect_3class_vs_annotated.py
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


def _rects_overlap(a, b):
    """判断两个矩形 (x1,y1,x2,y2) 是否重叠。"""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def draw_3class(img, r, color_map, cls_names):
    """用3类模型结果画细框+细标签（带置信度，防遮挡）。

    每类独立颜色，标签 = "类名 ID:置信度"，放在框上方，智能避让。
    """
    out = img.copy()
    if r.boxes is None:
        return out

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.4
    THICK = 1
    BOX_THICK = 1

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)

    # 先画所有框（细框）
    dets = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]
    for (x1, y1, x2, y2), c in zip(dets, clss):
        color = color_map.get(c, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICK)

    # 再画标签（防遮挡）
    placed = []
    labels = []
    for (x1, y1, x2, y2), conf, c in zip(dets, confs, clss):
        label = f"{cls_names.get(c, str(c))}:{conf:.2f}"
        labels.append(label)

    for idx, ((x1, y1, x2, y2), label, c) in enumerate(zip(dets, labels, clss)):
        color = color_map.get(c, (255, 255, 255))
        (tw, th), _ = cv2.getTextSize(label, FONT, SCALE, THICK)
        candidates = [
            (x1, y1 - th - 3),
            (x1, y1 - th - 11),
            (x1, y2 + 3),
            (x1, y2 + 11),
            (x1 + 2, y1 + th + 2),
            (x1 - tw - 3, y1 + th),
            (x2 + 3, y1 + th),
        ]
        chosen = candidates[0]
        for (lx, ly) in candidates:
            lrect = (lx, ly, lx + tw, ly + th)
            if any(_rects_overlap(lrect, dets[j]) for j in range(len(dets)) if j != idx):
                continue
            if any(_rects_overlap(lrect, p) for p in placed):
                continue
            chosen = (lx, ly)
            break
        lx, ly = chosen
        ov = out[ly:ly + th, lx:lx + tw]
        if ov.size > 0:
            out[ly:ly + th, lx:lx + tw] = cv2.addWeighted(ov, 0.4, np.zeros_like(ov), 0.6, 0)
        cv2.putText(out, label, (lx, ly + th), FONT, SCALE, color, THICK)
        placed.append((lx, ly, lx + tw, ly + th))
    return out


# ---------- 配置 ----------
SEQ = "20260714151020365"
ANNOTATED = rf"F:/车辆超员检测项目数据集/annotated_images/{SEQ}"
RAW = rf"F:/车辆超员检测项目数据集/超员原图0729/{SEQ}"
MODEL = r"D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target detection/run_3class/weights/best.pt"
OUT = r"D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target detection/detect_3class_vs_annotated"
os.makedirs(OUT, exist_ok=True)

# 3类颜色：0=car 绿, 1=head 红, 2=window 黄（与师兄模型颜色对应）
COLOR_MAP = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 255, 255)}
CLS_NAMES = {0: "car", 1: "head", 2: "window"}

print(f"[加载3类模型] {MODEL}")
model = YOLO(MODEL)
print(f"[类别] {model.names}")

files = sorted(glob.glob(os.path.join(RAW, "*.jpg")))
print(f"[序列] {SEQ} 共 {len(files)} 帧")

for i, raw_path in enumerate(files):
    name = os.path.basename(raw_path)
    annotated_path = os.path.join(ANNOTATED, name)

    left = imread_cn(annotated_path)   # 师兄模型3类框
    raw = imread_cn(raw_path)          # 原图
    r = model.predict(raw, conf=0.25, imgsz=640, verbose=False)[0]
    right = draw_3class(raw, r, COLOR_MAP, CLS_NAMES)

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
    cv2.putText(bar, f"frame {i} | LEFT: 师兄模型(红=head 绿=car 黄=window)  RIGHT: 新3类模型 run_3class  conf>=0.25",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    side = np.vstack([bar, side])
    out_p = os.path.join(OUT, f"vs_f{i:02d}.jpg")
    imwrite_cn(out_p, side)
    print(f"已保存 {out_p}")

print(f"\n完成，对比图输出到: {OUT}")
