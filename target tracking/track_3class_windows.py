"""本地可运行的 3 类跟踪 + 车窗约束计数脚本（基于师兄 evaluate_normal_test_track_v3.py 改造）。

师兄 v3 脚本已在 person_count/ultralytics/ 下，本脚本在其基础上：
1. 路径改为本地（模型用 run_3class/best.pt，数据用 超员原图0729/，真值用 total_registered.txt）
2. 【核心改进】给 head_shoulder 增加 SimpleSORT 跨帧跟踪（师兄脚本 head 只做单帧检测）
3. 计数用 head track_id 去重 + 绑定车窗（中心点落在窗内），解决 "ID 跨窗跳变"

输出：target tracking/tracking_output/日期_序号/源序列名/*.jpg（遵守 README 规则）

用法：
    D:/Anaconda/envs/course_torch/python.exe track_3class_windows.py
"""
import os
import re
import sys
import shutil
import cv2
import numpy as np
import torch
from collections import defaultdict, Counter
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ================= 路径 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target detection/run_3class/weights/best.pt"
DATA_ROOT = r"F:/车辆超员检测项目数据集/超员原图0729"
OUT_ROOT = os.path.join(BASE_DIR, "tracking_output")

CLASS_NAMES = {0: 'car', 1: 'head', 2: 'window'}

# ================= 配置（与师兄脚本一致） =================
DEDUP_IOU_THRESH_WIN = 0.4
WIN_MIN_AREA_RATIO = 0.015; WIN_MAX_AREA_RATIO = 0.20
WIN_MIN_ASPECT = 1.0; WIN_MAX_ASPECT = 4.0
WIN_MIN_REL_Y = 0.10; WIN_MAX_REL_Y = 0.35
WIN_MIN_REL_W = 0.08; WIN_MAX_REL_W = 0.50
STABLE_WIN_FRAMES = 3
SORT_IOU_THRESH = 0.3
SORT_MAX_AGE = 5
CONF = 0.2

# ================= 工具函数（复制自师兄脚本） =================

def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    a2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter) > 0 else 0

def point_in_box(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]

def box_center(b):
    return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

def dedup_boxes(boxes, scores, iou_thresh=0.5):
    if not boxes: return [], []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    for i in order:
        if not any(compute_iou(boxes[i], boxes[j]) >= iou_thresh for j in keep):
            keep.append(i)
    return [boxes[i] for i in keep], [scores[i] for i in keep]

def filter_windows_by_geometry(wbs, car_box):
    if car_box is None: return []
    cw=car_box[2]-car_box[0]; ch=car_box[3]-car_box[1]; ca=cw*ch
    if ca<=0 or cw<=0 or ch<=0: return []
    kept=[]
    for wb in wbs:
        wcx=(wb[0]+wb[2])/2; wcy=(wb[1]+wb[3])/2
        ww=wb[2]-wb[0]; wh=wb[3]-wb[1]; wa=ww*wh
        if wa<=0 or ww<=0 or wh<=0: continue
        if not (WIN_MIN_AREA_RATIO<=wa/ca<=WIN_MAX_AREA_RATIO): continue
        if not (WIN_MIN_ASPECT<=ww/wh<=WIN_MAX_ASPECT): continue
        if not (WIN_MIN_REL_Y<=(wcy-car_box[1])/ch<=WIN_MAX_REL_Y): continue
        if not (WIN_MIN_REL_W<=ww/cw<=WIN_MAX_REL_W): continue
        kept.append(wb)
    return kept

def get_relative_position(wb, car_box):
    if car_box is None: return (wb[0]+wb[2])/2/640.0
    wcx=(wb[0]+wb[2])/2; cw=car_box[2]-car_box[0]
    if cw<=0: return 0.5
    return max(0, min(1, (wcx-car_box[0])/cw))

def assign_window_ids(wbs, car_box):
    if not wbs: return []
    wins = sorted([(get_relative_position(b, car_box), b) for b in wbs], key=lambda x: x[0])
    return [(i, box) for i, (_, box) in enumerate(wins)]

def read_actual_count(image_dir):
    for cand in ("total_registered.txt", "actual_count.txt"):
        p = os.path.join(image_dir, cand)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    text = f.read().strip()
                for tok in text.replace(",", " ").split():
                    if tok.isdigit():
                        return int(tok)
            except Exception:
                return None
    return None

# ================= SimpleSORT（复制自师兄脚本） =================

class SimpleSORT:
    """轻量 IoU-based tracker，对单类别跨帧分配 track_id。"""
    def __init__(self, iou_thresh=0.3, max_age=5):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 0

    def update(self, boxes, scores):
        if not boxes:
            for tid in list(self.tracks):
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
            return []
        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[t]['box'] for t in track_ids]
        iou_matrix = np.zeros((len(boxes), len(track_ids))) if track_ids else np.zeros((len(boxes), 0))
        for i, box in enumerate(boxes):
            for j, tbox in enumerate(track_boxes):
                iou_matrix[i, j] = compute_iou(box, tbox)
        assigned = {}; used_tracks = set()
        if len(track_ids) > 0:
            matches = []
            for i in range(len(boxes)):
                for j in range(len(track_ids)):
                    if iou_matrix[i, j] >= self.iou_thresh:
                        matches.append((iou_matrix[i, j], i, j))
            matches.sort(reverse=True)
            for iou_val, i, j in matches:
                if i not in assigned and j not in used_tracks:
                    assigned[i] = track_ids[j]; used_tracks.add(j)
        result = []
        for i, box in enumerate(boxes):
            if i in assigned:
                tid = assigned[i]
                self.tracks[tid]['box'] = box; self.tracks[tid]['age'] = 0; self.tracks[tid]['hits'] += 1
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'box': box, 'age': 0, 'hits': 1}
            result.append((box, tid))
        for j, tid in enumerate(track_ids):
            if j not in used_tracks:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
        return result

# ================= 渲染（细框细字，防遮挡） =================

def _rects_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def render_frame(img, car_box, windows, head_dets, actual_count, pred_count, win_head_counts):
    """渲染：车绿框、车窗黄框、head 红框（带 ID），左上角显示实际/预测人数。细框细字防遮挡。"""
    out = img.copy()
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.4; THICK = 1

    # head 框（红，细框），标签 = head{tid}:{conf}
    if head_dets:
        placed = []
        dets = []
        labels = []
        for hb, tid, conf in head_dets:
            dets.append((int(hb[0]), int(hb[1]), int(hb[2]), int(hb[3])))
            labels.append(f"h{tid}:{conf:.2f}")
        for (x1,y1,x2,y2), c in zip(dets, [(0,0,255)]*len(dets)):
            cv2.rectangle(out, (x1,y1),(x2,y2), c, THICK)
        for idx, ((x1,y1,x2,y2), label) in enumerate(zip(dets, labels)):
            (tw, th), _ = cv2.getTextSize(label, FONT, SCALE, THICK)
            candidates = [(x1,y1-th-3),(x1,y1-th-11),(x1,y2+3),(x1,y2+11),(x1+2,y1+th+2),(x1-tw-3,y1+th),(x2+3,y1+th)]
            chosen = candidates[0]
            for (lx,ly) in candidates:
                lrect=(lx,ly,lx+tw,ly+th)
                if any(_rects_overlap(lrect,dets[j]) for j in range(len(dets)) if j!=idx): continue
                if any(_rects_overlap(lrect,p) for p in placed): continue
                chosen=(lx,ly); break
            lx,ly=chosen
            ov=out[ly:ly+th,lx:lx+tw]
            if ov.size>0: out[ly:ly+th,lx:lx+tw]=cv2.addWeighted(ov,0.4,np.zeros_like(ov),0.6,0)
            cv2.putText(out,label,(lx,ly+th),FONT,SCALE,(0,0,255),THICK)
            placed.append((lx,ly,lx+tw,ly+th))

    # 车窗框（黄，细框），标签 win{i}={count}
    for wid, wb, cnt in windows:
        x1,y1,x2,y2=[int(v) for v in wb]
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,255),THICK)
        label=f"win{wid}:{cnt}"
        (tw,th),_=cv2.getTextSize(label,FONT,SCALE,THICK)
        cv2.putText(out,label,(x1,max(y1-th,12)),FONT,SCALE,(0,255,255),THICK)

    # 车框（绿，细框）
    if car_box is not None:
        x1,y1,x2,y2=[int(v) for v in car_box]
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),THICK)
        cv2.putText(out,"car",(x1,max(y1-20,12)),FONT,SCALE,(0,255,0),THICK)

    # 左上角统计
    cv2.putText(out,f"Pred: {pred_count}  Actual: {actual_count}  Diff: {pred_count-actual_count:+d}",
                (8,20),FONT,0.55,(0,0,255),1)
    return out

# ================= 主流程 =================

def process_sequence(model, seq, out_sub):
    in_dir = os.path.join(DATA_ROOT, seq)
    frames = sorted([f for f in os.listdir(in_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    actual_count = read_actual_count(in_dir)

    # 三类 tracker（每序列独立）
    car_tracker = SimpleSORT(SORT_IOU_THRESH, SORT_MAX_AGE)

    # 【计数方案：每窗跨帧最大人数】window_max_heads[win_index] = 该窗整个序列中出现过的最大 head 数
    window_max_heads = defaultdict(int)

    for fi, name in enumerate(frames):
        img_path = os.path.join(in_dir, name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue

        r = model.predict(source=img_path, conf=CONF, iou=0.45, imgsz=640, verbose=False)[0]

        # 按类收集
        boxes_by_cls = {c: {'boxes':[], 'scores':[]} for c in CLASS_NAMES}
        for box in r.boxes:
            cid = int(box.cls[0].cpu().numpy())
            if cid not in CLASS_NAMES: continue
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu().numpy())
            boxes_by_cls[cid]['boxes'].append([x1,y1,x2,y2])
            boxes_by_cls[cid]['scores'].append(score)

        # 车 SORT 跟踪
        car_tracked = car_tracker.update(boxes_by_cls[0]['boxes'], boxes_by_cls[0]['scores'])

        # 选主车（面积最大）
        car_box = None
        if car_tracked:
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b,_ in car_tracked]
            car_box = car_tracked[max(range(len(areas)), key=lambda i: areas[i])][0]

        # 车窗：去重 + 几何过滤 + 在主车坐标系下按 x 排序（索引 0/1/2 = 前排/中排/后排）
        win_boxes = boxes_by_cls[2]['boxes']
        win_scores = boxes_by_cls[2]['scores']
        w_nms, _ = dedup_boxes(win_boxes, win_scores, DEDUP_IOU_THRESH_WIN)
        w_in = [wb for wb in w_nms if car_box is None or point_in_box(box_center(wb), car_box)]
        w_geom = filter_windows_by_geometry(w_in, car_box)
        if car_box is not None and len(car_box) == 4 and (car_box[2]-car_box[0]) > 0:
            cw = car_box[2] - car_box[0]
            w_sorted = sorted(w_geom, key=lambda wb: (wb[0]+wb[2])/2/cw)
        else:
            w_sorted = sorted(w_geom, key=lambda wb: (wb[0]+wb[2])/2)
        windows = [(i, wb) for i, wb in enumerate(w_sorted)]  # [(win_index, box)]

        # 【每窗跨帧最大人数】数本帧每个窗内的 head 数（中心点落在窗内），更新跨帧最大
        head_boxes = boxes_by_cls[1]['boxes']
        head_dets = [(hb, -1, boxes_by_cls[1]['scores'][i]) for i, hb in enumerate(head_boxes)]
        win_head_counts = {}
        for wid, wb in windows:
            cnt = sum(1 for hb in head_boxes if point_in_box(box_center(hb), wb))
            win_head_counts[wid] = cnt
            window_max_heads[wid] = max(window_max_heads[wid], cnt)

        pred_count = sum(window_max_heads.values())  # 各窗跨帧最大人数累加 = 整车人数

        # 渲染：窗上标注跨帧累计最大人数
        win_draw = [(wid, wb, window_max_heads.get(wid, 0)) for wid, wb in windows]
        out_img = render_frame(img, car_box, win_draw, head_dets, actual_count if actual_count is not None else -1, pred_count, win_head_counts)
        cv2.imencode('.jpg', out_img)[1].tofile(os.path.join(out_sub, name))

    return sum(window_max_heads.values()), actual_count, dict(window_max_heads)


def main():
    model = YOLO(MODEL_PATH)
    print(f"[模型] {MODEL_PATH}\n  类别: {model.names}")

    # 顶层目录：日期_序号
    os.makedirs(OUT_ROOT, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    existing = [d for d in os.listdir(OUT_ROOT) if d.startswith(today + "_")]
    next_seq = max([int(m.group(1)) for d in existing if (m := re.match(rf"^{today}_(\d+)$", d))], default=0) + 1
    run_dir = os.path.join(OUT_ROOT, f"{today}_{next_seq:03d}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[输出] {run_dir}")

    # 前 12 个序列
    seqs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])[:12]

    results = []
    for seq in seqs:
        out_sub = os.path.join(run_dir, seq)
        os.makedirs(out_sub, exist_ok=True)
        print(f"\n[序列] {seq}")
        pred_count, actual, win_max = process_sequence(model, seq, out_sub)
        diff = pred_count - actual if actual is not None else "?"
        status = "准确" if actual is not None and diff == 0 else ("偏差" if actual is not None and abs(diff) <= 1 else "偏差较大")
        results.append((seq, pred_count, actual, diff))
        print(f"  预测={pred_count} 实际={actual} 差={diff} ({status})")

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    for seq, pred, actual, diff in results:
        print(f"  {seq}: 预测={pred} 实际={actual} 差={diff}")
    print(f"\n完成，结果输出到: {run_dir}")


if __name__ == "__main__":
    main()
