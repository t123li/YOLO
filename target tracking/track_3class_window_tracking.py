"""C 方案：3 类检测 + 车窗约束 + 窗内跨帧跟踪（真正的跟踪，head 有跨帧 ID 且不跳窗）。

方案 C 核心：
- 每帧检测 3 类（car / head / window），head 绑定到它所在的车窗（中心点落在窗内）。
- 【真正的跟踪】每个车窗维护独立的窗内轨迹池：用"窗内归一化坐标 (rel_x, rel_y)"做跨帧关联，
  给每个 head 分配稳定 ID。同一窗内的 head 通过窗内位置关联维持 ID，跨窗绝对不共享 ID（不跳窗）。
- 计数：跨帧唯一的 head ID 总数 = 车内人数。

与"每窗跨帧最大人数"（非跟踪）的区别：本方案给 head 分配了跨帧 ID，能追踪单个目标，
而不是只取每窗最大数。

输出：target tracking/tracking_output/日期_序号/源序列名/*.jpg（遵守 README 规则）

用法：
    D:/Anaconda/envs/course_torch/python.exe track_3class_window_tracking.py
"""
import os
import re
import sys
import cv2
import numpy as np
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO

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

# ================= 配置 =================
DEDUP_IOU_THRESH_WIN = 0.4
WIN_MIN_AREA_RATIO = 0.015; WIN_MAX_AREA_RATIO = 0.20
WIN_MIN_ASPECT = 1.0; WIN_MAX_ASPECT = 4.0
WIN_MIN_REL_Y = 0.10; WIN_MAX_REL_Y = 0.35
WIN_MIN_REL_W = 0.08; WIN_MAX_REL_W = 0.50
CONF = 0.2
# 窗内关联门限（窗内归一化距离），超过视为不同人
WIN_ASSOC_DIST = 0.25
# 窗内轨迹最大丢失帧数
WIN_TRACK_MAX_AGE = 3

# ================= 工具 =================

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


def _rects_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def render_frame(img, car_box, windows, head_dets, actual_count, pred_count):
    """渲染：车绿框、车窗黄框（含人数）、head 红框（含ID）。细框细字防遮挡。"""
    out = img.copy()
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.4; THICK = 1

    # head 框（红，细框），标签只显示全局 ID：h{id}
    if head_dets:
        placed = []
        dets = []
        labels = []
        for hb, hid, conf in head_dets:
            dets.append((int(hb[0]), int(hb[1]), int(hb[2]), int(hb[3])))
            labels.append(f"h{hid}")
        for (x1,y1,x2,y2) in dets:
            cv2.rectangle(out, (x1,y1),(x2,y2),(0,0,255),THICK)
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

    # 车窗框（黄，细框），标签 win{i}:{count}
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

    cv2.putText(out,f"Pred: {pred_count}  Actual: {actual_count}  Diff: {pred_count-actual_count:+d}",
                (8,20),FONT,0.55,(0,0,255),1)
    return out


# ================= 全局 ID 分配器 + 窗内跟踪 =================

class GlobalIdAllocator:
    """全局唯一 ID 分配器（整个序列所有目标共用一个）。"""
    def __init__(self):
        self.next_id = 0

    def new_id(self):
        i = self.next_id
        self.next_id += 1
        return i


class WindowTracker:
    """窗内 head 轨迹池，用窗内归一化坐标做跨帧关联，ID 从全局分配器取（全局唯一、不跨窗）。"""
    def __init__(self, id_allocator, assoc_dist=WIN_ASSOC_DIST, max_age=WIN_TRACK_MAX_AGE):
        self.alloc = id_allocator
        self.assoc_dist = assoc_dist
        self.max_age = max_age
        self.tracks = {}   # head_id -> {'pos': (rel_x, rel_y), 'age': 0, 'box': (x1,y1,x2,y2)}
        self.win_index = None

    def update(self, heads):
        """heads: list of (rel_x, rel_y, box)。返回 list of (box, head_id, rel_x, rel_y)。"""
        assigned = {}   # head_idx -> head_id
        used_ids = set()
        for i, (rx, ry, box) in enumerate(heads):
            best_id = None; best_d = self.assoc_dist
            for hid, t in self.tracks.items():
                if hid in used_ids: continue
                d = np.hypot(rx - t['pos'][0], ry - t['pos'][1])
                if d < best_d:
                    best_d = d; best_id = hid
            if best_id is not None:
                assigned[i] = best_id; used_ids.add(best_id)

        result = []
        new_tracks = {}
        for i, (rx, ry, box) in enumerate(heads):
            if i in assigned:
                hid = assigned[i]
                self.tracks[hid]['pos'] = (rx, ry)
                self.tracks[hid]['box'] = box
                self.tracks[hid]['age'] = 0
                new_tracks[hid] = self.tracks[hid]
            else:
                hid = self.alloc.new_id()  # 全局唯一 ID
                self.tracks[hid] = {'pos': (rx, ry), 'box': box, 'age': 0}
                new_tracks[hid] = self.tracks[hid]
            result.append((box, hid, rx, ry))
        # 未匹配的旧轨迹 aging
        for hid, t in self.tracks.items():
            if hid not in new_tracks:
                t['age'] += 1
                if t['age'] <= self.max_age:
                    new_tracks[hid] = t
        self.tracks = new_tracks
        return result


# ================= 主流程 =================

class SimpleCarTracker:
    """极简车跟踪：选面积最大的 car 框。"""
    def update(self, boxes):
        if not boxes: return None
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        return boxes[max(range(len(areas)), key=lambda i: areas[i])]


def process_sequence(model, seq, out_sub):
    in_dir = os.path.join(DATA_ROOT, seq)
    frames = sorted([f for f in os.listdir(in_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
    actual_count = read_actual_count(in_dir)
    car_tracker = SimpleCarTracker()

    # 全局 ID 分配器：整个序列所有 head 共用一个（同帧同类 ID 不重复、跨帧唯一）
    head_alloc = GlobalIdAllocator()
    # 每个窗一个 WindowTracker（关联限制在窗内，ID 从全局分配器取）
    win_trackers = {}
    # 每窗已确认的全局 head ID 集合（跨帧），用于显示每窗人数
    win_seen_ids = defaultdict(set)

    for fi, name in enumerate(frames):
        img_path = os.path.join(in_dir, name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        r = model.predict(source=img_path, conf=CONF, iou=0.45, imgsz=640, verbose=False)[0]

        boxes_by_cls = {c: {'boxes':[]} for c in CLASS_NAMES}
        for box in r.boxes:
            cid = int(box.cls[0].cpu().numpy())
            if cid not in CLASS_NAMES: continue
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().tolist()
            boxes_by_cls[cid]['boxes'].append([x1,y1,x2,y2])

        car_box = car_tracker.update(boxes_by_cls[0]['boxes'])

        # 车窗排序索引
        win_boxes = boxes_by_cls[2]['boxes']
        w_nms, _ = dedup_boxes(win_boxes, [1.0]*len(win_boxes), DEDUP_IOU_THRESH_WIN)
        w_in = [wb for wb in w_nms if car_box is None or point_in_box(box_center(wb), car_box)]
        w_geom = filter_windows_by_geometry(w_in, car_box)
        if car_box is not None and len(car_box)==4 and (car_box[2]-car_box[0])>0:
            cw = car_box[2]-car_box[0]
            w_sorted = sorted(w_geom, key=lambda wb: (wb[0]+wb[2])/2/cw)
        else:
            w_sorted = sorted(w_geom, key=lambda wb: (wb[0]+wb[2])/2)
        windows = [(i, wb) for i, wb in enumerate(w_sorted)]

        # head 绑定窗 + 窗内跟踪
        head_boxes = boxes_by_cls[1]['boxes']
        # 先按窗分组当前帧 head
        heads_by_win = defaultdict(list)  # win_index -> list of (rel_x, rel_y, box)
        for hb in head_boxes:
            hc = box_center(hb)
            assigned = None
            for wid, wb in windows:
                if point_in_box(hc, wb):
                    assigned = wid; break
            if assigned is not None:
                wb = windows[assigned][1]
                ww, wh = wb[2]-wb[0], wb[3]-wb[1]
                if ww>0 and wh>0:
                    rx=(hc[0]-wb[0])/ww; ry=(hc[1]-wb[1])/wh
                    heads_by_win[assigned].append((rx, ry, hb))

        # 每个窗：窗内跟踪，分配全局唯一 ID
        head_dets = []  # (box, head_id, conf)
        win_head_counts = {}
        for wid, wb in windows:
            if wid not in win_trackers:
                win_trackers[wid] = WindowTracker(head_alloc)
            wt = win_trackers[wid]
            tracked = wt.update(heads_by_win.get(wid, []))
            for box, hid, rx, ry in tracked:
                win_seen_ids[wid].add(hid)
            win_head_counts[wid] = len(win_seen_ids[wid])
            head_dets.extend((box, hid, 0.0) for box, hid, rx, ry in tracked)

        pred_count = len(set().union(*win_seen_ids.values())) if win_seen_ids else 0  # 全局唯一 head ID 数

        # 渲染
        win_draw = [(wid, wb, win_head_counts.get(wid, 0)) for wid, wb in windows]
        out_img = render_frame(img, car_box, win_draw, head_dets, actual_count if actual_count is not None else -1, pred_count)
        cv2.imencode('.jpg', out_img)[1].tofile(os.path.join(out_sub, name))

    total_ids = len(set().union(*win_seen_ids.values())) if win_seen_ids else 0
    return total_ids, actual_count, {i: len(win_seen_ids[i]) for i in win_seen_ids}


def main():
    model = YOLO(MODEL_PATH)
    print(f"[模型] {MODEL_PATH}\n  类别: {model.names}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    existing = [d for d in os.listdir(OUT_ROOT) if d.startswith(today + "_")]
    next_seq = max([int(m.group(1)) for d in existing if (m := re.match(rf"^{today}_(\d+)$", d))], default=0) + 1
    run_dir = os.path.join(OUT_ROOT, f"{today}_{next_seq:03d}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[输出] {run_dir}")

    seqs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])[:12]

    results = []
    for seq in seqs:
        out_sub = os.path.join(run_dir, seq)
        os.makedirs(out_sub, exist_ok=True)
        print(f"\n[序列] {seq}")
        pred, actual, win = process_sequence(model, seq, out_sub)
        diff = pred - actual if actual is not None else "?"
        status = "准确" if actual is not None and diff==0 else ("偏差" if actual is not None and abs(diff)<=1 else "偏差较大")
        results.append((seq, pred, actual, diff))
        print(f"  预测={pred} 实际={actual} 差={diff} ({status}) 每窗={win}")

    print("\n" + "=" * 60)
    print("汇总（C方案：窗内跨帧跟踪）")
    print("=" * 60)
    for seq, pred, actual, diff in results:
        print(f"  {seq}: 预测={pred} 实际={actual} 差={diff}")
    print(f"\n完成，结果输出到: {run_dir}")


if __name__ == "__main__":
    main()
