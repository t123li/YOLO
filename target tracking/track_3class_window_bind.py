# -*- coding: utf-8 -*-
"""D 方案：3 类检测 + 车窗跨帧跟踪(稳定窗ID) + 运动补偿&OSNet 人跟踪 + 窗口绑定(不跳窗)。

分层（每层解决一个问题）：
1. 【车窗跨帧跟踪】车窗刚性固定、随车整体运动，是全场最易跨帧匹配的目标。
   用"车锚定坐标"(窗口中心相对车框的归一化 x) 做跨帧匹配，给车窗分配稳定 ID。
   不再像 C 方案那样每帧按 x 重新排序给索引（窗漏检时索引会错位 → 人绑定全乱）。
2. 【人 = 运动补偿 + OSNet】原样沿用 sideview_headshoulder_counter.py 的
   MotionCoherentTracker：RANSAC 估计整车平移(GMC 简化版) + 速度预测 + OSNet 外观。
   能处理人的独立运动（前倾/后仰/转头 → 相对车窗的位置漂移）。
3. 【窗口绑定】人 ID 首次出现时记录所在车窗，此后该 ID 只允许出现在绑定窗内
   （同帧、跨帧都不跳窗）。某帧绑定窗漏检时放宽约束（无法验证窗），仍靠外观+运动兜底。

计数：全局唯一 person ID 数 = 车内人数；每窗唯一 person ID 数 = 每窗人数。

用法：
    # 批量（前 12 个序列，输出到 tracking_output/日期_序号/源序列名/）
    D:/Anaconda/envs/course_torch/python.exe track_3class_window_bind.py
    # 单序列调试
    D:/Anaconda/envs/course_torch/python.exe track_3class_window_bind.py <序列名> <输出目录>
"""
import os
import re
import sys
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from datetime import datetime
from ultralytics import YOLO
from torchreid import models
from torchvision import transforms

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

# ================= 检测配置 =================
CONF_HEAD = 0.3     # head 阈值（3类模型 head_shoulder，可调：太高漏检→欠计数，太低混入头枕/反光→ID碎片）
CONF_WIN = 0.20     # window 阈值
CONF_CAR = 0.30     # car 阈值
PRED_CONF = 0.2     # 预测底层 conf（之后按类过滤）
IOU_NMS = 0.45
IMG_SIZE = 640

# 车窗几何过滤（同 C 方案，师兄参数；ASPECT 放宽到 0.7 让三角窗(w/h≈1)稳定通过）
DEDUP_IOU_THRESH_WIN = 0.4
WIN_MIN_AREA_RATIO = 0.015; WIN_MAX_AREA_RATIO = 0.20
WIN_MIN_ASPECT = 0.7; WIN_MAX_ASPECT = 4.0
WIN_MIN_REL_Y = 0.10; WIN_MAX_REL_Y = 0.35
WIN_MIN_REL_W = 0.08; WIN_MAX_REL_W = 0.50

# ================= 跟踪配置 =================
# 人跟踪（沿用 sideview 参数，但针对 3 类模型 head 框小、夜间特征弱放宽门限）
MAX_AGE = 8          # 轨迹最大丢失帧数
SPATIAL_R = 90       # 空间兜底半径（高速帧间位移大，适当放宽）
DIST_GATE = 120      # 阶段一位置门限（侧视高速下帧间位移可达 100px+，80 太紧导致 ID 碎片）
SIM_GATE = 0.55      # 阶段一外观相似度门限（3类模型 head 框小、夜间 OSNet 特征弱，0.68 太苛刻）
# 车窗跨帧跟踪：IoU 匹配（窗为刚体，随车平移，用中位数位移补偿后 IoU 稳定）
WIN_IOU_THRESH = 0.25     # 位移补偿后同窗 IoU 应 >0.8，0.25 门限对帧间位移大也够稳
WIN_TRACK_MAX_AGE = 5     # 车窗轨迹最大丢失帧数

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

def imread_cn(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


# ================= 第 1 层：车窗跨帧跟踪 =================

class WindowTrack:
    __slots__ = ('wid', 'box', 'age', 'hits')
    def __init__(self, wid, box):
        self.wid = wid
        self.box = box
        self.age = 0
        self.hits = 1


class WindowTracker:
    """车窗跨帧跟踪：IoU + 中位数位移补偿，给车窗分配稳定 ID。

    车窗是刚体、只随车辆整体平移。帧间先用所有"旧窗→本帧窗"点对的中位数位移
    补偿旧窗位置，再算 IoU 匹配（贪心，阈值 0.25）。这比"车锚定 relx"稳定得多：
    relx 依赖车框检测（快速移动时噪声大 → relx 漂移 → 窗 ID 重建），
    而窗自身的检测位移中位数直接反映车辆平移，不受车框框定位误差影响。
    窗漏检时轨迹保留 WIN_TRACK_MAX_AGE 帧。
    """
    def __init__(self, iou_thresh=WIN_IOU_THRESH, max_age=WIN_TRACK_MAX_AGE):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}       # wid -> WindowTrack
        self.next_wid = 0

    def update(self, win_boxes):
        """win_boxes: 本帧候选窗框(已 NMS + 几何过滤 + 在车内)。返回 [(wid, box)] 按 x 排序。"""
        if not win_boxes:
            for wid in [wid for wid, t in self.tracks.items() if (t.age + 1) > self.max_age]:
                del self.tracks[wid]
            for t in self.tracks.values():
                t.age += 1
            return []

        boxes = sorted(win_boxes, key=lambda b: (b[0]+b[2])/2.0)   # 仅定显示顺序

        if not self.tracks:
            result = []
            for b in boxes:
                wid = self.next_wid; self.next_wid += 1
                self.tracks[wid] = WindowTrack(wid, b)
                result.append((wid, b))
            return result

        # 中位数位移补偿：所有 (旧窗中心, 本帧窗中心) 点对的位移中位数
        old_c = [np.array(box_center(t.box)) for t in self.tracks.values()]
        new_c = [np.array(box_center(b)) for b in boxes]
        shifts = [dc - oc for oc in old_c for dc in new_c]
        shifts_arr = np.array(shifts)
        if len(shifts_arr):
            shift = np.median(shifts_arr, axis=0)
        else:
            shift = np.zeros(2)

        used = set()
        matched = {}
        for i, b in enumerate(boxes):
            best = None; best_iou = self.iou_thresh
            for wid, t in self.tracks.items():
                if wid in used: continue
                pred = [t.box[0]+shift[0], t.box[1]+shift[1], t.box[2]+shift[0], t.box[3]+shift[1]]
                iou = compute_iou(b, pred)
                if iou > best_iou:
                    best_iou = iou; best = wid
            if best is not None:
                used.add(best); matched[i] = best

        new_tracks = {}
        result = []
        for i, b in enumerate(boxes):
            if i in matched:
                wid = matched[i]
                t = self.tracks[wid]
                t.box = b; t.age = 0; t.hits += 1
            else:
                wid = self.next_wid; self.next_wid += 1
                t = WindowTrack(wid, b)
            new_tracks[wid] = t
            result.append((wid, b))
        for wid, t in self.tracks.items():
            if wid not in new_tracks:
                t.age += 1
                if t.age <= self.max_age:
                    new_tracks[wid] = t
        self.tracks = new_tracks
        result.sort(key=lambda wb: wb[1][0])
        return result


# ================= 第 2 层：运动补偿 + OSNet 人跟踪（+ 第 3 层窗口绑定） =================

class HeadShoulderTrack:
    def __init__(self, global_id, bbox, feat, win):
        self.global_id = global_id
        self.bbox = bbox
        self.feat_history = deque(maxlen=5)
        self.feat_history.append(feat)
        self.age = 0
        self.velocity = np.array([0.0, 0.0])       # 该目标独立运动向量 [dx, dy]
        self.missed_shift = np.array([0.0, 0.0])   # 漏检期间累计整车平移，用于修正预测位置
        self.bound_window = win                    # 绑定车窗 ID（None=尚未在窗内见过）
        self.rel_pos = None                        # 最近一次"窗内归一化坐标 (rel_x, rel_y)"
        self.bound_bbox = None                     # 绑定时所在窗的框，用于计算窗内 rel


class MotionCoherentTracker:
    """sideview 的协同运动追踪器 + 窗口绑定硬约束。

    - 阶段一：运动补偿位置 + OSNet 外观 双指标匹配（代价矩阵自适应加权）。
    - 阶段二：全局历史回溯认领 + 空间兜底。
    - 新增：匹配时若"本帧检测所在窗 == 轨迹绑定窗"不成立则拒绝（窗口绑定），
      彻底杜绝 ID 跨窗跳变。
    """
    def __init__(self):
        self.tracks = []
        self.next_global_id = 0
        self.global_id_history = {}   # { gid: deque([feat,...], maxlen=5) }
        self.gid_window = {}          # { gid: 绑定窗ID 或 None }
        self.global_vehicle_velocity = np.array([0.0, 0.0])
        self.global_shift = np.array([0.0, 0.0])     # RANSAC 估计的整车帧间平移
        self.shift_inliers = 0

    def _win_ok(self, det_win, track_win):
        """窗口绑定判定：双方都有窗时要求相等；任一方无窗(漏检/未绑定)则放行。"""
        if det_win is not None and track_win is not None:
            return det_win == track_win
        return True

    def _estimate_global_shift(self, dets, tol=30.0):
        """RANSAC 估计整车帧间平移（GMC 简化版，检测点代替光流关键点）。"""
        track_c = [np.array(get_center(t.bbox)) for t in self.tracks]
        det_c = [np.array(get_center(d)) for d in dets]
        if not track_c or not det_c:
            return np.array([0.0, 0.0]), 0
        best_shift, best_n = np.array([0.0, 0.0]), 0
        for tc in track_c:
            for dc in det_c:
                s = dc - tc
                n = 0
                for tc2 in track_c:
                    for dc2 in det_c:
                        if np.linalg.norm(dc2 - tc2 - s) < tol:
                            n += 1
                if n > best_n:
                    best_n, best_shift = n, s
        return best_shift, best_n

    def _rel_in_win(self, det, win_box):
        """检测中心在窗内的归一化坐标 (rel_x, rel_y)。win_box=None 时返回 None。"""
        if win_box is None:
            return None
        ww = win_box[2] - win_box[0]
        wh = win_box[3] - win_box[1]
        if ww <= 0 or wh <= 0:
            return None
        hc = get_center(det)
        return ((hc[0] - win_box[0]) / ww, (hc[1] - win_box[1]) / wh)

    def update(self, dets, feats, confs, wins, win_boxes):
        """wins[i]: 第 i 个检测所在的车窗 ID（或 None）；win_boxes: {wid: 当前帧窗框}。
        返回 (gids, wins_out)。"""
        for t in self.tracks:
            t.age += 1

        assigned_indices = [-1] * len(dets)
        matched_tracks = set()

        if len(dets) == 0:
            return [], []

        # 每个检测的窗内 rel 坐标（窗口丢失时为 None，回退全局位置）
        rels = [self._rel_in_win(det, win_boxes.get(w) if w is not None else None) for det, w in zip(dets, wins)]

        # ---- GMC 简化版：RANSAC 整车平移（全局通道用） ----
        self.global_shift, self.shift_inliers = self._estimate_global_shift(dets)
        use_shift = self.shift_inliers >= 2

        predicted_centers = {}
        for j, t in enumerate(self.tracks):
            cx, cy = get_center(t.bbox)
            if use_shift:
                predicted_centers[j] = np.array([cx, cy]) + self.global_shift + t.missed_shift
            else:
                v = t.velocity if np.linalg.norm(t.velocity) > 0 else self.global_vehicle_velocity
                predicted_centers[j] = np.array([cx, cy]) + v * t.age

        dist_gate = DIST_GATE if use_shift else 150.0

        # 阶段一辅助：计算 (i,j) 的距离。
        # 同窗 → 用窗内 rel 距离（乘窗宽转像素当量，自动抵消车速/透视）
        # 异窗/无窗 → 用全局像素距离
        win_size_cache = {}
        def pair_dist(i, j):
            t = self.tracks[j]
            if wins[i] is not None and t.bound_window == wins[i] and t.rel_pos is not None:
                wb = win_boxes.get(wins[i])
                if wb is not None:
                    ww = wb[2] - wb[0]
                    rx, ry = rels[i]
                    trx, try_ = t.rel_pos
                    return np.hypot((rx - trx) * ww, (ry - try_) * (wb[3] - wb[1])), True
            return np.linalg.norm(np.array(get_center(dets[i])) - predicted_centers[j]), False

        # ---- 阶段一：位置 + 外观 双指标配对（含窗口绑定硬约束） ----
        matches = []
        for i, det in enumerate(dets):
            d_cx, d_cy = get_center(det)
            c = confs[i]
            s_c = (c * c) / (c * c + (1.0 - c) * (1.0 - c))
            w_m = 0.2 + 0.3 * s_c
            for j, t in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                if not self._win_ok(wins[i], t.bound_window):
                    continue
                dist, is_rel = pair_dist(i, j)
                gate = dist_gate if not is_rel else max(0.35 * (win_boxes.get(wins[i])[2] - win_boxes.get(wins[i])[0]) if wins[i] in win_boxes else 120, 90)
                sims = np.dot(np.array(t.feat_history), feats[i])
                sim = float(np.max(sims))
                if dist < gate and sim > SIM_GATE:
                    motion_score = 1.0 - dist / gate
                    score = w_m * motion_score + (1.0 - w_m) * sim
                    matches.append((score, i, j, dist, sim, is_rel))

        matches.sort(key=lambda x: x[0], reverse=True)

        velocity_samples = []

        for score, i, j, dist, sim, is_rel in matches:
            if assigned_indices[i] == -1 and j not in matched_tracks:
                assigned_indices[i] = j
                matched_tracks.add(j)

                t = self.tracks[j]
                old_cx, old_cy = get_center(t.bbox)
                new_cx, new_cy = get_center(dets[i])
                inst_velocity = np.array([new_cx - old_cx, new_cy - old_cy])

                w = float(np.clip(confs[i], 0.2, 0.9))
                t.velocity = t.velocity * (1.0 - w) + inst_velocity * w
                velocity_samples.append(t.velocity)

                t.bbox = dets[i]
                t.feat_history.append(feats[i])
                t.age = 0
                t.missed_shift = np.array([0.0, 0.0])
                if wins[i] is not None:
                    t.bound_window = wins[i]
                    t.rel_pos = rels[i]
                    t.bound_bbox = win_boxes.get(wins[i])
                    self.gid_window[t.global_id] = wins[i]
                elif t.rel_pos is None:
                    pass  # 仍无窗，保持未绑定

        # 更新全局车速
        if len(velocity_samples) > 0:
            self.global_vehicle_velocity = np.mean(velocity_samples, axis=0)

        # ---- 阶段二：新出现 / 刚恢复的人，全局回溯认领 + 空间兜底（都受窗口绑定约束） ----
        claimed_gids = {self.tracks[j].global_id for j in matched_tracks}
        pre_track_indices = list(range(len(self.tracks)))

        for i in range(len(dets)):
            if assigned_indices[i] != -1:
                continue

            det_feat = feats[i]
            matched_gid = -1
            best_sim = 0

            claim_thr = 0.76 + (1.0 - confs[i]) * 0.10
            for gid, feat_list in self.global_id_history.items():
                if gid in claimed_gids:
                    continue
                if not self._win_ok(wins[i], self.gid_window.get(gid)):
                    continue
                sims = np.dot(np.array(feat_list), det_feat)
                max_s = float(np.max(sims))
                if max_s > best_sim:
                    best_sim = max_s
                    if max_s > claim_thr:
                        matched_gid = gid

            spatial_j = -1
            if matched_gid == -1:
                d_cx, d_cy = get_center(dets[i])
                best_d = SPATIAL_R * (0.5 + 0.5 * confs[i])
                for j in pre_track_indices:
                    if j in matched_tracks:
                        continue
                    if self.tracks[j].global_id in claimed_gids:
                        continue
                    if not self._win_ok(wins[i], self.tracks[j].bound_window):
                        continue
                    dist, is_rel = pair_dist(i, j)
                    if dist < best_d:
                        best_d = dist
                        spatial_j = j
                if spatial_j != -1:
                    matched_gid = self.tracks[spatial_j].global_id
                    matched_tracks.add(spatial_j)
                    best_sim = 0.0

            if matched_gid == -1:
                matched_gid = self.next_global_id
                self.next_global_id += 1
                self.global_id_history[matched_gid] = deque([det_feat], maxlen=5)
                self.gid_window[matched_gid] = wins[i]   # 首次出现即绑定车窗
                if wins[i] is not None:
                    pass  # bound_window 在新建 track 时设置
            else:
                self.global_id_history[matched_gid].append(det_feat)
            claimed_gids.add(matched_gid)

            existing_idx = None
            for idx, t in enumerate(self.tracks):
                if t.global_id == matched_gid:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                t = self.tracks[existing_idx]
                t.bbox = dets[i]
                t.feat_history.append(det_feat)
                t.age = 0
                t.missed_shift = np.array([0.0, 0.0])
                if wins[i] is not None:
                    t.bound_window = wins[i]
                    t.rel_pos = rels[i]
                    t.bound_bbox = win_boxes.get(wins[i])
                    self.gid_window[t.global_id] = wins[i]
                assigned_indices[i] = existing_idx
            else:
                new_track = HeadShoulderTrack(matched_gid, dets[i], det_feat, wins[i])
                if wins[i] is not None:
                    new_track.rel_pos = rels[i]
                    new_track.bound_bbox = win_boxes.get(wins[i])
                self.tracks.append(new_track)
                assigned_indices[i] = len(self.tracks) - 1

        # 未匹配旧轨迹：累计整车平移，供下次预测修正
        acc_shift = self.global_shift if use_shift else self.global_vehicle_velocity
        for j in pre_track_indices:
            if j not in matched_tracks:
                self.tracks[j].missed_shift += acc_shift

        frame_gids = []
        frame_wins = []
        for idx in assigned_indices:
            t = self.tracks[idx]
            frame_gids.append(t.global_id)
            frame_wins.append(t.bound_window)

        self.tracks = [t for t in self.tracks if t.age <= MAX_AGE]
        return frame_gids, frame_wins

    def total_count(self):
        return len(self.global_id_history)


# ================= 渲染 =================

def _rects_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def render_frame(img, car_box, windows, head_dets, actual_count, pred_count):
    """渲染：车绿框、车窗黄框(含人数)、head 红框(只显示 ID)。细框细字防遮挡。"""
    out = img.copy()
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.4; THICK = 1

    # head 框（红），标签只显示全局 ID：h{id}（不显示置信度）
    if head_dets:
        placed = []
        dets = []
        labels = []
        for hb, gid in head_dets:
            dets.append((int(hb[0]), int(hb[1]), int(hb[2]), int(hb[3])))
            labels.append(f"h{gid}")
        for (x1, y1, x2, y2) in dets:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), THICK)
        for idx, ((x1, y1, x2, y2), label) in enumerate(zip(dets, labels)):
            (tw, th), _ = cv2.getTextSize(label, FONT, SCALE, THICK)
            candidates = [(x1, y1-th-3), (x1, y1-th-11), (x1, y2+3), (x1, y2+11),
                          (x1+2, y1+th+2), (x1-tw-3, y1+th), (x2+3, y1+th)]
            chosen = candidates[0]
            for (lx, ly) in candidates:
                lrect = (lx, ly, lx+tw, ly+th)
                if any(_rects_overlap(lrect, dets[j]) for j in range(len(dets)) if j != idx):
                    continue
                if any(_rects_overlap(lrect, p) for p in placed):
                    continue
                chosen = (lx, ly); break
            lx, ly = chosen
            ov = out[ly:ly+th, lx:lx+tw]
            if ov.size > 0:
                out[ly:ly+th, lx:lx+tw] = cv2.addWeighted(ov, 0.4, np.zeros_like(ov), 0.6, 0)
            cv2.putText(out, label, (lx, ly+th), FONT, SCALE, (0, 0, 255), THICK)
            placed.append((lx, ly, lx+tw, ly+th))

    # 车窗框（黄），标签 win{wid}:{count}
    for wid, wb, cnt in windows:
        x1, y1, x2, y2 = [int(v) for v in wb]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), THICK)
        label = f"win{wid}:{cnt}"
        (tw, th), _ = cv2.getTextSize(label, FONT, SCALE, THICK)
        cv2.putText(out, label, (x1, max(y1-th, 12)), FONT, SCALE, (0, 255, 255), THICK)

    # 车框（绿）
    if car_box is not None:
        x1, y1, x2, y2 = [int(v) for v in car_box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), THICK)
        cv2.putText(out, "car", (x1, max(y1-20, 12)), FONT, SCALE, (0, 255, 0), THICK)

    cv2.putText(out, f"Pred: {pred_count}  Actual: {actual_count}  Diff: {pred_count-actual_count:+d}",
                (8, 20), FONT, 0.55, (0, 0, 255), 1)
    return out


# ================= 简单车跟踪 =================

class SimpleCarTracker:
    """极简车跟踪：选面积最大的 car 框。"""
    def update(self, boxes):
        if not boxes: return None
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        return boxes[max(range(len(areas)), key=lambda i: areas[i])]


# ================= 特征提取 =================

def extract_feat(img_crop, reid_model, transform, device):
    img = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(img)
    feat = feat.cpu().numpy().flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-6)
    return feat


def get_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


# ================= 主流程 =================

def process_sequence(model, reid_model, transform, device, seq, out_sub):
    in_dir = os.path.join(DATA_ROOT, seq)
    frames = sorted([f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    actual_count = read_actual_count(in_dir)

    car_tracker = SimpleCarTracker()
    window_tracker = WindowTracker()
    person_tracker = MotionCoherentTracker()
    window_seen = defaultdict(set)   # wid -> set(gid)
    last_car_box = None

    for fi, name in enumerate(frames):
        img = imread_cn(os.path.join(in_dir, name))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        r = model.predict(img, conf=PRED_CONF, iou=IOU_NMS, imgsz=IMG_SIZE, verbose=False)[0]

        boxes_by_cls = {c: {'boxes': [], 'scores': []} for c in CLASS_NAMES}
        for box in r.boxes:
            cid = int(box.cls[0].cpu().numpy())
            if cid not in CLASS_NAMES: continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu().numpy())
            thr = CONF_CAR if cid == 0 else (CONF_HEAD if cid == 1 else CONF_WIN)
            if score < thr: continue
            boxes_by_cls[cid]['boxes'].append([x1, y1, x2, y2])
            boxes_by_cls[cid]['scores'].append(score)

        # 车（面积最大；本帧漏检时沿用上一帧，保持窗锚定稳定）
        car_box = car_tracker.update(boxes_by_cls[0]['boxes'])
        if car_box is None:
            car_box = last_car_box
        else:
            last_car_box = car_box

        # 车窗：NMS + 几何过滤 + 在车内 → 跨帧跟踪（稳定窗 ID）
        win_boxes = boxes_by_cls[2]['boxes']
        w_nms, w_scores = dedup_boxes(win_boxes, boxes_by_cls[2]['scores'], DEDUP_IOU_THRESH_WIN)
        w_in = [wb for wb in w_nms if car_box is None or point_in_box(box_center(wb), car_box)]
        w_geom = filter_windows_by_geometry(w_in, car_box)
        windows = window_tracker.update(w_geom)   # [(wid, box)] 按 x 排序，窗 ID 跨帧稳定
        win_boxes = {wid: wb for wid, wb in windows}   # wid -> 窗框，供人 tracker 算窗内 rel

        # head：提特征 + 绑定窗（中心点落在哪个窗框内）
        head_boxes = boxes_by_cls[1]['boxes']
        head_scores = boxes_by_cls[1]['scores']
        dets, feats, confs, wins = [], [], [], []
        for i, hb in enumerate(head_boxes):
            x1, y1, x2, y2 = [int(v) for v in hb]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            feat = extract_feat(crop, reid_model, transform, device)
            dets.append((x1, y1, x2, y2))
            feats.append(feat)
            confs.append(head_scores[i])
            hc = box_center(hb)
            wins.append(next((wid for wid, wb in windows if point_in_box(hc, wb)), None))

        gids, frame_wins = person_tracker.update(dets, feats, confs, wins, win_boxes)
        for gid, w in zip(gids, frame_wins):
            if w is not None:
                window_seen[w].add(gid)
        pred_count = person_tracker.total_count()

        head_dets = [(dets[i], gids[i]) for i in range(len(dets))]
        win_draw = [(wid, wb, len(window_seen[wid])) for wid, wb in windows]
        out_img = render_frame(img, car_box, win_draw, head_dets,
                               actual_count if actual_count is not None else -1, pred_count)
        cv2.imencode('.jpg', out_img)[1].tofile(os.path.join(out_sub, name))

    total = person_tracker.total_count()
    return total, actual_count, {wid: len(win_seen) for wid, win_seen in window_seen.items()}


def main():
    model = YOLO(MODEL_PATH)
    print(f"[模型] {MODEL_PATH}\n  类别: {model.names}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {device}")
    reid_model = models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True).to(device).eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 单序列调试模式：python xx.py <序列名> <输出目录>
    if len(sys.argv) >= 3 and sys.argv[1].strip():
        seq = sys.argv[1].strip()
        out_sub = sys.argv[2].strip()
        os.makedirs(out_sub, exist_ok=True)
        print(f"[单序列] {seq} -> {out_sub}")
        pred, actual, win = process_sequence(model, reid_model, transform, device, seq, out_sub)
        diff = pred - actual if actual is not None else "?"
        print(f"[结果] 预测={pred} 实际={actual} 差={diff} 每窗={win}")
        return

    # 批量模式：前 12 个序列 → tracking_output/日期_序号/源序列名/
    os.makedirs(OUT_ROOT, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    existing = [d for d in os.listdir(OUT_ROOT) if d.startswith(today + "_")]
    next_seq = max([int(m.group(1)) for d in existing if (m := re.match(rf"^{today}_(\d+)$", d))], default=0) + 1
    run_dir = os.path.join(OUT_ROOT, f"{today}_{next_seq:03d}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[输出] {run_dir}")

    # 归档代码 + 日志
    log_path = os.path.join(run_dir, "run_window_bind.log")
    log_f = open(log_path, "w", encoding="utf-8")
    try:
        with open(os.path.abspath(__file__), encoding="utf-8") as f:
            code_src = f.read()
        with open(os.path.join(run_dir, "track_3class_window_bind.py"), "w", encoding="utf-8") as f:
            f.write(code_src)
    except Exception as e:
        print(f"[警告] 归档代码失败: {e}")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    seqs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])[:12]

    results = []
    for seq in seqs:
        out_sub = os.path.join(run_dir, seq)
        os.makedirs(out_sub, exist_ok=True)
        log(f"\n[序列] {seq}")
        pred, actual, win = process_sequence(model, reid_model, transform, device, seq, out_sub)
        diff = pred - actual if actual is not None else "?"
        status = "准确" if actual is not None and diff == 0 else ("偏差" if actual is not None and abs(diff) <= 1 else "偏差较大")
        results.append((seq, pred, actual, diff))
        log(f"  预测={pred} 实际={actual} 差={diff} ({status}) 每窗={win}")

    log("\n" + "=" * 60)
    log("汇总（D方案：车窗跨帧跟踪 + 运动补偿&OSNet + 窗口绑定）")
    log("=" * 60)
    for seq, pred, actual, diff in results:
        log(f"  {seq}: 预测={pred} 实际={actual} 差={diff}")
    log(f"\n完成，结果输出到: {run_dir}")
    log_f.close()


if __name__ == "__main__":
    main()
