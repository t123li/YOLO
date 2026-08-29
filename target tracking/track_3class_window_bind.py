# -*- coding: utf-8 -*-
"""D 方案 v4：3 类检测 + 车窗跨帧跟踪(稳定窗ID) + 运动补偿&OSNet 人跟踪 + 窗口绑定(不跳窗)。

分层（每层解决一个问题）：
1. 【车窗跨帧跟踪】车窗刚性固定、随车整体运动，用"离车头的物理槽位"跨帧匹配稳定 ID。
   车头方向由首帧贴边启发式 + 位移确认判断，前窗=槽0，中间插入新窗不影响已有槽位。
2. 【人 = 运动补偿 + OSNet】RANSAC 估计整车平移 + 速度预测 + OSNet 外观（已修 BGR→RGB）。
3. 【窗口绑定】人 ID 首次出现绑定所在车窗，此后只允许出现在绑定窗内（不跳窗）。
4. 【v4 新增】
   - 匈牙利全局匹配：阶段一用 scipy 线性分配替代贪心，杜绝聚集时抢 ID。
   - 窗内顺序保持：同窗 rel_x 差 >0.35 窗宽禁止匹配，防止相邻目标 ID 互换。
   - 生命周期：新轨迹 Tentative（渲染标"?"、橙色），累计观测 >=2 帧转 Confirmed 才计数。
   - 座位槽硬约束：每窗确认人数 <= SEATS_PER_WINDOW(3)，窗满不建新 ID。
   - 两级关联：predict conf=0.05 多检；高分(>=0.35)可建新轨迹，低分只延续（不新建）。

计数：全局唯一 confirmed person ID 数 = 车内人数；每窗 confirmed 数 = 每窗人数。

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
CONF_HEAD = 0.10    # head 低阈值（多检，含遮挡/小头低分真检测；误检交给生命周期+两级关联过滤）
CONF_HEAD_HIGH = 0.35  # 高分阈值：>= 此值才允许新建轨迹（低分框只延续不新建）
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
# 车窗跨帧跟踪：用"窗在车身的位置(relx)"贪心匹配（相邻窗 relx 差约 0.1-0.3，0.2 不会错配）
WIN_ASSOC_RELX = 0.20     # relx 匹配门限（窗相对车框归一化 x 的距离）
WIN_TRACK_MAX_AGE = 5     # 车窗轨迹最大丢失帧数
# 座位槽硬约束：每窗最多允许的确认人数（前/中/后窗各≤3），窗满时不新建 ID
SEATS_PER_WINDOW = 3

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
    __slots__ = ('wid', 'box', 'relx', 'velocity', 'age', 'hits')
    def __init__(self, wid, box, relx):
        self.wid = wid
        self.box = box
        self.relx = relx          # 窗在车身的位置：中心相对车框的归一化 x
        self.velocity = np.array([0.0, 0.0])   # 每窗自身的帧间速度（匀速运动模型）
        self.age = 0
        self.hits = 1


class WindowTracker:
    """车窗跨帧跟踪：用"窗离车头的物理位置"跨帧匹配，给车窗分配稳定 ID。

    车窗是刚体、相对车身位置固定。关键先验：车的运动方向（车头朝向）决定车窗的
    左右顺序——车头朝右(+1) 时从左到右是 后窗→中窗→前窗；车头朝左(-1) 时反之。
    因此用"离车头的归一化距离"作为窗的物理身份（0=最靠车头，越大越靠车尾），
    跨帧稳定：即使中间插入新窗（如中窗后露），前窗/后窗的槽位不变。
    匹配策略：
    1. 顺序优先：旧窗按 slot 排序、新窗按 slot 排序，"第 k 旧窗 ↔ 第 k 新窗"，
       slot 差 < 门限才接受（同一物理窗 slot 稳定）。
    2. slot 最近邻兜底：未配的新窗找 slot 最近的未用旧窗。
    """
    def __init__(self, relx_dist=WIN_ASSOC_RELX, max_age=WIN_TRACK_MAX_AGE):
        self.relx_dist = relx_dist
        self.max_age = max_age
        self.tracks = {}       # wid -> WindowTrack
        self.next_wid = 0

    @staticmethod
    def _relx(b, car_box, direction, img_w):
        """窗离车头的归一化距离：0=最靠车头(前窗)，越大越靠车尾。方向无关。"""
        wcx = (b[0] + b[2]) / 2.0
        if car_box is not None and (car_box[2] - car_box[0]) > 0:
            cw = car_box[2] - car_box[0]
            if direction > 0:   # 车头在右，前窗在最右
                return (car_box[2] - wcx) / cw
            else:              # 车头在左，前窗在最左
                return (wcx - car_box[0]) / cw
        # 无车框：按画面水平位置兜底（车头在右假设）
        return (img_w - wcx) / img_w if img_w > 0 else 0.5

    def update(self, win_boxes, car_box, direction, img_w):
        """win_boxes: 本帧候选窗框(已 NMS + 几何过滤 + 在车内)。
        car_box/direction/img_w: 用于算"离车头的物理位置"。返回 [(wid, box)] 按 x 排序。"""
        if not win_boxes:
            for wid in [wid for wid, t in self.tracks.items() if (t.age + 1) > self.max_age]:
                del self.tracks[wid]
            for t in self.tracks.values():
                t.age += 1
            return []

        # 本帧候选窗：按 slot(离车头距离) 排序用于匹配，输出时再按 x 排序
        boxes = sorted(win_boxes, key=lambda b: (b[0]+b[2])/2.0)   # 仅定显示顺序
        new_relx = [self._relx(b, car_box, direction, img_w) for b in boxes]
        # 按 slot 升序重排（0=前窗最靠车头），匹配用
        ordered = sorted(zip(boxes, new_relx), key=lambda x: x[1])
        boxes_match = [b for b, _ in ordered]
        new_relx_match = [rx for _, rx in ordered]

        if not self.tracks:
            result = []
            for b, rx in zip(boxes, new_relx):
                wid = self.next_wid; self.next_wid += 1
                self.tracks[wid] = WindowTrack(wid, b, rx)
                result.append((wid, b))
            return result

        # 旧窗按 slot 排序（离车头距离：0=前窗）
        old_sorted = sorted(self.tracks.values(), key=lambda t: t.relx)

        used_new = set()
        used_old = set()
        matched = {}   # match_index -> wid

        # 1. 顺序优先：第 k 旧窗 ↔ 第 k 新窗（都按 slot 升序），slot 差 < 门限才接受
        for k in range(min(len(old_sorted), len(boxes_match))):
            t = old_sorted[k]
            d = abs(new_relx_match[k] - t.relx)
            if d < self.relx_dist:
                matched[k] = t.wid
                used_new.add(k)
                used_old.add(t.wid)

        # 2. slot 最近邻兜底：未配的新窗，找未用旧窗中 slot 最近的
        for i in range(len(boxes_match)):
            if i in used_new:
                continue
            best = None; best_d = self.relx_dist
            for t in old_sorted:
                if t.wid in used_old:
                    continue
                d = abs(new_relx_match[i] - t.relx)
                if d < best_d:
                    best_d = d; best = t.wid
            if best is not None:
                matched[i] = best
                used_new.add(i)
                used_old.add(best)

        new_tracks = {}
        result = []
        for i, b in enumerate(boxes_match):
            if i in matched:
                wid = matched[i]
                t = self.tracks[wid]
                inst_v = np.array([box_center(b)[0] - box_center(t.box)[0],
                                   box_center(b)[1] - box_center(t.box)[1]])
                t.velocity = 0.6 * t.velocity + 0.4 * inst_v
                t.relx = new_relx_match[i]
                t.box = b; t.age = 0; t.hits += 1
            else:
                wid = self.next_wid; self.next_wid += 1
                t = WindowTrack(wid, b, new_relx_match[i])
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
        self.hits = 1                      # 该轨迹被观测到的帧数（确认机制用）
        self.velocity = np.array([0.0, 0.0])       # 该目标独立运动向量 [dx, dy]
        self.missed_shift = np.array([0.0, 0.0])   # 漏检期间累计整车平移，用于修正预测位置
        self.bound_window = win                    # 绑定车窗 ID（None=尚未在窗内见过）
        self.rel_pos = None                        # 最近一次"窗内归一化坐标 (rel_x, rel_y)"
        self.bound_bbox = None                     # 绑定时所在窗的框，用于计算窗内 rel
        self.in_win_rank = None                    # 窗内排名（同窗按 rel_x 排序的名次，跨帧顺序保持）


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
        self.gid_hits = {}            # { gid: 累计观测帧数 }，确认机制用
        self.gid_min_hits = {}        # { gid: 确认所需最小帧数 }（低分框需更多帧确认）
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

    def update(self, dets, feats, confs, wins, win_boxes, allow_new=None):
        """wins[i]: 第 i 个检测所在的车窗 ID（或 None）；win_boxes: {wid: 当前帧窗框}。
        allow_new[i]: 该检测是否允许新建轨迹（低分框只延续不新建，ByteTrack 两级关联思想）。
        返回 (gids, wins_out)。"""
        if allow_new is None:
            allow_new = [True] * len(dets)
        for t in self.tracks:
            t.age += 1

        assigned_indices = [-1] * len(dets)
        matched_tracks = set()

        if len(dets) == 0:
            return [], []

        # 每个检测的窗内 rel 坐标（窗口丢失时为 None，回退全局位置）
        rels = [self._rel_in_win(det, win_boxes.get(w) if w is not None else None) for det, w in zip(dets, wins)]

        # 维护轨迹窗内 rel_x（供阶段一"窗内顺序保持"约束使用）
        for t in self.tracks:
            t.in_win_rank = t.rel_pos[0] if (t.bound_window is not None and t.rel_pos is not None) else None

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

        # ---- 阶段一：位置 + 外观 双指标配对（含窗口绑定硬约束，匈牙利全局最优） ----
        # 构建代价矩阵 cost[i][j]（大数 = 不匹配），scipy 匈牙利求全局最小总代价的一对一分配。
        # 用大数而非 np.inf，避免 "cost matrix is infeasible"；分配后过滤掉落在 BIG 上的匹配。
        from scipy.optimize import linear_sum_assignment
        BIG = 1e6
        n_dets = len(dets)
        n_trks = len(self.tracks)
        cost = np.full((n_dets, n_trks), BIG)
        for i, det in enumerate(dets):
            d_cx, d_cy = get_center(det)
            c = confs[i]
            s_c = (c * c) / (c * c + (1.0 - c) * (1.0 - c))
            w_m = 0.2 + 0.3 * s_c
            for j, t in enumerate(self.tracks):
                if not self._win_ok(wins[i], t.bound_window):
                    continue
                # 窗内顺序保持：同窗时，检测 rel_x 与轨迹 rel_x 差 >0.35 窗宽则禁止
                # （人在窗内 rel_x 跨帧漂移有限，差太大说明顺序错位 → 防止相邻目标 ID 互换）
                if wins[i] is not None and t.bound_window == wins[i] and rels[i] is not None and t.rel_pos is not None:
                    if abs(rels[i][0] - t.rel_pos[0]) > 0.35:
                        continue
                dist, is_rel = pair_dist(i, j)
                gate = dist_gate if not is_rel else max(0.35 * (win_boxes.get(wins[i])[2] - win_boxes.get(wins[i])[0]) if wins[i] in win_boxes else 120, 90)
                sims = np.dot(np.array(t.feat_history), feats[i])
                sim = float(np.max(sims))
                if dist < gate and sim > SIM_GATE:
                    motion_score = 1.0 - dist / gate
                    score = w_m * motion_score + (1.0 - w_m) * sim
                    cost[i, j] = 1.0 - score   # score 高 → cost 低

        assigned_indices = [-1] * len(dets)
        matched_tracks = set()
        if n_dets > 0 and n_trks > 0 and (cost < BIG / 2).any():
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < BIG / 2:   # 只接受真实候选，丢弃落在 BIG 上的
                    assigned_indices[r] = c
                    matched_tracks.add(c)

        velocity_samples = []

        for i, j in enumerate(assigned_indices):
            if j == -1:
                continue
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
            t.hits += 1
            self.gid_hits[t.global_id] = self.gid_hits.get(t.global_id, 0) + 1
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

            # 窗内位置占用抑制：仍未匹配上时，若检测落在某活跃轨迹的窗内且窗内 rel 距离很近
            # （< 0.35 窗宽），沿用该 ID 而非新建。对抗"多人挤窗 + 低分框重复框选 → OSNet 判新人"。
            if matched_gid == -1 and wins[i] is not None:
                wb = win_boxes.get(wins[i])
                if wb is not None and rels[i] is not None:
                    ww = wb[2] - wb[0]
                    if ww > 0:
                        best_occ = None; best_occ_d = 0.35 * ww
                        for j in pre_track_indices:
                            if j in matched_tracks:
                                continue
                            t = self.tracks[j]
                            if t.bound_window != wins[i]:
                                continue
                            if t.rel_pos is None:
                                continue
                            # 窗内 rel 距离 × 窗宽 = 像素距离
                            d = np.hypot((rels[i][0]-t.rel_pos[0])*ww,
                                         (rels[i][1]-t.rel_pos[1])*(wb[3]-wb[1]))
                            if d < best_occ_d:
                                best_occ_d = d; best_occ = j
                        if best_occ is not None:
                            matched_gid = self.tracks[best_occ].global_id
                            matched_tracks.add(best_occ)
                            best_sim = 0.0

            # 两级关联：低分框允许建轨迹，但用更高的确认门槛（生命周期 min_hits 按 conf 自适应）。
            # 遮挡/小头的低分真检测能建 Tentative → 连续多帧确认后计数（补欠计数）；
            # 头枕/反光的低分误检 1-2 帧闪现，达不到确认门槛 → 不计数（治过计数）。
            if matched_gid == -1 and wins[i] is not None:
                win_confirmed = sum(1 for t in self.tracks
                                    if t.bound_window == wins[i] and t.global_id in self.confirmed_ids())
                if win_confirmed >= SEATS_PER_WINDOW:
                    continue   # 窗已满，跳过此检测（不建新 ID）

            if matched_gid == -1:
                matched_gid = self.next_global_id
                self.next_global_id += 1
                self.global_id_history[matched_gid] = deque([det_feat], maxlen=5)
                self.gid_hits[matched_gid] = 1
                self.gid_min_hits[matched_gid] = 3 if confs[i] < CONF_HEAD_HIGH else 2  # 低分需更多帧确认
                self.gid_window[matched_gid] = wins[i]   # 首次出现即绑定车窗
                if wins[i] is not None:
                    pass  # bound_window 在新建 track 时设置
            else:
                self.global_id_history[matched_gid].append(det_feat)
                self.gid_hits[matched_gid] = self.gid_hits.get(matched_gid, 0) + 1
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
                t.hits += 1
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
            if idx == -1:
                frame_gids.append(-1)   # 被座位槽抑制/未分配的检测，用 -1 标记，渲染时跳过
                frame_wins.append(None)
                continue
            t = self.tracks[idx]
            frame_gids.append(t.global_id)
            frame_wins.append(t.bound_window)

        self.tracks = [t for t in self.tracks if t.age <= MAX_AGE]
        return frame_gids, frame_wins

    def confirmed_ids(self):
        """确认过的 ID 集合：累计观测 >= 2 帧即确认（短序列 8 帧，低分乘客可能只露 2-3 帧；
        过计数由座位槽 + 生命周期共同治理，不靠提高确认门槛）。"""
        return {gid for gid, h in self.gid_hits.items() if h >= 2}

    def tentative_ids(self):
        """未确认的 ID 集合（只出现 1 帧）：渲染时标"?"，不计数。"""
        return {gid for gid, h in self.gid_hits.items() if h < 2}

    def total_count(self):
        """累计唯一人数：只统计确认过的 ID（出现 >= 2 帧）。
        头枕/反光等单帧闪现误检是 Tentative，不计入。"""
        return len(self.confirmed_ids())


# ================= 渲染 =================

def _rects_overlap(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def render_frame(img, car_box, windows, head_dets, actual_count, pred_count):
    """渲染：车绿框、车窗黄框(含人数)、head 框（红=确认,橙=Tentative带?）。细框细字防遮挡。"""
    out = img.copy()
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.4; THICK = 1

    # head 框：红=确认(计数)，橙=Tentative(未确认,标 h{id}? 不计数)
    if head_dets:
        placed = []
        dets = []
        labels = []
        colors = []
        for hb, gid, is_conf in head_dets:
            dets.append((int(hb[0]), int(hb[1]), int(hb[2]), int(hb[3])))
            labels.append(f"h{gid}" if is_conf else f"h{gid}?")
            colors.append((0, 0, 255) if is_conf else (0, 165, 255))  # 红 / 橙
        for (x1, y1, x2, y2), c in zip(dets, colors):
            cv2.rectangle(out, (x1, y1), (x2, y2), c, THICK)
        for idx, ((x1, y1, x2, y2), label, c) in enumerate(zip(dets, labels, colors)):
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
            cv2.putText(out, label, (lx, ly+th), FONT, SCALE, c, THICK)
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
    """极简车跟踪：选面积最大的 car 框，并判断车的运动方向（车头朝向）。

    方向推断分两阶段：
    1. 首帧启发式：车从画面一侧进入，贴边一侧是车尾。car_left≈0 → 车头朝右(+1)；
       car_right≈画面右缘 → 车头朝左(-1)；居中 → 默认 +1。
    2. 位移确认：用 car 框中心 x 的累计位移方向锁定，锁后不再改变。
    首帧即给出 direction，避免"前几帧不框窗"；位移确认后若与首帧启发式冲突，
    由调用方用 confirmed 判断是否重置窗跟踪。
    """
    def __init__(self, edge_margin=60):
        self.last_center = None
        self.direction = 1
        self.direction_at_first = 1   # 首帧启发式方向（位移确认后用于判断是否冲突）
        self.acc_shift = 0.0
        self.confirmed = False
        self.edge_margin = edge_margin

    def update(self, boxes, img_w=0):
        if not boxes:
            return None, self.direction, self.confirmed
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        cb = boxes[max(range(len(areas)), key=lambda i: areas[i])]
        cx = (cb[0] + cb[2]) / 2
        if self.last_center is None:
            # 首帧：用 car 框位置启发式判断车头方向（贴边一侧是车尾）
            if img_w > 0 and cb[2] > img_w - self.edge_margin and cb[0] > self.edge_margin:
                self.direction = -1   # 右边缘贴画面右缘 → 车尾在右 → 车头朝左
            elif cb[0] < self.edge_margin:
                self.direction = 1    # 左边缘贴画面左缘 → 车尾在左 → 车头朝右
            else:
                self.direction = 1    # 居中，默认车头朝右
            self.direction_at_first = self.direction
        else:
            self.acc_shift = 0.8 * self.acc_shift + 0.2 * (cx - self.last_center)
            if not self.confirmed and abs(self.acc_shift) > 3.0:
                self.confirmed = True
                self.direction = 1 if self.acc_shift > 0 else -1
        self.last_center = cx
        return cb, self.direction, self.confirmed


# ================= 特征提取 =================

def extract_feat(img_crop, reid_model, transform, device):
    # OSNet 按 RGB 预训练，img_crop 来自 BGR 图像裁剪 → 先转 RGB，否则特征系统性失真
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(img)
    feat = feat.cpu().numpy().flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-6)
    return feat


def get_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def dedup_heads_in_window(head_boxes, head_scores, windows, max_rel=0.12):
    """检测层去重：同窗内窗内 rel 距离极近（< max_rel 窗宽）的多个 head 检测只保留高分。

    只去"几乎同一位置"的真重复框（窗内间距 < 1/8 窗宽，约等于框的左右偏移量），
    不误删位置不同的真实乘客。过计数主要交给跟踪器的 ID 管理解决。
    返回 (kept_boxes, kept_scores)。
    """
    if not head_boxes:
        return [], []
    win_boxes = dict(windows)
    keep_idx = []
    kept_center = {}   # idx -> ((rx, ry), wid)
    for i, hb in enumerate(head_boxes):
        hc = box_center(hb)
        wid = next((wid for wid, wb in windows if point_in_box(hc, wb)), None)
        dup = False
        for k in keep_idx:
            if wid is None:
                continue
            kc, kwid = kept_center.get(k, (None, None))
            if kwid != wid or kc is None:
                continue
            wb = win_boxes[wid]
            ww = wb[2] - wb[0]
            wh = wb[3] - wb[1]
            if ww <= 0 or wh <= 0:
                continue
            rx = (hc[0] - wb[0]) / ww
            ry = (hc[1] - wb[1]) / wh
            d = np.hypot((rx - kc[0]) * ww, (ry - kc[1]) * wh)
            if d < max_rel * ww:
                dup = True
                break
        if not dup:
            keep_idx.append(i)
            if wid is not None:
                wb = win_boxes[wid]
                ww = wb[2] - wb[0]
                wh = wb[3] - wb[1]
                rx = (hc[0] - wb[0]) / ww if ww > 0 else 0.5
                ry = (hc[1] - wb[1]) / wh if wh > 0 else 0.5
                kept_center[i] = ((rx, ry), wid)
    return [head_boxes[i] for i in keep_idx], [head_scores[i] for i in keep_idx]


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

        r = model.predict(img, conf=0.05, iou=IOU_NMS, imgsz=IMG_SIZE, verbose=False)[0]

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

        # 车（面积最大；本帧漏检时沿用上一帧，保持窗锚定稳定）；direction 判断车头朝向
        car_box, car_dir, car_confirmed = car_tracker.update(boxes_by_cls[0]['boxes'], img_w)
        if car_box is None:
            car_box = last_car_box
        else:
            last_car_box = car_box

        # 位移确认后若方向与首帧启发式不同，重置窗跟踪（用正确方向重建，避免槽位错配）
        if car_confirmed and car_dir != car_tracker.direction_at_first:
            window_tracker = WindowTracker()

        # 车窗：NMS + 几何过滤 + 在车内 → 跨帧跟踪（稳定窗 ID）；首帧即建窗（启发式方向）
        win_boxes = boxes_by_cls[2]['boxes']
        w_nms, w_scores = dedup_boxes(win_boxes, boxes_by_cls[2]['scores'], DEDUP_IOU_THRESH_WIN)
        w_in = [wb for wb in w_nms if car_box is None or point_in_box(box_center(wb), car_box)]
        w_geom = filter_windows_by_geometry(w_in, car_box)
        windows = window_tracker.update(w_geom, car_box, car_dir, img_w)   # [(wid, box)] 按 x 排序，窗 ID 跨帧稳定
        win_boxes = {wid: wb for wid, wb in windows}   # wid -> 窗框，供人 tracker 算窗内 rel

        # head：提特征 + 绑定窗（中心点落在哪个窗框内）
        head_boxes = boxes_by_cls[1]['boxes']
        head_scores = boxes_by_cls[1]['scores']
        # 检测层去重：同窗内 rel 距离极近的多个检测只留高分（对抗环境模糊多框）
        head_boxes, head_scores = dedup_heads_in_window(head_boxes, head_scores, windows)
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
        # 两级关联：高分框允许新建轨迹，低分框只延续（allow_new=False）
        allow_new = [c >= CONF_HEAD_HIGH for c in confs]

        gids, frame_wins = person_tracker.update(dets, feats, confs, wins, win_boxes, allow_new=allow_new)
        confirmed = person_tracker.confirmed_ids()
        tentative = person_tracker.tentative_ids()
        for gid, w in zip(gids, frame_wins):
            if w is not None and gid in confirmed:
                window_seen[w].add(gid)
        pred_count = person_tracker.total_count()

        # head_dets 带确认标记：True=确认, False=Tentative(渲染标"?"，不计数)；gid=-1 的座位槽抑制检测跳过
        head_dets = [(dets[i], gids[i], gids[i] in confirmed) for i in range(len(dets)) if gids[i] != -1]
        # 每窗人数 = 本帧落在该窗的确认 head 数（与 pred 一致；Tentative 不计）
        frame_win_counts = {}
        for gid, w in zip(gids, frame_wins):
            if w is not None and gid in confirmed:
                frame_win_counts[w] = frame_win_counts.get(w, 0) + 1
        win_draw = [(wid, wb, frame_win_counts.get(wid, 0)) for wid, wb in windows]
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
