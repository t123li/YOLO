# -*- coding: utf-8 -*-
"""V4 实验：D 方案 + Long-Term Identity Memory（待认领池 + Mutual Best Match）。

基于 track_3class_window_bind_annotated.py（V3 baseline）改造，核心差异只在
MotionCoherentTracker：借鉴 McByte++ 的"长期身份记忆"思路治过计数（同一人被拆多 ID）。

改动点（相对 V3）：
1. 【长期特征】每条轨迹在 feat_history（短窗 maxlen=5）之外，维护 long_feat 长期平均特征，
   随观测滚动更新，作为待认领池里的身份表示。
2. 【待认领池】轨迹 age > MAX_AGE 时不再直接删除，而是移入 pending_pool（保留
   global_id / 绑定窗 / long_feat / 最后出现帧），等待未来重现时被认领回原 ID。
3. 【Mutual Best Match 认领】阶段二新增一步：新检测与待认领池做双向最优匹配——
   (a) 检测对池里所有 ID 取余弦相似度最高者；(b) 反过来该 ID 当前也最像这个检测；
   (a)(b) 同时成立且 sim > CLAIM_SIM 才恢复原 ID。加窗口绑定前提（待认领 ID 绑定窗 ==
   检测所在窗才认领）。"少认错比乱认更重要"，防止把相邻相似乘客误并。
4. 【池老化】待认领池按 PENDING_MAX_AGE 过期，避免累积垃圾身份。
5. 【计数口径=进过窗】total_count() 只统计"曾落入车窗内的 ID"。窗外误检/侧景人
   （不在任何车窗内）不算乘客，从根源上治"窗外误检被当成人"的过计数；
   窗内真乘客（含单帧低置信）全部保留，避免欠计数。与渲染口径一致（不出现
   "图上画了、统计却不计"的矛盾）。

计数：全局唯一 person ID 数 = 车内人数；每窗唯一 person ID 数 = 每窗人数。

用法：
    # 批量（21 个无后缀序列）
    D:/Anaconda/envs/course_torch/python.exe track_3class_window_bind_v4_pending.py
    # 单序列调试
    D:/Anaconda/envs/course_torch/python.exe track_3class_window_bind_v4_pending.py <序列名> <输出目录>
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
DATA_ROOT = r"F:/车辆超员检测项目数据集/超员原图0729-标注人数版本"
OUT_ROOT = r"D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target tracking/tracking_output-标注人数版本"

CLASS_NAMES = {0: 'car', 1: 'head', 2: 'window'}

# ================= 检测配置 =================
CONF_HEAD = 0.15    # head 阈值（3类模型 head_shoulder；降低以召回遮挡/小目标，误检交给确认机制）
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
# V4 新增：待认领池（Long-Term Identity Memory）
CLAIM_SIM = 0.80     # Mutual Best Match 认领门限（长期平均特征 vs 检测，取高保真恢复，防误并）
PENDING_MAX_AGE = 30 # 待认领池身份保留帧数（超龄仍未重现则丢弃，防累积垃圾身份）
# V4 新增：同窗匹配位置硬约束（治"h1 从人物A跳到人物B"的同窗 ID 跳变）
WIN_REL_GATE = 0.22  # 同窗匹配时，检测与轨迹的窗内 rel 距离须 < 此阈值（人坐姿稳定，窗内位置不会乱移）
# V4 新增：确认机制（滤低置信单帧碎片，但不误伤高置信真乘客）
HIGH_CONF = 0.7      # 窗内 ID 平均置信度 >= 此值时单帧也确认（真乘客）；低于则需出现>=2帧
# V5 新增：低分 head 框分级处理——高分才新建身份，低分只暂存待高分重现时认领
# （治"遮挡/模糊导致某帧分数低 → 被当新 ID"的碎片过计数，同时不丢低分真头）
CONF_NEW_ID = 0.25    # 新建身份的最低置信度（低于此不建新 ID，进低分暂存池）
LOWCONF_AGE = 5       # 低分暂存池保留帧数（超过则丢弃；覆盖短暂遮挡/模糊）
# 车窗跨帧跟踪：用"窗在车身的位置(relx)"贪心匹配（相邻窗 relx 差约 0.1-0.3，0.2 不会错配）
WIN_ASSOC_RELX = 0.20     # relx 匹配门限（窗相对车框归一化 x 的距离）
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
        self.long_feat = np.array(feat, copy=True)   # V4：长期平均特征（待认领池的身份表示）
        self.long_feat_count = 1                     # V4：long_feat 累计观测数（滚动均值计数）
        self.age = 0
        self.hits = 1                      # 该轨迹被观测到的帧数（确认机制用）
        self.velocity = np.array([0.0, 0.0])       # 该目标独立运动向量 [dx, dy]
        self.missed_shift = np.array([0.0, 0.0])   # 漏检期间累计整车平移，用于修正预测位置
        self.bound_window = win                    # 绑定车窗 ID（None=尚未在窗内见过）
        self.rel_pos = None                        # 最近一次"窗内归一化坐标 (rel_x, rel_y)"
        self.bound_bbox = None                     # 绑定时所在窗的框，用于计算窗内 rel

    def update_long_feat(self, feat):
        """滚动更新长期平均特征（防止短期遮挡期间特征漂移，取稳定平均而非最近一帧）。"""
        n = self.long_feat_count
        self.long_feat = (self.long_feat * n + np.array(feat)) / (n + 1.0)
        self.long_feat = self.long_feat / (np.linalg.norm(self.long_feat) + 1e-6)
        self.long_feat_count = n + 1


class PendingIdentity:
    """待认领池条目：超龄轨迹保留的身份表示，等待未来重现时被 Mutual Best Match 认领。"""
    __slots__ = ('global_id', 'long_feat', 'bound_window', 'last_frame', 'hits')
    def __init__(self, global_id, long_feat, bound_window, last_frame, hits):
        self.global_id = global_id
        self.long_feat = np.array(long_feat, copy=True)
        self.bound_window = bound_window
        self.last_frame = last_frame   # 最后一次活跃/更新帧索引（用于池老化）
        self.hits = hits


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
        self.gid_window = {}          # { gid: 绑定窗ID 或 None }
        self.gid_conf_sum = {}        # V4: { gid: 累计置信度和 }，算平均置信度，供确认机制区分真乘客/误检
        self.global_vehicle_velocity = np.array([0.0, 0.0])
        self.global_shift = np.array([0.0, 0.0])     # RANSAC 估计的整车帧间平移
        self.shift_inliers = 0
        self.pending_pool = {}        # V4: { gid: PendingIdentity } 待认领池（长期身份记忆）
        self.frame_idx = 0            # V4: 帧计数，用于池老化（避免依赖墙钟时间）
        # V5: 低分暂存池。{ gid: (feat, win, last_frame) } —— 低置信 head 框暂存，
        #     不建轨迹/不计数，等待高分重现时认领回同一身份（治碎片 + 不丢低分真头）。
        self.lowconf_pool = {}
        self.lowconf_gids = set()     # V5: 标记哪些 gid 是"低分暂存身份"（正常递增但非正式轨迹）

    def _to_pending(self, t):
        """V4：轨迹超龄时移入待认领池（保留长期特征/绑定窗/观测数），不直接删除。"""
        self.pending_pool[t.global_id] = PendingIdentity(
            t.global_id, t.long_feat, t.bound_window, self.frame_idx, t.hits)

    def _claim_from_pending(self, det_feat, det_win, frame_feats, det_idx):
        """V4：Mutual Best Match 认领。返回 (gid, sim) 或 (None, 0)。

        双向最优 + 窗口绑定前提：
        - 正向：检测 D 在待认领池里最像身份 P*（sim 超 CLAIM_SIM 才进入候选）。
        - 反向：P* 对本帧**所有检测**里最像的也是 D（"它认定我、我也认定它"两边成立）。
        - 待认领 ID 的绑定窗与检测所在窗不一致则排除（沿用窗口绑定硬约束）。
        - "少认错比乱认更重要"：双向最优 + 高门限，防止把相邻相似乘客误并。
        """
        if not self.pending_pool:
            return None, 0.0
        best_gid, best_sim = None, 0.0
        for gid, pe in self.pending_pool.items():
            if det_win is not None and pe.bound_window is not None and pe.bound_window != det_win:
                continue
            sim = float(np.dot(pe.long_feat, det_feat))
            if sim > best_sim:
                best_sim, best_gid = sim, gid
        if best_gid is None or best_sim <= CLAIM_SIM:
            return None, 0.0
        # 反向：P* 对本帧所有检测里最像的是否就是 D（防止 P* 这帧更亲另一个检测）
        best_other = best_sim
        for k, f in enumerate(frame_feats):
            if k == det_idx:
                continue
            s = float(np.dot(self.pending_pool[best_gid].long_feat, f))
            if s > best_other:
                return None, 0.0
        return best_gid, best_sim

    def _bump_gid(self, gid, conf):
        """累加一次 gid 的观测（hits + 置信度和）。确认机制用平均置信度区分真乘客/误检。"""
        self.gid_hits[gid] = self.gid_hits.get(gid, 0) + 1
        self.gid_conf_sum[gid] = self.gid_conf_sum.get(gid, 0.0) + float(conf)

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
        self.frame_idx += 1
        for t in self.tracks:
            t.age += 1

        assigned_indices = [-1] * len(dets)
        matched_tracks = set()
        lowconf_gid_by_det = {}   # V5: { det_idx: 低分暂存 gid }，供 frame_gids 返回

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
                # 同窗匹配：位置硬约束——窗内 rel 距离须 < WIN_REL_GATE 窗宽。
                # 人坐姿稳定，窗内相对位置不会乱移；靠此防止"A模糊/B露出时 h1 从 A 跳到 B"。
                if is_rel:
                    ww = win_boxes.get(wins[i])[2] - win_boxes.get(wins[i])[0]
                    if ww <= 0 or dist >= WIN_REL_GATE * ww:
                        continue
                    gate = max(0.35 * ww, 90)
                else:
                    gate = dist_gate
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
                t.update_long_feat(feats[i])
                t.age = 0
                t.hits += 1
                self._bump_gid(t.global_id, confs[i])
                t.missed_shift = np.array([0.0, 0.0])
                if wins[i] is not None:
                    t.bound_window = wins[i]
                    t.rel_pos = rels[i]
                    t.bound_bbox = win_boxes.get(wins[i])
                    self.gid_window[t.global_id] = wins[i]
                elif t.rel_pos is None:
                    pass  # 仍无窗，保持未绑定

                # V4：一旦被匹配到，移出待认领池（身份已恢复）
                self.pending_pool.pop(t.global_id, None)

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

            # V4：先试"待认领池"的 Mutual Best Match 认领（长期身份恢复，最高优先级，
            # 挡在"新建 ID"之前，治超龄轨迹被拆成新 ID 的过计数）。
            pending_gid, pending_sim = self._claim_from_pending(det_feat, wins[i], feats, i)
            if pending_gid is not None:
                matched_gid = pending_gid
                best_sim = pending_sim
                self.pending_pool.pop(pending_gid, None)   # 认领成功，移出待认领池
                self.gid_window[pending_gid] = wins[i] if wins[i] is not None else self.gid_window.get(pending_gid)
                self._bump_gid(pending_gid, confs[i])

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

            # V5：高分检测认领低分暂存池里的身份（同一人遮挡/模糊后分数回升 → 复用原身份，
            # 不产生新 ID 碎片）。认领到即升级为正式身份：从 lowconf_pool 移到 global_id_history。
            if matched_gid == -1 and confs[i] >= CONF_NEW_ID and self.lowconf_pool:
                best_lc_gid, best_lc_sim = None, 0.0
                for gid, (feat_lc, win_lc, _) in self.lowconf_pool.items():
                    if gid in claimed_gids:
                        continue
                    if not self._win_ok(wins[i], win_lc):
                        continue
                    sim = float(np.dot(feat_lc, det_feat))
                    if sim > best_lc_sim:
                        best_lc_sim, best_lc_gid = sim, gid
                if best_lc_gid is not None and best_lc_sim > claim_thr:
                    matched_gid = best_lc_gid
                    best_sim = best_lc_sim
                    # 升级为正式身份
                    del self.lowconf_pool[best_lc_gid]
                    self.lowconf_gids.discard(best_lc_gid)
                    self.global_id_history[best_lc_gid] = deque([det_feat], maxlen=5)
                    self._bump_gid(best_lc_gid, confs[i])

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
                    # 同窗兜底同样受窗内 rel 硬约束（防阶段二也发生同窗跳变）
                    if is_rel:
                        ww = win_boxes.get(wins[i])[2] - win_boxes.get(wins[i])[0]
                        if ww <= 0 or dist >= WIN_REL_GATE * ww:
                            continue
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

            if matched_gid == -1:
                if confs[i] >= CONF_NEW_ID:
                    # 高分：正常新建正式身份
                    matched_gid = self.next_global_id
                    self.next_global_id += 1
                    self.global_id_history[matched_gid] = deque([det_feat], maxlen=5)
                    self.gid_hits[matched_gid] = 0
                    self.gid_conf_sum[matched_gid] = 0.0
                    self._bump_gid(matched_gid, confs[i])
                    self.gid_window[matched_gid] = wins[i]   # 首次出现即绑定车窗
                else:
                    # 低分：不建正式 ID，进低分暂存池（分配临时 gid，标记 lowconf）。
                    # 不建轨迹/不计数，等高分重现时被阶段二认领回同一身份；
                    # 若只是噪声（头枕/反光），5 帧内无人认领则自动老化丢弃。
                    matched_gid = self.next_global_id
                    self.next_global_id += 1
                    self.lowconf_pool[matched_gid] = (det_feat, wins[i], self.frame_idx)
                    self.lowconf_gids.add(matched_gid)
                    self.gid_window[matched_gid] = wins[i]
                    self.gid_hits[matched_gid] = 1      # 暂存身份也记 1 次，防后续误判
                    self.gid_conf_sum[matched_gid] = float(confs[i])
                    lowconf_gid_by_det[i] = matched_gid
            else:
                self.global_id_history[matched_gid].append(det_feat)
                self._bump_gid(matched_gid, confs[i])
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
                t.update_long_feat(det_feat)
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
                if matched_gid in self.lowconf_gids:
                    # V5: 低分暂存身份不建轨迹（不参与运动匹配/不计数），只留 lowconf_pool
                    # 等待高分认领。assigned_indices 保持 -1，frame_gids 循环对 -1 特殊处理。
                    pass
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
        for i, idx in enumerate(assigned_indices):
            if idx == -1:
                # V5: 低分暂存框（未建轨迹）——返回其暂存 gid，渲染可用；不计数
                if i in lowconf_gid_by_det:
                    frame_gids.append(lowconf_gid_by_det[i])
                    frame_wins.append(self.gid_window.get(lowconf_gid_by_det[i]))
                else:
                    frame_gids.append(-1)
                    frame_wins.append(None)
                continue
            t = self.tracks[idx]
            frame_gids.append(t.global_id)
            frame_wins.append(t.bound_window)

        # V4：超龄轨迹移入待认领池（而非直接删除），供长期身份恢复；保留仍在等待的池条目
        alive = []
        for t in self.tracks:
            if t.age <= MAX_AGE:
                alive.append(t)
            else:
                self._to_pending(t)
        self.tracks = alive

        # V4：待认领池老化——超过 PENDING_MAX_AGE 帧仍未重现的身份丢弃（防累积垃圾身份）
        stale = [gid for gid, pe in self.pending_pool.items()
                 if self.frame_idx - pe.last_frame > PENDING_MAX_AGE]
        for gid in stale:
            del self.pending_pool[gid]

        # V5：低分暂存池老化——超过 LOWCONF_AGE 帧仍未升级为正式身份则丢弃
        # （短暂遮挡/模糊的低分真头会在几帧内被高分认领；噪声则自然过期）
        stale_low = [gid for gid, (_, _, last) in self.lowconf_pool.items()
                     if self.frame_idx - last > LOWCONF_AGE]
        for gid in stale_low:
            del self.lowconf_pool[gid]
            self.lowconf_gids.discard(gid)
            self.gid_window.pop(gid, None)
            self.gid_hits.pop(gid, None)
            self.gid_conf_sum.pop(gid, None)
        return frame_gids, frame_wins

    def confirmed_ids(self):
        """确认过的 ID 集合 = 进过窗，且 (hits>=2 或 平均置信度>=HIGH_CONF)。

        - 窗外 ID 一律不算（窗外误检/侧景人，不是车内乘客）。
        - 窗内低置信单帧碎片（avg_conf < HIGH_CONF 且只出现 1 帧）滤掉——
          这类多是紧邻重复框/模糊误检，不确认防过计数。
        - 窗内高置信单帧（avg_conf >= HIGH_CONF）仍确认——短序列里只出现
          1 帧的真乘客，避免欠计数（不出现"图上画了、统计却不计"）。
        """
        out = set()
        for gid, h in self.gid_hits.items():
            if gid in self.lowconf_gids:
                continue   # V5: 低分暂存身份永不计入（等待高分认领，不直接计数）
            if self.gid_window.get(gid) is None:
                continue
            if h >= 2:
                out.add(gid)
                continue
            avg_conf = self.gid_conf_sum.get(gid, 0.0) / h if h > 0 else 0.0
            if avg_conf >= HIGH_CONF:
                out.add(gid)
        return out

    def total_count(self):
        """累计唯一人数 = 确认过的 ID 数。

        窗外误检被"进过窗"过滤；窗内真乘客（含高置信单帧）保留；
        低置信单帧碎片被滤掉（防过计数）。"""
        return len(self.confirmed_ids())


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
    img = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(img)
    feat = feat.cpu().numpy().flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-6)
    return feat


def get_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def dedup_heads_in_window(head_boxes, head_scores, windows, max_rel=0.07):
    """检测层去重：同窗内窗内 rel 距离极近（< max_rel 窗宽）的多个 head 检测只保留高分。

    只去"几乎同一位置"的真重复框（窗内间距 < 0.07 窗宽，约半个头宽，即同一头的双框），
    不误删位置不同的真实乘客。紧邻乘客中心距约 0.1 窗宽，0.07 不会误删。
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

        gids, frame_wins = person_tracker.update(dets, feats, confs, wins, win_boxes)
        for gid, w in zip(gids, frame_wins):
            if w is not None:
                window_seen[w].add(gid)
        pred_count = person_tracker.total_count()

        head_dets = [(dets[i], gids[i]) for i in range(len(dets))]
        # 每窗人数 = 本帧落在该窗的 head 检测数（当前帧可见人数，不做确认过滤，避免滞后）
        frame_win_counts = {}
        for gid, w in zip(gids, frame_wins):
            if w is not None:
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

    # 批量模式：只跑 21 个无后缀序列（带后缀的为数据质量较差样本，本轮不跑）
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

    # 无后缀 = 数据质量较好；带后缀（-small target/-blurry head/-bully head/-target overlap）本轮排除
    seqs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and "-" not in d])

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
