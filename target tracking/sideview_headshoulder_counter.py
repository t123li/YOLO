import os
import sys
import cv2
import numpy as np
import torch
from collections import deque
from ultralytics import YOLO
from torchreid import models
from torchvision import transforms

# ==========================================================
# 1. 核心路径与配置（本地版：原服务器路径已注释保留）
# ==========================================================
# 原代码（服务器路径，本地不存在，已注释）：
# IMAGE_DIR = "/mnt/disk1/wsk/ultralytics/ultralytics/data/test_data/xfh_yt"
# OUTPUT_DIR = "/mnt/disk1/wsk/ultralytics/ultralytics/data/test_data/xfh_yt_output"
# MODEL_PATH = "/mnt/disk1/wsk/ultralytics/runs/detect/xd_xfh_260616/train/v8s/weights/best.pt"

# 本地路径（改动1：换成你电脑上的帧目录、输出目录、模型）
# 支持命令行传入输入目录，自动绑定输出目录：
# python sideview_run_20260715175447965.py <输入目录> [输出目录]
DEFAULT_IMAGE_DIR = r"F:/车辆超员检测项目数据集/超员原图0729/20260715175447965"
DEFAULT_OUTPUT_DIR = r"F:/vehicle_dataset"
if len(sys.argv) >= 2 and sys.argv[1].strip():
    IMAGE_DIR = os.path.normpath(sys.argv[1])
    folder_name = os.path.basename(IMAGE_DIR)
    OUTPUT_DIR = os.path.join(r"F:/vehicle_dataset", f"sideview_output_{folder_name}")
    if len(sys.argv) >= 3 and sys.argv[2].strip():
        OUTPUT_DIR = os.path.normpath(sys.argv[2])
else:
    IMAGE_DIR = DEFAULT_IMAGE_DIR
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR
MODEL_PATH = r"D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target detection/run_head_seqsplit/weights/best.pt"             # 检测模型：run_head_seqsplit 序列级划分重训的 best.pt

CONF_THRES = 0.6    # 检测置信度阈值：低于此值的 head 框不参与追踪（0.25 太松，会混入头枕/反光误检）
MAX_AGE = 8         # 轨迹最大存活帧数：连续 8 帧没被匹配到就判定"该人消失"
SPATIAL_R = 70      # 空间兜底半径：外观匹配失败时，位置在半径内的存活轨迹沿用其 ID
DIST_GATE = 80      # 阶段一位置门限：整车平移已由 RANSAC 补偿，剩余位移应很小（RANSAC 失效时回退用 150）

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2. 神经网络初始化
# ==========================================================
yolo = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------
# 原代码：用 torchreid 的 OSNet 提取外观特征（ReID）
# 本地改动2（已解决）：OSNet 预训练权重原本要从 Google Drive 下载、国内访问不了，
#           后来开了代理、权重已下载到本地缓存 ~/.cache/torch/checkpoints/，
#           所以现在恢复使用 OSNet，效果最好。
# ----------------------------------------------------------
reid_model = models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
).to(device).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_feat(img_crop):
    """把一个人头裁剪图变成一个归一化的外观特征向量（OSNet）。"""
    img = transform(img_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(img)
    feat = feat.cpu().numpy().flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-6)  # L2 归一化，方便点积算相似度
    return feat


def get_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def read_actual_count(image_dir):
    """从输入序列目录读取人工核实的总人数（total_registered.txt，格式：total_registered <N>）。

    找不到文件时返回 None（不显示对照）。用于快速判断跟踪计数的准确度。
    """
    for cand in ("total_registered.txt", "actual_count.txt"):
        p = os.path.join(image_dir, cand)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    text = f.read().strip()
                # 支持 "total_registered 7" 或纯 "7" 两种格式
                for tok in text.replace(",", " ").split():
                    if tok.isdigit():
                        return int(tok)
            except Exception as e:
                print(f"[警告] 读取 {cand} 失败: {e}")
                return None
    return None


def imread_cn(path):
    """安全读取中文路径图片（cv2.imread 对中文路径会返回 None）。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

# ==========================================================
# 3. 协同运动与拓扑锁定追踪器（逻辑与原代码完全一致）
# ==========================================================
class HeadShoulderTrack:
    def __init__(self, global_id, bbox, feat):
        self.global_id = global_id
        self.bbox = bbox
        self.feat_history = deque(maxlen=5)
        self.feat_history.append(feat)
        self.age = 0
        self.velocity = np.array([0.0, 0.0])  # 该目标的独立运动向量 [dx, dy]
        self.missed_shift = np.array([0.0, 0.0])  # 漏检期间累计的整车平移，用于修正预测位置


class MotionCoherentTracker:
    def __init__(self):
        self.tracks = []
        self.next_global_id = 0
        self.global_id_history = {}  # { global_id: deque([feat,...], maxlen=5) }
        self.global_vehicle_velocity = np.array([0.0, 0.0])  # 全局车速
        self.global_shift = np.array([0.0, 0.0])  # RANSAC 估计的整车帧间平移（GMC 简化版）
        self.shift_inliers = 0

    def _estimate_global_shift(self, dets, tol=30.0):
        """RANSAC 估计整车帧间平移：寻找使最多 (旧轨迹中心, 新检测中心) 点对对齐的位移。

        对应论文 GMC（稀疏光流 + RANSAC 仿射变换）的思想，但直接用检测点代替光流关键点。
        """
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

    def update(self, dets, feats, confs):
        for t in self.tracks:
            t.age += 1

        assigned_indices = [-1] * len(dets)
        matched_tracks = set()

        if len(dets) == 0:
            return ["Detecting..."] * len(dets)

        # ---- GMC 简化版（论文"稀疏光流+RANSAC"的检测点版）----
        # 用 RANSAC 从 (旧轨迹, 新检测) 点对里估计整车帧间平移，
        # 解决车速 EMA 在车辆加速/减速时预测滞后导致的跨帧错配（ID 切换主因）。
        self.global_shift, self.shift_inliers = self._estimate_global_shift(dets)
        use_shift = self.shift_inliers >= 2

        predicted_centers = {}
        for j, t in enumerate(self.tracks):
            cx, cy = get_center(t.bbox)
            if use_shift:
                # bbox 是上次匹配时的位置；若中途漏检，需加上漏检期间累计的整车平移
                predicted_centers[j] = np.array([cx, cy]) + self.global_shift + t.missed_shift
            else:
                v = t.velocity if np.linalg.norm(t.velocity) > 0 else self.global_vehicle_velocity
                predicted_centers[j] = np.array([cx, cy]) + v * t.age

        dist_gate = DIST_GATE if use_shift else 150.0

        # 阶段一：运动补偿位置 + 外观 双指标配对
        # 论文改进3：代价矩阵自适应加权（式4-19/4-20），w_m 由检测置信度经 S 型函数决定。
        # 注意：论文原式在高置信时 w_m→1（纯运动匹配），适用于 MOT17 大目标场景；
        # 本场景人头框小(40-80px)、位置噪声 15-20px、邻座间距仅 ~35px，运动区分不了
        # 邻座，因此把 S 型曲线映射到 [0.2, 0.5]，外观始终占主导、置信度只做微调。
        matches = []
        for i, det in enumerate(dets):
            d_cx, d_cy = get_center(det)
            c = confs[i]
            s_c = (c * c) / (c * c + (1.0 - c) * (1.0 - c))
            w_m = 0.2 + 0.3 * s_c
            for j, t in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                dist = np.linalg.norm(np.array([d_cx, d_cy]) - predicted_centers[j])
                sims = np.dot(np.array(t.feat_history), feats[i])
                sim = float(np.max(sims))
                if dist < dist_gate and sim > 0.68:
                    motion_score = 1.0 - dist / dist_gate
                    score = w_m * motion_score + (1.0 - w_m) * sim
                    matches.append((score, i, j, dist, sim))

        matches.sort(key=lambda x: x[0], reverse=True)

        velocity_samples = []

        for score, i, j, dist, sim in matches:
            if assigned_indices[i] == -1 and j not in matched_tracks:
                assigned_indices[i] = j
                matched_tracks.add(j)

                t = self.tracks[j]
                old_cx, old_cy = get_center(t.bbox)
                new_cx, new_cy = get_center(dets[i])
                inst_velocity = np.array([new_cx - old_cx, new_cy - old_cy])

                # 论文改进2(NSA, 式4-18)思想：高置信→更信任新观测，低置信→保守更新
                w = float(np.clip(confs[i], 0.2, 0.9))
                t.velocity = t.velocity * (1.0 - w) + inst_velocity * w
                velocity_samples.append(t.velocity)

                t.bbox = dets[i]
                t.feat_history.append(feats[i])
                t.age = 0
                t.missed_shift = np.array([0.0, 0.0])

        # 更新全局车速
        if len(velocity_samples) > 0:
            self.global_vehicle_velocity = np.mean(velocity_samples, axis=0)

        # 阶段二：新出现 / 刚恢复的人，全局回溯认领
        # 修复1：claimed_gids 保证"一个 gid 本帧只认领一次"，避免同帧重复 ID；
        #        同时把阶段一已匹配到的 gid 也加入，防止阶段二重复认领。
        claimed_gids = {self.tracks[j].global_id for j in matched_tracks}
        pre_track_indices = list(range(len(self.tracks)))  # 阶段二开始时已有的轨迹快照

        for i in range(len(dets)):
            if assigned_indices[i] != -1:
                continue

            det_feat = feats[i]
            matched_gid = -1
            best_sim = 0

            # 历史库：每个 gid 存最近若干帧特征(deque)，用 max 匹配（比 EMA 单向量更稳）
            # 论文改进2(NSA)思想：低置信度检测"权威性"低，需更高相似度才能认领旧 ID
            claim_thr = 0.76 + (1.0 - confs[i]) * 0.10
            for gid, feat_list in self.global_id_history.items():
                if gid in claimed_gids:
                    continue
                sims = np.dot(np.array(feat_list), det_feat)
                max_s = float(np.max(sims))
                if max_s > best_sim:
                    best_sim = max_s
                    if max_s > claim_thr:  # 全局合并门槛（置信度自适应）
                        matched_gid = gid

            # 空间兜底：外观匹配失败时，若某条"未被匹配的存活轨迹"预测位置离这个
            # 检测框很近，说明是同一个人只是外观暂时变差，直接沿用其 gid。
            spatial_j = -1
            if matched_gid == -1:
                d_cx, d_cy = get_center(dets[i])
                best_d = SPATIAL_R * (0.5 + 0.5 * confs[i])   # 低置信→更小兜底半径
                for j in pre_track_indices:
                    if j in matched_tracks:
                        continue
                    if self.tracks[j].global_id in claimed_gids:
                        continue
                    dist = np.linalg.norm(np.array([d_cx, d_cy]) - predicted_centers[j])
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
            else:
                self.global_id_history[matched_gid].append(det_feat)  # 滚动更新
            claimed_gids.add(matched_gid)

            tag = " (spatial)" if spatial_j != -1 else ""
            print(f"    stage2: det#{i} best_sim={best_sim:.3f} -> gid {matched_gid}{tag}")

            # 修复2：若该 gid 已有存活 track，直接复用更新它，而不是再建一条
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
                assigned_indices[i] = existing_idx
            else:
                new_track = HeadShoulderTrack(matched_gid, dets[i], det_feat)
                self.tracks.append(new_track)
                assigned_indices[i] = len(self.tracks) - 1

        # 本帧未被匹配的旧轨迹：累计整车平移，供下次预测修正位置
        acc_shift = self.global_shift if use_shift else self.global_vehicle_velocity
        for j in pre_track_indices:
            if j not in matched_tracks:
                self.tracks[j].missed_shift += acc_shift

        # 生成本帧标签
        frame_labels = []
        for idx in assigned_indices:
            t = self.tracks[idx]
            frame_labels.append(f"Passenger_ID:{t.global_id}")

        # 清理消亡轨迹
        self.tracks = [t for t in self.tracks if t.age <= MAX_AGE]

        return frame_labels

    def total_count(self):
        return len(self.global_id_history)


def _rects_overlap(a, b):
    """判断两个矩形 (x1, y1, x2, y2) 是否重叠。"""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def draw_boxes_with_labels(img, dets, labels):
    """画框 + 智能放置标签（细框细字，避免互相遮挡）。

    框线宽 1、字号 0.4/线宽 1，适合目标多、框密的情况。
    标签默认放框正上方；若与其他框或已放标签重叠，依次尝试下方(高低两档)、
    框内顶部、左/右侧，并给每个标签加半透明黑色底纹，保证可读且不遮挡。
    """
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.4
    THICK = 1
    BOX_THICK = 1

    # 先画所有框（细框）
    for (x1, y1, x2, y2) in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), BOX_THICK)

    placed = []  # 已放置标签的矩形，避免标签之间互相遮挡
    for idx, ((x1, y1, x2, y2), label) in enumerate(zip(dets, labels)):
        (tw, th), _ = cv2.getTextSize(label, FONT, SCALE, THICK)
        # 候选位置：上/下/框内/左/右，细框下给更密的候选档
        candidates = [
            (x1, y1 - th - 3),        # 上方(贴近)
            (x1, y1 - th - 11),       # 上方(更高)
            (x1, y2 + 3),             # 下方(贴近)
            (x1, y2 + 11),            # 下方(更低)
            (x1 + 2, y1 + th + 2),    # 框内顶部
            (x1 - tw - 3, y1 + th),   # 左侧
            (x2 + 3, y1 + th),        # 右侧
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
        # 半透明黑色底纹 + 绿色细字，保证可读性且不压住其他框
        lx, ly = chosen
        ov = img[ly:ly + th, lx:lx + tw]
        if ov.size > 0:
            img[ly:ly + th, lx:lx + tw] = cv2.addWeighted(ov, 0.4, np.zeros_like(ov), 0.6, 0)
        cv2.putText(img, label, (lx, ly + th), FONT, SCALE, (0, 255, 0), THICK)
        placed.append((lx, ly, lx + tw, ly + th))


# ==========================================================
# 4. 主前向循环流水线（路径已改本地，其余与原代码一致）
# ==========================================================
tracker = MotionCoherentTracker()
images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
])

# 读取人工核实的实际总人数（可选，用于快速判断计数准确度）
ACTUAL_COUNT = read_actual_count(IMAGE_DIR)
if ACTUAL_COUNT is not None:
    print(f"[INFO] 人工核实实际人数 = {ACTUAL_COUNT}（来自 {IMAGE_DIR}/total_registered.txt）")

print(f"[START] 协同运动预测头肩追踪器已激活。")

for i, name in enumerate(images):
    img = imread_cn(os.path.join(IMAGE_DIR, name))
    if img is None:
        continue

    result = yolo.predict(img, conf=CONF_THRES, verbose=False)[0]
    dets, feats, confs = [], [], []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        feat = extract_feat(crop)
        dets.append((x1, y1, x2, y2))
        feats.append(feat)
        confs.append(float(box.conf[0].cpu().numpy()))

    display_labels = tracker.update(dets, feats, confs)

    # 标签带置信度：Passenger_ID:3 0.92（细框细字渲染，防遮挡）
    render_labels = [f"{lbl} {c:.2f}" for lbl, c in zip(display_labels, confs)]

    ids_in_frame = [lbl.split(":")[-1] for lbl in display_labels]
    det_str = ", ".join(f"({d[0]},{d[1]})c={c:.2f}" for d, c in zip(dets, confs))
    print(f"[{i}/{len(images)}] {name} -> IDs = {ids_in_frame}")
    print(f"      dets: {det_str}")
    print(f"      ransac shift=({tracker.global_shift[0]:.0f},{tracker.global_shift[1]:.0f}) inliers={tracker.shift_inliers}")

    # 智能标签放置（改动3：标签默认放框上方，重叠时自动换到其他方向）
    draw_boxes_with_labels(img, dets, render_labels)

    v_str = f"RANSAC Shift: dx={tracker.global_shift[0]:.1f}, dy={tracker.global_shift[1]:.1f} (inliers={tracker.shift_inliers})"
    cv2.putText(img, v_str, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.putText(img, f"Total Registered Passengers: {tracker.total_count()}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if ACTUAL_COUNT is not None:
        diff = tracker.total_count() - ACTUAL_COUNT
        color = (0, 255, 0) if diff == 0 else ((0, 165, 255) if abs(diff) <= 1 else (0, 0, 255))
        cv2.putText(img, f"Actual: {ACTUAL_COUNT}  Diff: {diff:+d}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, name), img)
    if (i + 1) % 10 == 0 or (i + 1) == len(images):
        print(f"[{i+1}/{len(images)}] 运动补偿中 | 当前沉淀总人数 = {tracker.total_count()}")

print("\n" + "=" * 40)
print(f" 🏆 纯头肩检测运动协同去重最终总人数: {tracker.total_count()}")
if ACTUAL_COUNT is not None:
    diff = tracker.total_count() - ACTUAL_COUNT
    status = "准确" if diff == 0 else (f"偏差{diff:+d}" if abs(diff) <= 1 else f"偏差较大{diff:+d}")
    print(f" 📌 人工核实实际人数: {ACTUAL_COUNT}，差值 {diff:+d}（{status}）")
print("=" * 40)
