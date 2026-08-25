import os
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
IMAGE_DIR = r"F:/车辆超员检测项目数据集/超员原图0729/20260714151020365"   # 输入：连续帧目录（8帧）
OUTPUT_DIR = r"F:/vehicle_dataset/sideview_output"                          # 输出：追踪结果图
MODEL_PATH = r"F:/vehicle_dataset/runs/head_v2/weights/best.pt"             # 检测模型：head_v2 训练好的 best.pt

CONF_THRES = 0.25   # 检测置信度阈值：低于此值的 head 框不参与追踪
MAX_AGE = 8         # 轨迹最大存活帧数：连续 8 帧没被匹配到就判定"该人消失"

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

    def get_avg_feat(self):
        avg_f = np.mean(np.array(self.feat_history), axis=0)
        return avg_f / (np.linalg.norm(avg_f) + 1e-6)


class MotionCoherentTracker:
    def __init__(self):
        self.tracks = []
        self.next_global_id = 0
        self.global_id_history = {}  # { global_id: [feats] }
        self.global_vehicle_velocity = np.array([0.0, 0.0])  # 全局车速

    def update(self, dets, feats):
        for t in self.tracks:
            t.age += 1

        assigned_indices = [-1] * len(dets)
        matched_tracks = set()

        if len(dets) == 0:
            return ["Detecting..."] * len(dets)

        # 协同运
        predicted_centers = {}
        for j, t in enumerate(self.tracks):
            cx, cy = get_center(t.bbox)
            v = t.velocity if np.linalg.norm(t.velocity) > 0 else self.global_vehicle_velocity
            predicted_centers[j] = np.array([cx, cy]) + v

        # 阶段一：运动补偿位置 + 外观 双指标配对
        matches = []
        for i, det in enumerate(dets):
            d_cx, d_cy = get_center(det)
            for j, t in enumerate(self.tracks):
                if j in matched_tracks:
                    continue
                dist = np.linalg.norm(np.array([d_cx, d_cy]) - predicted_centers[j])
                sim = np.dot(feats[i], t.get_avg_feat())
                # 距离 <150px 且 相似度 >0.68 才认为是同一个人
                if dist < 150 and sim > 0.68:
                    score = sim * 0.7 + (1.0 - dist / 150.0) * 0.3
                    matches.append((score, i, j, dist))

        matches.sort(key=lambda x: x[0], reverse=True)

        velocity_samples = []

        for score, i, j, dist in matches:
            if assigned_indices[i] == -1 and j not in matched_tracks:
                assigned_indices[i] = j
                matched_tracks.add(j)

                t = self.tracks[j]
                old_cx, old_cy = get_center(t.bbox)
                new_cx, new_cy = get_center(dets[i])
                inst_velocity = np.array([new_cx - old_cx, new_cy - old_cy])

                t.velocity = t.velocity * 0.5 + inst_velocity * 0.5
                velocity_samples.append(t.velocity)

                t.bbox = dets[i]
                t.feat_history.append(feats[i])
                t.age = 0

        # 更新全局车速
        if len(velocity_samples) > 0:
            self.global_vehicle_velocity = np.mean(velocity_samples, axis=0)

        # 阶段二：新出现 / 刚恢复的人，全局回溯认领
        # 修复同帧重复ID：本帧新建的 ID 先记录到临时字典，帧尾再写入历史库，
        # 否则本帧后面另一个框回溯时会匹配到这个"刚建的新 ID"，导致两个框拿到同一个 ID。
        new_history_entries = {}
        for i in range(len(dets)):
            if assigned_indices[i] != -1:
                continue

            det_feat = feats[i]
            matched_gid = -1
            best_sim = 0

            for gid, feat_list in self.global_id_history.items():
                sims = np.dot(np.array(feat_list), det_feat)
                max_s = np.max(sims)
                if max_s > best_sim:
                    best_sim = max_s
                    if max_s > 0.76:  # 全局合并门槛
                        matched_gid = gid

            if matched_gid == -1:
                matched_gid = self.next_global_id
                self.next_global_id += 1
                new_history_entries[matched_gid] = [det_feat]   # 原：直接写 self.global_id_history
            else:
                if len(self.global_id_history[matched_gid]) < 5 and best_sim > 0.82:
                    self.global_id_history[matched_gid].append(det_feat)

            new_track = HeadShoulderTrack(matched_gid, dets[i], det_feat)
            self.tracks.append(new_track)
            assigned_indices[i] = len(self.tracks) - 1

        # 帧尾：把本帧新建的 ID 正式写入历史库
        for gid, feat in new_history_entries.items():
            self.global_id_history[gid] = feat

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
    """画框 + 智能放置标签（带底纹，避免互相遮挡）。

    默认放框正上方；若与其他框或已放标签重叠，依次尝试下方(高低两档)、
    框内顶部、左/右侧，并给每个标签加黑色底纹，靠近时也能看清。
    """
    for (x1, y1, x2, y2) in dets:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    placed = []  # 已放置标签的矩形，避免标签之间互相遮挡
    for idx, ((x1, y1, x2, y2), label) in enumerate(zip(dets, labels)):
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        candidates = [
            (x1, y1 - th - 4),        # 上方(贴近)
            (x1, y1 - th - 16),       # 上方(更高)
            (x1, y2 + 4),             # 下方(贴近)
            (x1, y2 + 16),            # 下方(更低)
            (x1 + 2, y1 + th + 2),    # 框内顶部
            (x1 - tw - 4, y1 + th),   # 左侧
            (x2 + 4, y1 + th),        # 右侧
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
        # 黑色底纹 + 绿色文字，保证可读性
        lx, ly = chosen
        cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
        cv2.putText(img, label, chosen, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        placed.append((lx, ly, lx + tw, ly + th))


# ==========================================================
# 4. 主前向循环流水线（路径已改本地，其余与原代码一致）
# ==========================================================
tracker = MotionCoherentTracker()
images = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
])

print(f"[START] 协同运动预测头肩追踪器已激活。")

for i, name in enumerate(images):
    img = imread_cn(os.path.join(IMAGE_DIR, name))
    if img is None:
        continue

    result = yolo.predict(img, conf=CONF_THRES, verbose=False)[0]
    dets, feats = [], []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        feat = extract_feat(crop)
        dets.append((x1, y1, x2, y2))
        feats.append(feat)

    display_labels = tracker.update(dets, feats)

    # 智能标签放置（改动3：标签默认放框上方，重叠时自动换到其他方向）
    draw_boxes_with_labels(img, dets, display_labels)

    v_str = f"Estimated Vehicle Speed: dx={tracker.global_vehicle_velocity[0]:.1f}, dy={tracker.global_vehicle_velocity[1]:.1f}"
    cv2.putText(img, v_str, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.putText(img, f"Total Registered Passengers: {tracker.total_count()}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, name), img)
    if (i + 1) % 10 == 0 or (i + 1) == len(images):
        print(f"[{i+1}/{len(images)}] 运动补偿中 | 当前沉淀总人数 = {tracker.total_count()}")

print("\n" + "=" * 40)
print(f" 🏆 纯头肩检测运动协同去重最终总人数: {tracker.total_count()}")
print("=" * 40)
