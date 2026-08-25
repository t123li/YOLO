import os
from collections import deque

import cv2
import numpy as np
import torch
from torchreid import models
from torchvision import transforms

from ultralytics import YOLO

# ==========================================================
# 1. 核心路径与配置
# ==========================================================
IMAGE_DIR = "/mnt/disk1/wsk/ultralytics/ultralytics/data/test_data/xfh_yt"
OUTPUT_DIR = "/mnt/disk1/wsk/ultralytics/ultralytics/data/test_data/xfh_yt_output"
MODEL_PATH = "/mnt/disk1/wsk/ultralytics/runs/detect/xd_xfh_260616/train/v8s/weights/best.pt"

CONF_THRES = 0.25
MAX_AGE = 8  # 允许颠簸漏检 8 帧

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2. 神经网络初始化 (YOLOv8 + ReID OSNet)
# ==========================================================
yolo = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

reid_model = models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True).to(device).eval()

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_feat(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = reid_model(img)
    feat = feat.cpu().numpy().flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-6)
    return feat


def get_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


# ==========================================================
# 3. 协同运动与拓扑锁定追踪器
# ==========================================================
class HeadShoulderTrack:
    def __init__(self, global_id, bbox, feat):
        self.global_id = global_id
        self.bbox = bbox
        self.feat_history = deque(maxlen=5)
        self.feat_history.append(feat)
        self.age = 0
        self.velocity = np.array([0.0, 0.0])  # 记录该目标的独立运动向量 [dx, dy]

    def get_avg_feat(self):
        avg_f = np.mean(np.array(self.feat_history), axis=0)
        return avg_f / (np.linalg.norm(avg_f) + 1e-6)


class MotionCoherentTracker:
    def __init__(self):
        self.tracks = []
        self.next_global_id = 0
        self.global_id_history = {}  # { global_id: [feats] }

        # 全局背景车速预测器 [avg_dx, avg_dy]
        self.global_vehicle_velocity = np.array([0.0, 0.0])

    def update(self, dets, feats):
        # 1. 寿命老化
        for t in self.tracks:
            t.age += 1

        assigned_indices = [-1] * len(dets)
        matched_tracks = set()

        if len(dets) == 0:
            return ["Detecting..."] * len(dets)

        # 2. 【核心】协同运动预测（利用上一帧推算出的全局车速，给所有活着的 Track 补尝位移）
        predicted_centers = {}
        for j, t in enumerate(self.tracks):
            cx, cy = get_center(t.bbox)
            # 如果该目标自身有速度优先用自身的，没有则用全车平均速度来“预测”它当前帧应该漂移到哪
            v = t.velocity if np.linalg.norm(t.velocity) > 0 else self.global_vehicle_velocity
            predicted_centers[j] = np.array([cx, cy]) + v

        # 3. 阶段一：基于【运动补偿位置 + ReID双指标】进行精细配对
        matches = []
        for i, det in enumerate(dets):
            d_cx, d_cy = get_center(det)
            for j, t in enumerate(self.tracks):
                if j in matched_tracks:
                    continue

                # 计算检测框中心点与该 Track “预测位移点”的实际物理距离
                dist = np.linalg.norm(np.array([d_cx, d_cy]) - predicted_centers[j])
                sim = np.dot(feats[i], t.get_avg_feat())

                # 运动抗干扰核心：只要在全车平移补偿后，距离在 150 像素以内，且特征过得去(>0.68)
                if dist < 150 and sim > 0.68:
                    # 综合得分：相似度越高、离预测点越近，得分越高
                    score = sim * 0.7 + (1.0 - dist / 150.0) * 0.3
                    matches.append((score, i, j, dist))

        matches.sort(key=lambda x: x[0], reverse=True)

        # 用于计算当前帧实际全局车身车速的增量池
        velocity_samples = []

        for score, i, j, dist in matches:
            if assigned_indices[i] == -1 and j not in matched_tracks:
                assigned_indices[i] = j
                matched_tracks.add(j)

                t = self.tracks[j]
                # 计算真实的帧间瞬时位移速度
                old_cx, old_cy = get_center(t.bbox)
                new_cx, new_cy = get_center(dets[i])
                inst_velocity = np.array([new_cx - old_cx, new_cy - old_cy])

                # 动量平滑更新目标自身速度
                t.velocity = t.velocity * 0.5 + inst_velocity * 0.5
                velocity_samples.append(t.velocity)

                # 更新状态
                t.bbox = dets[i]
                t.feat_history.append(feats[i])
                t.age = 0

        # 4. 动态更新“全局背景车速”，供下一帧那些被遮挡、漏检的人作位置补偿
        if len(velocity_samples) > 0:
            self.global_vehicle_velocity = np.mean(velocity_samples, axis=0)

        # 5. 阶段二：处理新出现的、或者漏检严重刚恢复的人（全局排他重连）
        for i in range(len(dets)):
            if assigned_indices[i] != -1:
                continue

            det_feat = feats[i]
            matched_gid = -1
            best_sim = 0

            # 跨时空大回溯：去历史库里找最神似的身份证
            for gid, feat_list in self.global_id_history.items():
                sims = np.dot(np.array(feat_list), det_feat)
                max_s = np.max(sims)
                if max_s > best_sim:
                    best_sim = max_s
                    if max_s > 0.76:  # 严格的全局合并门槛
                        matched_gid = gid

            if matched_gid == -1:
                # 实在找不到，开辟新全局 ID
                matched_gid = self.next_global_id
                self.next_global_id += 1
                self.global_id_history[matched_gid] = [det_feat]
            else:
                if len(self.global_id_history[matched_gid]) < 5 and best_sim > 0.82:
                    self.global_id_history[matched_gid].append(det_feat)

            # 实例化新追踪并推入队列
            new_track = HeadShoulderTrack(matched_gid, dets[i], det_feat)
            self.tracks.append(new_track)
            assigned_indices[i] = len(self.tracks) - 1

        # 6. 生成本帧渲染标签
        frame_labels = []
        for idx in assigned_indices:
            t = self.tracks[idx]
            frame_labels.append(f"Passenger_ID:{t.global_id}")

        # 7. 清理消亡轨迹
        self.tracks = [t for t in self.tracks if t.age <= MAX_AGE]

        return frame_labels

    def total_count(self):
        return len(self.global_id_history)


# ==========================================================
# 4. 主前向循环流水线
# ==========================================================
tracker = MotionCoherentTracker()
images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))])

print("[START] 协同运动预测头肩追踪器已激活。")

for i, name in enumerate(images):
    img = cv2.imread(os.path.join(IMAGE_DIR, name))
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

    # 协同运动更新
    display_labels = tracker.update(dets, feats)

    # 可视化渲染
    for (x1, y1, x2, y2), label in zip(dets, display_labels):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 可视化：在左下角打印当前追踪器感知到的全局车辆移动速度（像素/帧）
    v_str = f"Estimated Vehicle Speed: dx={tracker.global_vehicle_velocity[0]:.1f}, dy={tracker.global_vehicle_velocity[1]:.1f}"
    cv2.putText(img, v_str, (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.putText(
        img,
        f"Total Registered Passengers: {tracker.total_count()}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR, name), img)
    if (i + 1) % 10 == 0 or (i + 1) == len(images):
        print(f"[{i + 1}/{len(images)}] 运动补偿中 | 当前沉淀总人数 = {tracker.total_count()}")

print("\n" + "=" * 40)
print(f" 🏆 纯头肩检测运动协同去重最终总人数: {tracker.total_count()}")
print("=" * 40)
