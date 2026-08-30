# -*- coding: utf-8 -*-
"""对比 V3 与 V4 在同一序列的 ID 确认明细，定位确认机制误伤了谁。"""
import os
import sys
import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import track_3class_window_bind_v4_pending as v4

MODEL_PATH = r"D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target detection/run_3class/weights/best.pt"
DATA_ROOT = r"F:/车辆超员检测项目数据集/超员原图0729-标注人数版本"


def run(seq):
    in_dir = os.path.join(DATA_ROOT, seq)
    frames = sorted([f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    actual = v4.read_actual_count(in_dir)

    model = v4.YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reid_model = v4.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True).to(device).eval()
    transform = v4.transforms.Compose([
        v4.transforms.ToPILImage(),
        v4.transforms.Resize((256, 128)),
        v4.transforms.ToTensor(),
        v4.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    car_tracker = v4.SimpleCarTracker()
    window_tracker = v4.WindowTracker()
    person_tracker = v4.MotionCoherentTracker()
    last_car_box = None

    for fi, name in enumerate(frames):
        img = v4.imread_cn(os.path.join(in_dir, name))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        r = model.predict(img, conf=v4.PRED_CONF, iou=v4.IOU_NMS, imgsz=v4.IMG_SIZE, verbose=False)[0]
        boxes_by_cls = {c: {'boxes': [], 'scores': []} for c in v4.CLASS_NAMES}
        for box in r.boxes:
            cid = int(box.cls[0].cpu().numpy())
            if cid not in v4.CLASS_NAMES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            score = float(box.conf[0].cpu().numpy())
            thr = v4.CONF_CAR if cid == 0 else (v4.CONF_HEAD if cid == 1 else v4.CONF_WIN)
            if score < thr:
                continue
            boxes_by_cls[cid]['boxes'].append([x1, y1, x2, y2])
            boxes_by_cls[cid]['scores'].append(score)

        car_box, car_dir, car_confirmed = car_tracker.update(boxes_by_cls[0]['boxes'], img_w)
        if car_box is None:
            car_box = last_car_box
        else:
            last_car_box = car_box
        if car_confirmed and car_dir != car_tracker.direction_at_first:
            window_tracker = v4.WindowTracker()

        win_boxes = boxes_by_cls[2]['boxes']
        w_nms, w_scores = v4.dedup_boxes(win_boxes, boxes_by_cls[2]['scores'], v4.DEDUP_IOU_THRESH_WIN)
        w_in = [wb for wb in w_nms if car_box is None or v4.point_in_box(v4.box_center(wb), car_box)]
        w_geom = v4.filter_windows_by_geometry(w_in, car_box)
        windows = window_tracker.update(w_geom, car_box, car_dir, img_w)
        win_boxes = {wid: wb for wid, wb in windows}

        head_boxes = boxes_by_cls[1]['boxes']
        head_scores = boxes_by_cls[1]['scores']
        head_boxes, head_scores = v4.dedup_heads_in_window(head_boxes, head_scores, windows)
        dets, feats, confs, wins = [], [], [], []
        for i, hb in enumerate(head_boxes):
            x1, y1, x2, y2 = [int(v) for v in hb]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            feat = v4.extract_feat(crop, reid_model, transform, device)
            dets.append((x1, y1, x2, y2))
            feats.append(feat)
            confs.append(head_scores[i])
            hc = v4.box_center(hb)
            wins.append(next((wid for wid, wb in windows if v4.point_in_box(hc, wb)), None))

        gids, frame_wins = person_tracker.update(dets, feats, confs, wins, win_boxes)

    print(f"\n[序列] {seq} 实际={actual} 预测={person_tracker.total_count()} 差={person_tracker.total_count()-actual}")
    print("gid 明细 (hits, window, avg_conf, confirmed):")
    confirmed = person_tracker.confirmed_ids()
    for gid in sorted(person_tracker.gid_hits):
        h = person_tracker.gid_hits[gid]
        w = person_tracker.gid_window.get(gid)
        avg = person_tracker.gid_conf_sum.get(gid, 0.0) / h if h > 0 else 0.0
        mark = "CONFIRMED" if gid in confirmed else "EXCLUDED"
        print(f"  gid={gid}: hits={h} window={w} avg_conf={avg:.3f} -> {mark}")


if __name__ == "__main__":
    run(sys.argv[1])
