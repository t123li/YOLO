# -*- coding: utf-8 -*-
"""诊断：单序列逐帧打印 head 检测与 ID 分配，定位过计数来源。

用法：
    D:/Anaconda/envs/course_torch/python.exe diag_v4_ids.py 20260719221354578
"""
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


def main():
    seq = sys.argv[1]
    in_dir = os.path.join(DATA_ROOT, seq)
    frames = sorted([f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    actual = v4.read_actual_count(in_dir)
    print(f"[序列] {seq} 实际={actual} 帧数={len(frames)}")

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
    window_seen = {}
    last_car_box = None
    first_seen = {}   # gid -> 首次出现帧

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
        for gid, w in zip(gids, frame_wins):
            if w is not None:
                window_seen[w] = window_seen.get(w, set()) | {gid}
            if gid not in first_seen:
                first_seen[gid] = fi

        # 本帧新出现的 ID 单独标出
        new_in_frame = [gid for gid in gids if first_seen[gid] == fi]
        print(f"\n--- 帧{fi} {name} 检测数={len(dets)} 累计ID={person_tracker.total_count()} 池={len(person_tracker.pending_pool)} ---")
        for i, (hb, gid, w) in enumerate(zip(dets, gids, frame_wins)):
            hc = v4.box_center(hb)
            mark = "  <-- 新ID" if gid in new_in_frame else ""
            print(f"  det{i}: box=({int(hb[0])},{int(hb[1])},{int(hb[2])},{int(hb[3])}) "
                  f"conf={confs[i]:.3f} c=({int(hc[0])},{int(hc[1])}) win={w} -> gid={gid}{mark}")

    print(f"\n===== 汇总 =====")
    print(f"实际={actual} 预测={person_tracker.total_count()} 差={person_tracker.total_count() - actual}")
    print(f"首次出现帧分布: {dict(sorted(first_seen.items()))}")
    print(f"每窗ID: { {w: len(s) for w, s in sorted(window_seen.items())} }")


if __name__ == "__main__":
    main()
