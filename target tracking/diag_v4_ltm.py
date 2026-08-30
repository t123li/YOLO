# -*- coding: utf-8 -*-
"""验证：Long-Term Identity Memory 能否治 `20260715071110945` 的 ID 跳变。

假设：帧3 让 gid=3 闲置进待认领池（不抢相邻检测），帧4 它重现时靠 OSNet
Mutual Best Match 认领回原 ID。本脚本验证前提——重现帧的 OSNet 特征和待认领池
里 gid=3 的长期特征，相似度能否过 CLAIM_SIM=0.80。

跑法：python diag_v4_ltm.py
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
SEQ = "20260715071110945"


def main():
    in_dir = os.path.join(DATA_ROOT, SEQ)
    frames = sorted([f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"[序列] {SEQ} 帧数={len(frames)}")

    model = v4.YOLO(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reid_model = v4.models.build_model(name="osnet_x1_0", num_classes=1000, pretrained=True).to(device).eval()
    transform = v4.transforms.Compose([
        v4.transforms.ToPILImage(),
        v4.transforms.Resize((256, 128)),
        v4.transforms.ToTensor(),
        v4.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 逐帧收集所有检测的 (bbox, conf, window, feat)，不跑跟踪器，只看特征相似度
    frame_dets = []   # 每帧: list of (bbox, conf, win, feat)
    car_tracker = v4.SimpleCarTracker()
    window_tracker = v4.WindowTracker()
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

        w_nms, _ = v4.dedup_boxes(boxes_by_cls[2]['boxes'], boxes_by_cls[2]['scores'], v4.DEDUP_IOU_THRESH_WIN)
        w_in = [wb for wb in w_nms if car_box is None or v4.point_in_box(v4.box_center(wb), car_box)]
        w_geom = v4.filter_windows_by_geometry(w_in, car_box)
        windows = window_tracker.update(w_geom, car_box, car_dir, img_w)
        win_boxes = {wid: wb for wid, wb in windows}

        head_boxes = boxes_by_cls[1]['boxes']
        head_scores = boxes_by_cls[1]['scores']
        head_boxes, head_scores = v4.dedup_heads_in_window(head_boxes, head_scores, windows)
        dets = []
        for i, hb in enumerate(head_boxes):
            x1, y1, x2, y2 = [int(v) for v in hb]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            feat = v4.extract_feat(crop, reid_model, transform, device)
            hc = v4.box_center(hb)
            w = next((wid for wid, wb in windows if v4.point_in_box(hc, wb)), None)
            dets.append(([x1, y1, x2, y2], head_scores[i], w, feat))
        frame_dets.append((fi, name, dets))
        print(f"  帧{fi} {name}: {len(dets)} 个检测")

    # ---- 分析：gid=3 帧2 在(429,672)，帧3 漏检(左)，帧4 在(869,695) ----
    # 这里用诊断已知的 bbox 作为参照。gid=3 在帧2 的框，特征在 frame_dets[2] 里。
    # 帧4 里"重现的左侧乘客"是哪个检测？诊断帧4 有 gid=2(869)、gid=3(906)、gid=4(811)。
    # 实际帧4 的左侧/中间/右侧头对应诊断的 gid=4(811)、gid=2(869)、gid=3(906)。
    # 但按识图，帧3 漏检的是后排"左侧/中间"那一位。
    print()
    print("===== 分析 =====")
    # 找到帧2 里 gid=3 对应的检测（帧2 窗1 左侧 = c~429）
    f2_dets = frame_dets[2][2]
    f3_dets = frame_dets[3][2] if len(frame_dets) > 3 else []
    f4_dets = frame_dets[4][2] if len(frame_dets) > 4 else []
    f5_dets = frame_dets[5][2] if len(frame_dets) > 5 else []

    print("\n帧2(5_original) 窗1 检测:")
    for b, c, w, f in f2_dets:
        if w == 1:
            print(f"  bbox={b} conf={c:.3f} c=({int((b[0]+b[2])/2)},{int((b[1]+b[3])/2)})")

    print("\n帧3(6_original) 窗1 检测:")
    for b, c, w, f in f3_dets:
        if w == 1:
            print(f"  bbox={b} conf={c:.3f} c=({int((b[0]+b[2])/2)},{int((b[1]+b[3])/2)})")

    print("\n帧4(7_original) 窗1 检测:")
    for b, c, w, f in f4_dets:
        if w == 1:
            print(f"  bbox={b} conf={c:.3f} c=({int((b[0]+b[2])/2)},{int((b[1]+b[3])/2)})")

    # gid=3 在帧2 的特征（窗1 左侧，c=429 那个）
    gid3_f2_feat = None
    gid3_f2_box = None
    for b, c, w, f in f2_dets:
        if w == 1:
            cx = (b[0] + b[2]) / 2
            if 400 < cx < 450:
                gid3_f2_feat = f
                gid3_f2_box = b
                break
    if gid3_f2_feat is None:
        print("\n[警告] 帧2 未找到 gid=3 的检测（c~429）")
        return

    print(f"\n帧2 gid=3 参照框: {gid3_f2_box}")

    # 模拟：帧2 后 gid=3 进待认领池（长期特征 = 帧2 特征），帧4 重现时做 Mutual Best Match
    # 待认领池里假设只有 gid=3 一个（后排），看帧4 的检测们谁最像它
    print("\n===== Mutual Best Match 验证 =====")
    print(f"待认领池特征 = 帧2 gid=3 (长期)")
    for b, c, w, f in f4_dets:
        if w == 1:
            sim = float(np.dot(gid3_f2_feat, f))
            cx = (b[0] + b[2]) / 2
            mark = " <== 最像" if sim > 0.8 else ""
            print(f"  帧4 检测 c=({int(cx)},{int((b[1]+b[3])/2)}) conf={c:.3f} sim={sim:.3f}{mark}")

    # 反向：该最像检测，是否也最像待认领池里的 gid=3（而非池里其他人）
    # 待认领池里再加一个"相邻乘客"的长期特征（帧2 gid=2，后排右侧）做对照
    gid2_f2_feat = None
    for b, c, w, f in f2_dets:
        if w == 1:
            cx = (b[0] + b[2]) / 2
            if 450 < cx < 500:
                gid2_f2_feat = f
                break
    if gid2_f2_feat is not None:
        print("\n对照：待认领池里同时有 gid=2(帧2 右侧) 和 gid=3(帧2 左侧)，帧4 各检测对两者的相似度:")
        for b, c, w, f in f4_dets:
            if w == 1:
                s3 = float(np.dot(gid3_f2_feat, f))
                s2 = float(np.dot(gid2_f2_feat, f))
                cx = (b[0] + b[2]) / 2
                pick = "gid=3" if s3 > s2 else "gid=2"
                print(f"  帧4 c=({int(cx)},{int((b[1]+b[3])/2)}) conf={c:.3f} sim_gid3={s3:.3f} sim_gid2={s2:.3f} -> 更亲{pick}")


if __name__ == "__main__":
    main()
