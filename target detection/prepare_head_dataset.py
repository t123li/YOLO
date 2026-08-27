"""从 3 类数据集提取 head_shoulder(class 1) → 重映射 class 0 → 合并成纯 1 类 head 数据集。

划分策略（2026-08-26 修改）：
    原版按“图像级随机 shuffle”划分 train/val/test，会把同一辆车的连续帧切进不同 split，
    造成跨集合泄漏（验证集与训练集几乎相同），使 mAP/recall 虚高。
    现改为【序列级划分】：以文件名前 17 位时间戳作为“车辆序列 ID”，
    同一序列的所有帧必须完整落在同一个 split，组内不可拆分。

    注意：不把“序号段”（_N / _N_original）加入分组键。public0705修2 / public0629修正
    用“相同17位时间戳 + 帧序号”表示同一车连续帧（如 20260705001844620_4_original.jpg）；
    原图5-已审完 则是每帧唯一时间戳（如 20260402145036000_1.jpg），其序号只是文件序号，
    不是车序列。因此分组键只能是 17 位时间戳前缀。

用法：
    D:/Anaconda/envs/course_torch/python.exe F:/vehicle_dataset/prepare_head_dataset.py

输出：
    F:/车辆超员检测项目数据集/head_train_val_test/
        data.yaml
        images/{train,val,test}/...
        labels/{train,val,test}/...
"""
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 保守方案源目录（3 类，已修正/已审完，质量较高）
SOURCE_DIRS = [
    r"F:/车辆超员检测项目数据集/public0705修2",
    r"F:/车辆超员检测项目数据集/public0629修正",
    r"F:/车辆超员检测项目数据集/原图5-已审完",
]

OUT_ROOT = r"F:/车辆超员检测项目数据集/head_train_val_test"

# 划分比例 train : val : test
RATIO = (0.8, 0.1, 0.1)
SEED = 42

HEAD_CLS = 1  # 源数据里 head_shoulder 的 class id

SEQ_TS_LEN = 17  # 文件名前 17 位数字 = 时间戳 = 车辆序列 ID


def seq_key(filename):
    """从文件名提取序列 ID：前 17 位数字时间戳。取不到则返回 None（无法分组）。"""
    prefix = ""
    for ch in filename:
        if ch.isdigit():
            prefix += ch
            if len(prefix) == SEQ_TS_LEN:
                return prefix
        elif prefix:
            break
    return prefix if len(prefix) == SEQ_TS_LEN else None


def main():
    # 1) 收集所有 图+标签 对，并按序列 ID 分组
    pairs = []  # (img_path, txt_path)
    for src in SOURCE_DIRS:
        for root, _, files in os.walk(src):
            jpgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for j in jpgs:
                img = os.path.join(root, j)
                txt = os.path.splitext(img)[0] + ".txt"
                if os.path.exists(txt):
                    pairs.append((img, txt))

    print(f"扫描到 {len(pairs)} 个 图+标签 对")

    # 2) 提取 head：只留 class 1，重映射成 0；没 head 的图丢弃
    valid = []  # (seq_id, img_path, [label_lines])
    ungrouped = 0
    for img, txt in pairs:
        kept = []
        with open(txt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(float(parts[0]))
                if cls == HEAD_CLS:
                    parts[0] = "0"  # 重映射 class 1 -> 0
                    kept.append(" ".join(parts))
        if kept:  # 至少有一个 head 才保留
            sid = seq_key(os.path.basename(img))
            if sid is None:
                ungrouped += 1
                sid = f"__ungrouped_{ungrouped}__"
            valid.append((sid, img, kept))

    print(f"提取 head 后保留 {len(valid)} 张图（无法提取序列ID: {ungrouped} 张）")

    # 3) 序列级划分：按序列 ID 分组，组整体进同一个 split，组内不可拆
    seq_groups = defaultdict(list)  # seq_id -> [(img, labels)]
    for sid, img, kept in valid:
        seq_groups[sid].append((img, kept))

    seq_ids = list(seq_groups.keys())
    random.Random(SEED).shuffle(seq_ids)

    n_seq = len(seq_ids)
    n_train = int(n_seq * RATIO[0])
    n_val = int(n_seq * RATIO[1])
    seq_split = {}
    for i, sid in enumerate(seq_ids):
        if i < n_train:
            seq_split[sid] = "train"
        elif i < n_train + n_val:
            seq_split[sid] = "val"
        else:
            seq_split[sid] = "test"

    # 打印一个抽样，确认同序列的帧确实被分到同一 split
    print("\n[验证] 抽查 8 个含≥2帧的序列，确认其所有帧落在同一 split:")
    shown = 0
    for sid, frames in seq_groups.items():
        if len(frames) >= 2:
            sp = seq_split[sid]
            print(f"  seq {sid}: {len(frames)}帧 -> {sp}")
            shown += 1
            if shown >= 8:
                break

    # 按 split 汇总图片数（帧级统计）
    frame_count = Counter()
    for sid, frames in seq_groups.items():
        frame_count[seq_split[sid]] += len(frames)
    print(f"\n[统计] 序列数: {n_seq} -> train {n_train} / val {n_val} / test {n_seq-n_train-n_val}")
    print(f"[统计] 帧数: train {frame_count['train']} / val {frame_count['val']} / test {frame_count['test']}")

    # 4) 复制图片 + 写标签（保留原扩展名）
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

    counter = 0
    for split in ["train", "val", "test"]:
        for sid in seq_ids:
            if seq_split[sid] != split:
                continue
            for img, lines in seq_groups[sid]:
                name = f"img_{counter:07d}"
                counter += 1
                ext = os.path.splitext(img)[1].lower()
                shutil.copy2(img, os.path.join(OUT_ROOT, "images", split, name + ext))
                with open(os.path.join(OUT_ROOT, "labels", split, name + ".txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")

    # 5) 生成 data.yaml
    yaml = (
        f"path: {OUT_ROOT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['head']\n"
    )
    with open(os.path.join(OUT_ROOT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    print("\n完成！")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {frame_count[split]} 帧 / {sum(1 for s in seq_split.values() if s==split)} 序列")
    print(f"输出目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
