"""构建 3 类数据集（car / head / car_window）：合并 7 个文件夹 → 去重 → 序列级划分。

源数据：
    - public0705修2, public0629修正  （3 类标注，class 0=car, 1=head_shoulder, 2=car_window）
    - 原图5系列 5 个文件夹（3 类标注，同样 class 0/1/2）

去重策略（方案 A）：
    - 重名只发生在"原图5"系列内部（3051 张），public 系列与它们零重名。
    - 重名时保留优先级最高的文件夹的标签：
      原图5-已审完 > 原图5_剔除已标注-已标完部分 > -1 > -2 > -3
    - 图片相同但标签不同的重名图，只保留高优先级那份的标签（不混用）。

划分策略（关键）：
    - 按文件名前 17 位时间戳作为"车辆序列 ID"，同一序列的所有帧必须完整落在同一个 split，
      禁止跨 train/val/test（避免同车连续帧泄漏）。
    - 划分比例 RATIO=(0.8, 0.1, 0.1)，SEED=42。

输出：
    F:/车辆超员检测项目数据集/car_head_window_train_val_test/
        data.yaml  (nc=3, names: ['car','head','window'])
        images/{train,val,test}/...
        labels/{train,val,test}/...

用法：
    D:/Anaconda/envs/course_torch/python.exe <本脚本路径>
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

DATA_ROOT = r"F:/车辆超员检测项目数据集"
OUT_ROOT = os.path.join(DATA_ROOT, "car_head_window_train_val_test")

# 7 个源文件夹，按去重优先级排序（重名时靠前者优先保留）
SOURCE_DIRS = [
    "原图5-已审完",
    "原图5_剔除已标注-已标完部分",
    "原图5_剔除已标注-已标完部分-1",
    "原图5_剔除已标注-已标完部分-2",
    "原图5_剔除已标注-已标完部分-3",
    "public0705修2",
    "public0629修正",
]
# 优先级：下标越小优先级越高
PRIORITY = {d: i for i, d in enumerate(SOURCE_DIRS)}

RATIO = (0.8, 0.1, 0.1)
SEED = 42
SEQ_TS_LEN = 17

CLASS_NAMES = ["car", "head", "window"]


def seq_key(filename):
    """从文件名提取序列 ID：前 17 位数字时间戳。"""
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
    # 1) 收集所有 (文件名, 文件夹, 图路径, 标签路径)，按文件名去重（保留高优先级）
    #    同文件夹内也保证唯一
    best = {}  # 文件名 -> (priority, 文件夹, 图路径, 标签路径)
    total_scan = 0
    for d in SOURCE_DIRS:
        root = os.path.join(DATA_ROOT, d)
        if not os.path.isdir(root):
            print(f"[警告] 源目录不存在: {root}")
            continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    total_scan += 1
                    img_p = os.path.join(dp, fn)
                    txt_p = os.path.splitext(img_p)[0] + ".txt"
                    if not os.path.exists(txt_p):
                        continue
                    prio = PRIORITY[d]
                    if fn not in best or prio < best[fn][0]:
                        best[fn] = (prio, d, img_p, txt_p)

    print(f"扫描 {total_scan} 个文件（含重复），去重后唯一图 {len(best)} 张")

    # 统计去重来源分布
    src_counter = Counter(v[1] for v in best.values())
    print("去重后各文件夹贡献:")
    for d in SOURCE_DIRS:
        print(f"  {d}: {src_counter.get(d, 0)} 张")

    # 2) 按序列 ID 分组
    seq_groups = defaultdict(list)  # seq_id -> [(图路径, 标签路径)]
    ungrouped = 0
    for fn, (prio, d, img_p, txt_p) in best.items():
        sid = seq_key(fn)
        if sid is None:
            ungrouped += 1
            sid = f"__ungrouped_{ungrouped}__"
        seq_groups[sid].append((img_p, txt_p))
    print(f"序列总数: {len(seq_groups)}（无法提取序列ID: {ungrouped} 张）")

    # 3) 序列级划分
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

    # 帧数统计
    frame_count = Counter()
    for sid, frames in seq_groups.items():
        frame_count[seq_split[sid]] += len(frames)
    print(f"[划分] 序列: train {n_train} / val {n_val} / test {n_seq-n_train-n_val}")
    print(f"[划分] 帧数: train {frame_count['train']} / val {frame_count['val']} / test {frame_count['test']}")

    # 4) 复制图片 + 写标签（保留 3 类，不重映射）
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "labels", split), exist_ok=True)

    counter = 0
    for split in ["train", "val", "test"]:
        for sid in seq_ids:
            if seq_split[sid] != split:
                continue
            for img_p, txt_p in seq_groups[sid]:
                name = f"img_{counter:07d}"
                counter += 1
                ext = os.path.splitext(img_p)[1].lower()
                shutil.copy2(img_p, os.path.join(OUT_ROOT, "images", split, name + ext))
                shutil.copy2(txt_p, os.path.join(OUT_ROOT, "labels", split, name + ".txt"))

    # 5) 生成 data.yaml（3 类）
    yaml = (
        f"path: {OUT_ROOT}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 3\n"
        f"names: {CLASS_NAMES}\n"
    )
    with open(os.path.join(OUT_ROOT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    print("\n完成！")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {frame_count[split]} 帧 / {sum(1 for s in seq_split.values() if s==split)} 序列")
    print(f"输出目录: {OUT_ROOT}")


if __name__ == "__main__":
    main()
