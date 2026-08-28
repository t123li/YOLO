"""用序列级划分后的 head_train_val_test 训练 1 类 head 检测模型（YOLOv8s）。

背景：head_v2 训练用的旧划分存在跨 split 数据泄漏（同车相邻帧被切散），
mAP50=96% / recall=0.901 虚高。本脚本用修复后的序列级划分重新训练，
拿到真实基线指标。

用法：
    python train.py

说明：
    - 模型从本地 yolov8s.pt 预训练权重起步（与 head_v2 一致）。
    - batch=160、epochs=100（用户指定）。
    - 输出 run 保存在本脚本所在目录下的 runs/<name>/。
    - 适用于云 GPU：数据、权重路径请改成云上的实际路径。
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ultralytics import YOLO

# 脚本所在目录（target detection），run 输出到这里
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR  # yolo 的 project 参数：输出到 <project>/<name>

# 预训练权重（云上 yolov8s.pt 实际路径；也可写 'yolov8s.pt' 让 ultralytics 自动下载）
PRETRAINED = r"/root/autodl-tmp/yolov8s.pt"

# 序列级划分后的数据集（yaml 内 train/val/test 指向新划分）
DATA = r"/autodl-fs/data/head_train_val_test/data.yaml"

# 训练超参
EPOCHS = 100
BATCH = 16
IMGSZ = 640
SEED = 0
NAME = "run_head_seqsplit"


def list_outputs(root):
    """列出 root 下所有文件（相对路径 + 大小），用于训练结束后展示新产物。"""
    if not os.path.isdir(root):
        return []
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, root)
            size = os.path.getsize(p)
            out.append((rel, size))
    return sorted(out)


def main():
    # 权重存在性检查（云上路径不对时给出明确报错）
    if not os.path.exists(PRETRAINED):
        raise FileNotFoundError(
            f"预训练权重不存在: {PRETRAINED}\n"
            "请改成云 GPU 上 yolov8s.pt 的实际路径，或改为 'yolov8s.pt' 自动下载。"
        )
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"数据集 yaml 不存在: {DATA}\n请确认已上传 head_train_val_test 到云上。")

    print(f"[train] project={PROJECT_DIR} name={NAME} data={DATA}")
    print(f"[train] epochs={EPOCHS} batch={BATCH} imgsz={IMGSZ}")

    model = YOLO(PRETRAINED)
    model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        project=PROJECT_DIR,
        name=NAME,
        exist_ok=True,
        seed=SEED,
        device=0,
        verbose=True,
    )

    # 训练结束后打印最佳权重路径与验证指标
    best = os.path.join(PROJECT_DIR, NAME, "weights", "best.pt")
    print("\n[完成] best.pt:", best)

    # 列出本次训练产物清单（方便知道新产生了哪些文件）
    out_root = os.path.join(PROJECT_DIR, NAME)
    print(f"\n[产物清单] {out_root}")
    for rel, size in list_outputs(out_root):
        kb = size / 1024
        unit = "KB" if kb < 1024 else "MB"
        val = kb if kb < 1024 else kb / 1024
        print(f"  {rel}  ({val:.1f} {unit})")

    if os.path.exists(best):
        m = YOLO(best)
        r = m.val(data=DATA, imgsz=IMGSZ, device=0)
        print(f"[val] mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} "
              f"precision={r.box.mp:.4f} recall={r.box.mr:.4f}")


if __name__ == "__main__":
    main()
