"""训练 3 类检测模型（car / head / window），YOLOv8s，序列级划分数据集。

数据集：F:/车辆超员检测项目数据集/car_head_window_train_val_test
    - 由 build_3class_dataset.py 构建：7 文件夹合并 + 去重 + 序列级划分
    - nc=3，names=['car','head','window']

预训练：本地 yolov8s.pt（COCO 预训练，含 car 类先验）
    - 若本地权重不存在，改为 'yolov8s.pt' 自动下载

用途：训练好的模型同时检测 车辆/人头/车窗，为跟踪提供"车窗约束"（人头 ID 绑定车窗槽）。

用法：
    python train_3class.py
    （云上运行时把 PRETRAINED / DATA 改为云上实际路径）
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
PROJECT_DIR = r"/root/autodl-tmp"  # 云上产物写到 /root/autodl-tmp/run_3class/

# 预训练权重：云上 /root/autodl-tmp/yolov8s.pt（COCO 预训练）
PRETRAINED = r"/root/autodl-tmp/yolov8s.pt"
if not os.path.exists(PRETRAINED):
    PRETRAINED = "yolov8s.pt"  # 自动下载

# 3 类数据集（序列级划分，nc=3），云上解压到 /autodl-fs/data
DATA = r"/autodl-fs/data/car_head_window_train_val_test/data.yaml"

# 训练超参（batch=32 适配 4090 24G；yolov8s @ 640 约用 14-16G）
EPOCHS = 100
BATCH = 32
IMGSZ = 640
SEED = 0
NAME = "run_3class"


def list_outputs(root):
    """列出 root 下所有文件（相对路径 + 大小）。"""
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
    if not os.path.exists(PRETRAINED) and not PRETRAINED.endswith(".pt"):
        pass
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"3类数据集 yaml 不存在: {DATA}\n请先运行 build_3class_dataset.py。")

    print(f"[train] project={PROJECT_DIR} name={NAME} data={DATA}")
    print(f"[train] epochs={EPOCHS} batch={BATCH} imgsz={IMGSZ} pretrained={PRETRAINED}")

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

    print(f"\n[产物清单] {os.path.join(PROJECT_DIR, NAME)}")
    for rel, size in list_outputs(os.path.join(PROJECT_DIR, NAME)):
        kb = size / 1024
        unit = "KB" if kb < 1024 else "MB"
        val = kb if kb < 1024 else kb / 1024
        print(f"  {rel}  ({val:.1f} {unit})")

    if os.path.exists(best):
        m = YOLO(best)
        r = m.val(data=DATA, imgsz=IMGSZ, device=0)
        print(f"[val] mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} "
              f"precision={r.box.mp:.4f} recall={r.box.mr:.4f}")
        # 每类单独指标
        for i, cname in enumerate(["car", "head", "window"]):
            ap50 = r.box.ap50[i] if hasattr(r.box, "ap50") else None
            if ap50 is not None:
                print(f"[val] {cname} AP50={ap50:.4f}")

    # 提示下载命令（云上训练后把产物拉回本地）
    print("\n" + "=" * 60)
    print("[下载提示] 训练产物在云端，拉回本地命令：")
    print(f"  scp -P <端口> -r root@<AutoDL地址>:{os.path.join(PROJECT_DIR, NAME)}/ \\")
    print('  "D:/桌面文件/文档素材/东南大学/1.学术相关/车辆超员检测项目/YOLO/target detection/"')
    print("=" * 60)


if __name__ == "__main__":
    main()
