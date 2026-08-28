"""批量运行 sideview_headshoulder_counter.py：用 run_head_seqsplit 新模型，跑 超员原图0729 前 12 个序列。

- 输入：F:/车辆超员检测项目数据集/超员原图0729/ 下按名排序前 12 个文件夹
- 输出：target tracking/tracking_output/日期_序号/<源序列名>/ （顶层 日期_序号 区分同一天多次实验，
        子目录直接用源序列文件夹名，与 超员原图0729 对应）
- 模型：主脚本已改为 run_head_seqsplit/weights/best.pt

用法：
    D:/Anaconda/envs/course_torch/python.exe run_tracking_batch.py
"""
import os
import subprocess
import sys
import re
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_PY = os.path.join(BASE_DIR, "sideview_headshoulder_counter.py")

DATA_ROOT = r"F:/车辆超员检测项目数据集/超员原图0729"
OUT_ROOT = os.path.join(BASE_DIR, "tracking_output")

# 按名排序取前 12 个文件夹
seqs = sorted([d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))])[:12]
print(f"[批量] 共 {len(seqs)} 个序列:")
for s in seqs:
    print(f"  {s}")

os.makedirs(OUT_ROOT, exist_ok=True)

# 确定本次运行的顶层目录：YYYYMMDD_序号（三位，当天序号自动递增）
today = datetime.now().strftime("%Y%m%d")
existing = [d for d in os.listdir(OUT_ROOT) if d.startswith(today + "_")]
next_seq = max([int(m.group(1)) for d in existing
                if (m := re.match(rf"^{today}_(\d+)$", d))], default=0) + 1
run_dir = os.path.join(OUT_ROOT, f"{today}_{next_seq:03d}")
os.makedirs(run_dir, exist_ok=True)
print(f"[输出顶层目录] {run_dir}")

for i, seq in enumerate(seqs, 1):
    in_dir = os.path.join(DATA_ROOT, seq)
    out_sub = os.path.join(run_dir, seq)  # 子目录直接用源序列名
    os.makedirs(out_sub, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"[{i}/{len(seqs)}] 序列 {seq} -> {out_sub}")
    print(f"{'='*50}")

    # 调用主脚本：python sideview_headshoulder_counter.py <输入目录> <输出目录>
    cmd = [sys.executable, COUNTER_PY, in_dir, out_sub]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    # 打印主脚本输出（过滤大量逐帧日志，保留关键信息）
    lines = (r.stdout or "").splitlines()
    for ln in lines:
        if ("START" in ln or "总人数" in ln or "已激活" in ln or "帧" in ln and "IDs" in ln) and len(lines) < 500:
            pass
    # 保留开头和结尾的关键输出
    print(f"[序列 {seq}] 输出日志片段:")
    if r.stdout:
        out_lines = r.stdout.splitlines()
        # 打印首尾关键行
        for ln in out_lines[:5]:
            print(f"  {ln}")
        print(f"  ... (共 {len(out_lines)} 行) ...")
        for ln in out_lines[-3:]:
            print(f"  {ln}")
    if r.returncode != 0:
        print(f"[序列 {seq}] 运行异常，stderr 前几行:")
        for ln in (r.stderr or "").splitlines()[:10]:
            print(f"  {ln}")

print(f"\n全部完成，结果输出到: {run_dir}")
