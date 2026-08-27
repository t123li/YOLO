# Development records

_项目根目录变更记录。每次变更按 `YYYY_MM_DD_序号` 新增一个条目；同一天的序号从 `001` 递增。_

---

## 2026_08_25_001

### 项目初始三次 commit（git 历史递进关系）

> 项目最初由三个 commit 建立，本文档补记其递进关系，便于追溯。

| commit | 内容 | 递进关系 |
| --- | --- | --- |
| `dd3e3c6` Initial commit: YOLO project and sideview baseline | 整个 ultralytics 仓库骨架 + 侧视头肩检测基线 | 项目起点：Ultralytics YOLO 仓库 + 侧视头肩计数的初始版本 |
| `4c2bcea` Add sideview head-shoulder tracking baseline 001 | `sideview_headshoulder_counter.py` 大幅改动（+108/−58） | 在初始基线上完成 head-shoulder 跟踪基线 001：本地化、中文路径兼容、同帧 ID 修复 |
| `caebb62` Add sideview tracking version 002 and disable Dependabot | 再次改动跟踪脚本（+148/−61）+ 移除 dependabot | 跟踪改进版 002：运动补偿、跨帧关联、批量运行增强 |

**递进关系总结：** 服务器原始代码（000）→ 本地化基线（001）→ 当前跟踪改进版（002）。三次 commit 对应 `sideview_headshoulder_counter_改动记录.md` 中 001/002 两个版本迭代，均由 `t123li` 于 2026-08-25 提交。

---

## 2026_08_26_001

### `target detection/文献调研_车辆超员检测目标检测涨点_20260826/`

**作用：** 车辆超员场景目标检测“涨点”文献调研资料。目录中的 `PDFs/`、`sources/` 和 `figures/` 保存论文、来源和图表；
`文献综述.md`、`论文证据表.csv`、`检索记录.md`、`目标检测涨点计划.md` 用于沉淀证据、检索过程和后续检测改进方案。

**使用方法：**

1. 先阅读 `README.md` 和 `目标检测涨点计划.md`，了解建议的改进方向和验证顺序
2. 需要查论文依据时，在 `论文证据表.csv` 中按方法、数据集或结论筛选，再打开 `PDFs/` 中对应文献
3. 有新的论文、实验结论或可复现来源时，同步更新 `检索记录.md`、`论文证据表.csv` 和 `references.bib`
4. 只有在同一验证集上确认检测召回或 AP 改善后，才将方法写入正式训练/实验计划

**注意：** 此目录用于研究与决策记录，不应直接替代训练数据、模型权重或运行结果目录。

---

## 2026_08_26_002

### `target tracking/check_data_leak.py`

**作用：** 验证 `head_train_val_test` 数据划分是否存在跨 split 泄漏（车辆连续帧被随机切散）。

**背景：** `prepare_head_dataset.py` 用 `random.shuffle` 做图像级随机划分，而源数据是"车辆连续帧序列"（文件名 17 位数字前缀为序列 ID，同一车 2-6 帧）。同车相邻帧被切进 train/val/test 不同集合，会使 `head_v2` 的 mAP/recall 虚高。

**运行结果（2026-08-26，全量 10933 个源序列）：**

- 被切进 1 个 split 的序列：8281（75.7%）
- 被切进 2 个 split 的序列：1955（17.9%）
- 被切进 3 个 split 的序列：225（2.1%）
- **泄漏序列合计（≥2 split）：2180（19.9%）**，其中 1213 个序列的部分帧进了 val、1245 个进了 test
- 像素级完全重复帧极少（train 0.01% / test 0.10%）——泄漏形态是"同车相邻帧被切散"，不是同图复制

**结论：** `mAP50=96%`、`recall=0.901` 因泄漏而虚高，不可作为真实检测水平依据。需改为按源序列前缀分组（GroupKFold / 手动按前缀切）重新划分后重训或重估。

**运行方式：** `python target tracking/check_data_leak.py`（需 F: 盘数据可访问）。

---

## 2026_08_26_003

### `target tracking/sideview_headshoulder_counter.py` 的跟踪改进计划报告

**作用：** 桌面新生成《车辆超员场景_车内人员ID匹配改进方案_20260826.md》为跟踪改进总方案，包含数据泄漏修复（阶段 0，最高优先级）、检测与输入修正、替换为仓库现成 ByteTrack/FASTTracker、车辆局部坐标 + 座位槽硬约束、计数语义修正与回归验证。旧报告《车辆超员场景_车内人员跟踪问题分析与代码改进计划.md》根因清单经逐行核对全部成立，无 worktree 内容需删除；`.claude/settings.local.json` 中残留的 `.worktrees/tracking-baseline` 权限记录已清理。

**下一步：** 按序列前缀重新划分数据集（阶段 0）→ 用新划分重训/重估 head_v2 → 再进入跟踪端改进。

---

## 2026_08_26_004

### 序列级数据划分修复（prepare_head_dataset.py 重写）

**背景：** `prepare_head_dataset.py` 原用 `random.shuffle` 图像级随机划分，导致 19.9% 的车辆序列（2180/10933）被切进 ≥2 个 split，`head_v2` 的 `mAP50=96%`、`recall=0.901` 虚高（同车相邻帧跨集合泄漏）。

**改动：** 重写 `target tracking/runs/prepare_head_dataset.py`，改为**序列级划分**——以文件名前 17 位时间戳为序列 ID，同序列所有帧整体落入同一 split。**不使用序号段**（`public0705修2`/`public0629修正` 用"相同时间戳+帧序号"表示连续帧；`原图5-已审完` 是"每帧唯一时间戳"，序号只是文件序号）。

**分组键量化结论（探查脚本验证）：** 17 位时间戳前缀为唯一正确键（10928 组，42.5% 含≥2帧）；加序号段会把连续帧打散（20338 组，几乎全单帧）——错误。

**执行：** 旧划分备份到 `F:/车辆超员检测项目数据集/head_train_val_test_backup_20260826/` 后，清空并重新生成 `head_train_val_test`。

**修复后全量复核（check_data_leak.py）：** 泄漏序列 0.0%（修复前 19.9%）；train/val/test 像素级重复全 0.00%。新规模：train 15974 帧/8368 序列、val 1955 帧/1046 序列、test 1963 帧/1047 序列。

**注意：** 新划分下 head_v2 需重新训练，新的 mAP/recall 预计明显低于 96%，那才是真实检测水平。

---

## 2026_08_27_001

### 序列级划分重训 + 与师兄模型检测性能对比

**背景：** 数据泄漏修复后，用新划分重新训练 head 检测模型，拿到无泄漏污染的真实基线；并与师兄模型的检测结果对比。

**重训（云 GPU，AutoDL）：**
- 训练脚本：`target detection/train.py`（新写，`batch=16`、`epochs=100`、`imgsz=640`、`seed=0`、从本地 `yolov8s.pt` 起步，与旧 head_v2 同口径）。
- 产物：`target detection/run_head_seqsplit/`（含 `weights/best.pt`、`results.csv`）。

**重训结果（序列级划分，100 轮）：**

| 指标 | 旧划分 head_v2 | 新划分 run_head_seqsplit |
| --- | --- | --- |
| mAP50 | 0.9535 | **0.9599** |
| mAP50-95 | 0.6793 | **0.6857** |
| recall | 0.9012 | **0.9073** |
| precision | 0.9321 | **0.9367** |

**结论：** 去掉泄漏后指标不降反升，说明该数据集上 head 检测本就强，泄漏对 mAP 影响小；检测基线可信，问题更聚焦到跟踪端。

**检测性能对比（新模型 vs 师兄模型）：**
- 脚本：`target detection/detect_vs_annotated_2model.py`（新写）：左图用 `annotated_images/20260714151020365`（师兄模型 3 类框，红=head），右图用新模型 `run_head_seqsplit/best.pt` 检测 head（绿框，conf≥0.25）。
- 产物：`target detection/detect_vs_annotated_2model/`（vs_f00~07.jpg）。

**对比结论（抽样帧 0/3，肉眼定性）：** 新模型召回更全——帧0 师兄 2 个 head vs 新模型 4 个；帧3 师兄 4 个 vs 新模型 5 个。师兄模型漏检的前排乘员/前挡风处头部，新模型均能检出；已检出目标两边位置基本一致。**新模型 head 检测优于师兄模型。**

**注意：** 该对比为单序列、conf=0.25 的定性对比（肉眼看图），非定量指标。
