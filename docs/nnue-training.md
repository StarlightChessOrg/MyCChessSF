# NNUE 浮点训练

目录：`nnue_training/`。使用 PyTorch 训练稀疏 NNUE，拟合 `xqwl_gen_nnue` 产生的搜索分。

## 依赖

```bash
pip install -r nnue_training/requirements.txt
```

依赖含 `torch`、`numpy`、`pyyaml`、`tqdm`。

## 数据

- 分块格式：`worker_*/chunk_*.txt`（FEN `\t` vl），见 [NNUE 训练数据生成](nnue-data-generation.md)
- 单文件格式：`merged.txt`（多批数据合并后的单行样本文件）
- 标签 z-score：`target_norm = (vl - mean) / std`，`mean`/`std` 由**训练集**统计并写入 checkpoint

### 验证集划分

| 方式 | 配置 | 适用场景 |
|------|------|----------|
| 按 worker 目录 | `val_workers: [30, 31]` | 分块 `worker_*/chunk_*.txt` |
| 随机比例 | `val_ratio: 0.01` + `val_seed: 42` | 单文件 `merged.txt` 等 |

## 配置文件

| 文件 | 用途 |
|------|------|
| `configs/smoke.yaml` | 小规模 / CPU 冒烟测试 |
| `configs/full_gpu.yaml` | 正式 GPU 训练（约千万行 `merged.txt`） |

### 冒烟测试

```bash
cd nnue_training
python train.py --config configs/smoke.yaml
```

### 正式 GPU 训练

```bash
cd nnue_training
python train.py --config configs/full_gpu.yaml
```

`full_gpu.yaml` 要点：

| 参数 | 值 | 说明 |
|------|-----|------|
| `data.source` | `../nnue_data` | 相对 `nnue_training/`，仓库内 `nnue_data/merged.txt`（须含 PST 列） |
| `data.pattern` | `merged.txt` | 单文件全量数据 |
| `data.val_ratio` | `0.01` | 随机 1% 作验证集 |
| `data.load_workers` | `auto` | 多进程加载；`auto` 取 `min(8, CPU 核数)` |
| `train.batch_size` | `16384` | GPU batch；显存不足可降至 `8192` |
| `train.precompute_features` | `true` | 启动时一次性算好 PSQ 特征 |
| `train.feature_workers` | `auto` | 特征预计算并行度 |
| `train.num_workers` | `auto` | 预计算后 DataLoader 默认 `0`（CSR 已在内存） |
| `train.tqdm` | `true` | epoch 内 tqdm 进度条 |
| `train.device` | `cuda` | GPU 训练 |
| `train.checkpoint_dir` | `checkpoints/full_gpu` | 与 smoke 分开存放 |

并行配置说明：

- `auto` / `0` / 负数：按 CPU 核数自动取值；**IO 类任务**（加载、特征预计算）上限 **8**，避免 Windows 下 spawn 过多子进程触发 PyTorch OOM
- 显式正整数：使用指定 worker 数（如 `load_workers: 12`）

## 模型

| 项目 | 默认 |
|------|------|
| 特征 | `features/xqwl_psq.py`，1260 维（XQWL-PSQ 动态 PST） |
| L1 / L2 / L3 | 512 / 32 / 32 |
| FT | `EmbeddingBag` + bias |
| 头 | FC + pairwise 变换 + sq/clipped ReLU 拼接 |

实现见 `model/nnue.py`、`model/activations.py`。

## 训练流程与日志

启动后大致经历以下阶段（均有 `[data]` / `[label]` / `[loader]` 日志）：

1. **多进程加载** — 读取 `merged.txt` 或分块文件，校验 FEN
2. **FEN 去重**（默认 `dedupe_fen: true`）— 按「棋盘 + 行棋方」合并重复局面，同一 FEN 多条 `vl` 取**平均值**
3. **划分 train/val** — 按 `val_ratio` 或 `val_workers`
4. **静态局面过滤**（默认 `quiet_only: true`）— 仅保留：非将杀区、行棋方未被将军、\|搜索分 − PST\| ≤ `quiet_pst_margin`（PST/in_check 来自数据第 3/4 列）
5. **将杀分重映射**（`quiet_only: false` 时可选 `mate_remap: true`）— 扫描 quiet 极值，\|vl\| ≥ 9800 改为 ±cap
6. **z-score 统计** — 对训练集计算 mean/std
7. **特征预计算**（可选）— 多进程生成 PSQ 稀疏索引
8. **CSR 打包** — 将特征压成紧凑数组
9. **训练循环** — 全部样本参与 loss；早停与 best checkpoint 看全量 `val_loss`

损失为 **50% MSE + 50% 排序损失**（默认权重，可配置）：

| 分量 | 空间 | 作用 |
|------|------|------|
| MSE | z-score 归一化 | 约束预测量级，拟合绝对搜索分 |
| 排序（RankNet logistic） | 同 z-score 空间 | batch 内随机 pairwise，要求预测分顺序与标签一致 |

配置项（`train` 段）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `mse_loss_weight` | `0.5` | MSE 权重 |
| `ranking_loss_weight` | `0.5` | 排序损失权重；设为 `0` 则退化为纯 MSE |
| `ranking_max_pairs` | `4096` | 每个 batch 最多采样的 pairwise 对数 |

### 静态局面过滤（quiet_only）

与 NNUE 文献一致：NNUE 拟合的是**静态评估**，训练样本应排除将军、将杀带，以及搜索分与 PST 差距过大的「非静态」局面。

| 配置 | 默认 | 说明 |
|------|------|------|
| `data.quiet_only` | `true` | 仅保留静态局面；关闭则保留全部样本 |
| `data.quiet_pst_margin` | `70` | 保留条件：\|vl − PST\| ≤ 此值（cp） |
| `data.mate_threshold` | `9800` | \|vl\| ≥ 此值视为将杀带并**丢弃**（非 remap） |

依赖：数据文件第 3/4 列为 PST 与 `in_check`（新生成的 `xqwl_gen_nnue` 已自带；旧 `FEN\tvl` 需先运行 `scripts/augment_nnue_data_pst.py`，见 [NNUE 训练数据生成](nnue-data-generation.md)）。Windows 训练无需安装 `xqwlight_core`。

`quiet_only: true` 时 **`mate_remap` 应设为 `false`**（配置已默认）；将杀样本已被丢弃，无需再 remap。

### 将杀分处理（quiet_only: false 时）

| 配置 | 默认 | 说明 |
|------|------|------|
| `data.dedupe_fen` | `true` | 训练前按 FEN 去重；重复 `vl` 取平均值 |
| `data.quiet_only` | `true` | 仅静态局面；见上节 |
| `data.quiet_pst_margin` | `70` | PST 与搜索分允许偏差（cp） |
| `data.mate_threshold` | `9800` | \|vl\| ≥ 此值视为将杀带 |
| `data.mate_remap` | `false` | `quiet_only: true` 时关闭；`false` 时可将将杀分钳到 quiet 极值 |
| `data.mate_exclude_from_zscore` | `false` | 设为 `true` 可回退：z-score 不含将杀（需 `mate_remap: false`） |
| `train.mate_exclude_from_loss` | `false` | 设为 `true` 可回退：loss 不含将杀（需 `mate_remap: false`） |

推理侧：C++ 搜索加载 NNUE 时使用纯 NNUE 静态分，不与 PST 混合；未加载 NNUE 时使用纯 PST。

示例日志片段：

```text
train 1/384: 100%|██████████| 659/659 [00:35<00:00, 18.50batch/s, loss=0.614805, samples=10,783,368]
val 1/384:   100%|██████████| 7/7 [00:00<00:00, 42.10batch/s]

── epoch 1/384 (35.9s) ──
  train_loss    0.614805
  val_loss      0.519952  ← best
  val_mae_norm  0.258649
  val_rmse_norm 0.721077
  val_mae_vl    576.94 (all)
  val_corr_vl   0.6913 (all)
  val_mae_quiet 42.15  corr 0.8123 (n=104,500)
  val_mate      skipped n=4,422 (|vl|>=9800)
  checkpoint    saved best.pt (prev best n/a)

train 2/384: 100%|██████████| 659/659 [00:49<00:00, 13.26batch/s]
...
```

epoch 汇总通过 `tqdm.write` 输出，不与进度条抢行。

## 训练指标

组合损失在 **z-score 归一化空间**计算；同时输出还原到原始搜索分 `vl` 的指标便于直观理解。

### Epoch 内（tqdm）

| 字段 | 含义 |
|------|------|
| `train N/M` | 当前 epoch / 总 epoch；进度条为 batch 进度 |
| `loss` | 截至当前 batch 的**训练集** running average 组合损失 | 
| `samples` | 当前 epoch 已处理的训练样本数 |

验证阶段 tqdm 显示 `val N/M`，不单独打印逐步 loss。

### Epoch 结束（每轮一行汇总）

| 指标 | 空间 | 含义 | 参考 |
|------|------|------|------|
| `train_loss` | 归一化 | 训练集组合损失 | 越小越好 |
| `val_loss` | 归一化 | 验证集组合损失；**早停与 best.pt 依据** | 越小越好 |
| `val_mse` | 归一化 | 验证集 MSE 分量（启用排序时打印） | 越小越好 |
| `val_rank` | 归一化 | 验证集排序损失分量（启用排序时打印） | 越小越好 |
| `val_mae_norm` | 归一化 | 验证集平均绝对误差 `\|pred_norm - target_norm\|` | 越小越好 |
| `val_rmse_norm` | 归一化 | 验证集 RMSE | 越小越好 |
| `val_mae_vl` | 原始 vl | 将预测还原为 `pred_vl = pred_norm * std + mean` 后的 MAE（单位：搜索分） | 越小越好；典型数百～数千 cp 视 std 而定 |
| `val_corr_vl` | 原始 vl | 预测分与标签搜索分的 Pearson 相关系数 | 越接近 1 越好；>0.9 通常表示拟合较好 |

还原公式（与 `train.py` / checkpoint 一致）：

```text
pred_vl = pred_norm * label_std + label_mean
```

其中 `label_mean`、`label_std` 来自训练集 z-score 统计，写入 `checkpoints/*.pt`。

### 早停

- `early_stopping_patience`（默认 32）：验证集 `val_loss` 连续 N 个 epoch 无提升则停止
- 每轮保存 `last.pt`；`val_loss` 创新低时额外保存 `best.pt`

## 主要超参

| 参数 | smoke | full_gpu | 说明 |
|------|-------|----------|------|
| `max_epochs` | 384 | 384 | 最大 epoch |
| `early_stopping_patience` | 32 | 32 | 验证集无提升则早停 |
| `batch_size` | 2048 | 16384 | |
| `lr` | 0.001 | 0.001 | Adam |
| `weight_decay` | 0.0 | 0.0 | |
| `mse_loss_weight` | 0.5 | 0.5 | MSE 分量权重 |
| `ranking_loss_weight` | 0.5 | 0.5 | 排序损失权重 |
| `ranking_max_pairs` | 4096 | 8192 | 每 batch pairwise 采样上限 |
| `num_workers` | 4 | auto | DataLoader；预计算大数据集建议 0 |
| `prefetch_factor` | 2 | 4 | `num_workers > 0` 时生效 |

## 输出

- `checkpoints/best.pt` — 验证 `val_loss` 最优
- `checkpoints/last.pt` — 最后一轮
- checkpoint 内含 `label_mean`、`label_std`、`model_state_dict`、`config`

正式训练权重路径示例：`checkpoints/full_gpu/best.pt`。量化时需在 `nnue_qINT8/configs/default.yaml` 中指向对应 checkpoint。

## 下一步

→ [INT8 量化与 C++ 推理](nnue-int8-inference.md)（`quantize.py` 默认读取 `../nnue_training/checkpoints/best.pt`）
