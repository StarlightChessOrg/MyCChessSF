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
| `data.source` | `../../nnue_data` | 相对 `nnue_training/`，指向 workspace 根目录数据 |
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
2. **FEN 去重**（默认 `dedupe_fen: true`）— 按「棋盘 + 行棋方」合并重复局面，同一 FEN 多条 `vl` 取**中位数**
3. **划分 train/val** — 按 `val_ratio` 或 `val_workers`
4. **将杀分重映射**（默认 `mate_remap: true`）— 扫描全数据集 quiet 样本的 min/max，\|vl\| ≥ 9800 的样本改为 **±max(\|min\|, \|max\|)**，保留在训练集中
5. **z-score 统计** — 对重映射后的**全部**样本计算 mean/std
6. **特征预计算**（可选）— 多进程生成 PSQ 稀疏索引
7. **CSR 打包** — 将特征压成紧凑数组
8. **训练循环** — 全部样本参与 loss；早停与 best checkpoint 看全量 `val_loss`

### 将杀分处理

| 配置 | 默认 | 说明 |
|------|------|------|
| `data.dedupe_fen` | `true` | 训练前按 FEN 去重；重复 `vl` 取中位数 |
| `data.mate_threshold` | `9800` | \|vl\| ≥ 此值视为将杀带（与 C++ `WIN_VALUE` 一致） |
| `data.mate_remap` | `true` | 将杀分钳到 quiet 极值的最大绝对值，不丢弃样本 |
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

损失函数为 **MSE**，在 **z-score 归一化空间**计算；同时输出还原到原始搜索分 `vl` 的指标便于直观理解。

### Epoch 内（tqdm）

| 字段 | 含义 |
|------|------|
| `train N/M` | 当前 epoch / 总 epoch；进度条为 batch 进度 |
| `loss` | 截至当前 batch 的**训练集** running average MSE（归一化空间） |
| `samples` | 当前 epoch 已处理的训练样本数 |

验证阶段 tqdm 显示 `val N/M`，不单独打印逐步 loss。

### Epoch 结束（每轮一行汇总）

| 指标 | 空间 | 含义 | 参考 |
|------|------|------|------|
| `train_loss` | 归一化 | 训练集 MSE：`(pred_norm - target_norm)²` 均值 | 越小越好；应持续下降 |
| `val_loss` | 归一化 | 验证集 MSE；**早停与 best.pt 依据** | 越小越好 |
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
| `num_workers` | 4 | auto | DataLoader；预计算大数据集建议 0 |
| `prefetch_factor` | 2 | 4 | `num_workers > 0` 时生效 |

## 输出

- `checkpoints/best.pt` — 验证 `val_loss` 最优
- `checkpoints/last.pt` — 最后一轮
- checkpoint 内含 `label_mean`、`label_std`、`model_state_dict`、`config`

正式训练权重路径示例：`checkpoints/full_gpu/best.pt`。量化时需在 `nnue_qINT8/configs/default.yaml` 中指向对应 checkpoint。

## 下一步

→ [INT8 量化与 C++ 推理](nnue-int8-inference.md)（`quantize.py` 默认读取 `../nnue_training/checkpoints/best.pt`）
