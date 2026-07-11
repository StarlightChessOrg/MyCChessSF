# NNUE 浮点训练

目录：`nnue_training/`。使用 PyTorch 训练稀疏 NNUE，拟合 `xqwl_gen_nnue` 产生的搜索分。

## 依赖

```bash
pip install -r nnue_training/requirements.txt
```

## 数据

- 默认读取 `../nnue_data/worker_*.txt`（FEN `\t` vl）
- 配置见 `configs/smoke.yaml`（可改为完整训练配置）
- 标签 z-score：`target_norm = (vl - mean) / std`，mean/std 写入 checkpoint

## 模型

| 项目 | 默认 |
|------|------|
| 特征 | `features/xqwl_psq.py`，1260 维 |
| L1 / L2 / L3 | 512 / 32 / 32 |
| FT | `EmbeddingBag` + bias |
| 头 | FC + pairwise 变换 + sq/clipped ReLU 拼接 |

实现见 `model/nnue.py`、`model/activations.py`。

## 训练

```bash
cd nnue_training
python3 train.py --config configs/smoke.yaml
```

主要超参（`configs/smoke.yaml`）：

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `max_epochs` | 384 | 最大 epoch |
| `early_stopping_patience` | 32 | 验证集无提升则早停 |
| `batch_size` | 2048 | |
| `num_workers` | 4 | DataLoader 并行 |
| `val_workers` | [30, 31] | 验证集 worker 文件 id |

## 输出

- `checkpoints/best.pt` — 验证最优
- `checkpoints/last.pt` — 最后一轮
- checkpoint 内含 `label_mean`、`label_std`、`model_state_dict`、`config`

## 指标

训练日志含归一化空间 loss/MAE，以及还原到 vl 空间的 MAE、相关系数（辅助判断拟合搜索分的效果）。

## 下一步

→ [INT8 量化与 C++ 推理](nnue-int8-inference.md)（`quantize.py` 默认读取 `../nnue_training/checkpoints/best.pt`）
