# NNUE 管线总览

**象眸 SF** 的 NNUE 用于 **拟合象棋小巫师搜索分**（行棋方视角），并集成到 C++ 搜索的静态评估中。

## 流程图

```
xqwl_gen_nnue          nnue_training/           nnue_qINT8/              cpp/
     │                      │                      │                    │
     ▼                      ▼                      ▼                    ▼
 FEN + vl 文本  ──►  浮点 PyTorch 训练  ──►  INT8 量化 + 校准  ──►  .xqnnue.bin
 (nnue_data/)         (checkpoints/)          (output/)          load_nnue()
```

## 各阶段文档

| 阶段 | 文档 |
|------|------|
| 1. 生成监督数据 | [NNUE 训练数据生成](nnue-data-generation.md) |
| 2. 浮点模型训练 | [NNUE 浮点训练](nnue-training.md) |
| 3. INT8 量化与部署 | [INT8 量化与 C++ 推理](nnue-int8-inference.md) |
| 架构参考调研 | [NNUE 架构调研（Pikafish）](nnue_pikafish_research.md) |

## 模型概要

| 项目 | 值 |
|------|-----|
| 特征 | XQWL-PSQ，1260 维稀疏（己方/对方 × 7 子力 × 90 格） |
| 结构 | FT（EmbeddingBag）→ L1=512 → FC 512→32 → FC 64→32 → FC 64→1 |
| 标签 | 搜索分 `vl`，z-score 归一化（mean/std 写入 checkpoint 与 bin） |
| 推理 | FT int32 累加器 + 增量更新；FC 纯 int8 点积 + per-layer input scale |

## 快速命令

```bash
# 1. 生成数据（示例：小规模）
./xqwl_gen_nnue --max-positions 10000 --jobs 4 --output-dir nnue_data

# 2. 训练（冒烟 / 正式 GPU）
cd nnue_training && python3 train.py --config configs/smoke.yaml
# 或：python3 train.py --config configs/full_gpu.yaml

# 3. 量化 + 导出
cd nnue_qINT8 && python3 quantize.py

# 4. 测试 C++ 与 Python 一致性
cd .. && PYTHONPATH=. python3 scripts/test_nnue_wsl.py
```

## 搜索中的行为

- 加载 NNUE 后，根节点 / QSearch 停着用 `evaluate_vl`（Full）；内部节点用 `evaluate_vl_raw`（Raw）
- **数据生成**（`xqwl_gen_nnue`）默认从 **`deployment/model/quantized.xqnnue.bin`** 加载（构建时自 `data/nnue_model/` 复制）
- 空着剪枝仍用 PST 子力（`NullOkay()`）
- **单视角 FT 累加器**：每步只更新走子后视角；`acc_valid[]` lazy refresh；评估只读 `acc[sdPlayer]`
- TT 缓存 `staticEval`，并驱动 razoring / futility / LMP 等剪枝

细节见 [引擎算法技术](engine-algorithms.md) 与 [INT8 量化与 C++ 推理](nnue-int8-inference.md)。
