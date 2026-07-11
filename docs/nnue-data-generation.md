# NNUE 训练数据生成

`xqwl_gen_nnue` 批量生成 **FEN + 搜索评分** 监督数据，供 [NNUE 浮点训练](nnue-training.md) 使用。

## 行为

- 自对弈采样：每个局面写一行，走子后继续；终局后 `Startup()` 重新开局
- 走子策略：**20% 随机合法着** / **80% 引擎搜索结果**（可调）
- **不使用开局库**
- **评估函数**：若存在 NNUE 权重（`nnue_qINT8/output/quantized.xqnnue.bin` 或通过 `--nnue` 指定），搜索使用 NNUE；否则使用 PST。可用 `--pst-only` 强制 PST
- `vl` 为**当前行棋方视角**的根节点搜索分（与 `search_best_detail()['score']` 一致）
- 默认主搜索最大深度 **6 层**（`--max-depth`），与 `--think-ms` 共同决定标签质量与生成速度
- FEN 在 C++ 内联编码（`cpp/inc/xqwl_fen.inc`）
- 多进程：`fork()` 启动 worker，默认进程数 = CPU 核数
- 每 worker 独立文件：`{output_dir}/worker_{id}.txt`

## 用法

```bash
# 默认：1000 万局面，CPU 核数个进程，输出到 nnue_data/
./xqwl_gen_nnue

# 自定义
./xqwl_gen_nnue \
  --output-dir ./nnue_data \
  --max-positions 10000000 \
  --think-ms 100 \
  --max-depth 6 \
  --jobs 24 \
  --random-pct 20
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--output-dir` | `nnue_data` | 输出目录（自动创建） |
| `--max-positions` | `10000000` | 局面总数上限 |
| `--think-ms` | `100` | 每局面搜索时限（毫秒） |
| `--max-depth` | `6` | 迭代加深主搜索最大层数（1–64） |
| `--jobs` | `0`（= CPU 核数） | 并行 worker 数 |
| `--random-pct` | `20` | 随机走子概率（0–100） |
| `--nnue` | （自动检测） | NNUE 权重路径（`.xqnnue.bin`）；未找到则回退 PST |
| `--pst-only` | — | 强制使用 PST，忽略 NNUE 权重 |

## 输出格式

每行一条样本，制表符分隔：

```text
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1	7
```

- **左侧**：标准象棋 FEN（红方在 FEN 底部，大写为红）
- **右侧**：整数搜索分（行棋方视角；开局约 ±1~±15，优势局面 ±100+，将杀级接近 ±9800）

## 磁盘与进度

- 1000 万行未压缩约 **0.8–1.0 GB**（均行 ~80 字节）
- 24 worker 时每个文件约 **30–40 MB**（按配额均分）
- stderr 每 10000 局面打印进度；输出文件采用全缓冲，进度点刷盘，可配合 `wc -l` 实时查看：

```bash
wc -l nnue_data/worker_*.txt
du -sh nnue_data/
head nnue_data/worker_0.txt
```

## 小规模验证

```bash
./xqwl_gen_nnue --max-positions 1000 --jobs 2 --output-dir nnue_test
```

## 数据质量提示

- 训练加载时会校验 FEN 并跳过坏行（`dataset.py` 统计 `skipped`）
- 生成端（`xqwl_gen_nnue`）写入前也会校验棋盘/FEN，异常局面丢弃并重新开局
- 可用审计脚本扫描已有数据：

```bash
python scripts/audit_nnue_data.py nnue_data
```

### 已知历史问题（约 0.1% 坏行）

早期 `xqwl_gen_nnue` 将约 **16 MiB** 搜索哈希表放在**线程栈**上（默认栈仅 ~8 MiB），worker 运行时发生**栈溢出**，导致：

| 坏行类型 | 表现 | 数量（17.4 万行样本） |
|----------|------|------------------------|
| 棋盘损坏 | FEN 出现 10 行以上（两段棋局拼在一起） | ~131 |
| 行粘连 | 同一物理行含两条 `FEN\tscore`（上一条缺换行） | ~20 |
| 其他非法 FEN | 棋子/格式无法解析 | ~16 |

已在 commit `3df8cef` 将哈希表改为**堆分配**；写入前校验。若 `skipped > 0`，建议**重新编译后整库重生成**，不要追加到旧文件。

### 性能说明

早期实现使用行缓冲（`_IOLBF`）并对每行 `fflush`，在 WSL/DrvFS（`/mnt/c`）上会随着输出文件变大而**越跑越慢**。现已改为 **256 KiB 全缓冲**，仅在进度汇报时刷盘。

- `--think-ms` 越大，标签噪声越小、生成越慢
- 提高 `--random-pct` 可增加局面多样性，但标签方差也会增大

## 下一步

→ [NNUE 浮点训练](nnue-training.md)
