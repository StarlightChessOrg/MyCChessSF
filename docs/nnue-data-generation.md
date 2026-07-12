# NNUE 训练数据生成

`xqwl_gen_nnue` 批量生成 **FEN + 搜索评分** 监督数据，供 [NNUE 浮点训练](nnue-training.md) 使用。

## 行为

- 自对弈采样：每个局面写一行，走子后继续；终局后 `Startup()` 重新开局
- 走子策略：**20% 随机合法着** / **80% 引擎搜索结果**（可调）
- **不使用开局库**
- **评估函数**：默认自动加载 **`deployment/model/quantized.xqnnue.bin`**（`build_linux.sh` 从 `data/nnue_model/` 复制而来）；开发树中亦回退检测 `data/nnue_model/`。可用 `--nnue PATH` 指定，或 `--pst-only` 强制 PST
- `vl` 为**当前行棋方视角**的根节点搜索分（与 `search_best_detail()['score']` 一致）
- 默认主搜索最大深度 **6 层**（`--max-depth`），与 `--think-ms` 共同决定标签质量与生成速度
- FEN 在 C++ 内联编码（`cpp/inc/xqwl_fen.inc`）
- 多进程：`fork()` 启动 worker，默认进程数 = CPU 核数
- 每个 worker 独立子目录，每 **chunk-size** 局面一个分块文件（默认 2000）：`{output_dir}/worker_{id}/chunk_{n}.txt`

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
| `--chunk-size` | `2000` | 每个分块文件的局面数 |
| `--nnue` | （自动检测） | 优先 `deployment/model/`，其次 `data/nnue_model/`（见 [data/nnue_model/README.md](../data/nnue_model/README.md)） |
| `--pst-only` | — | 强制使用 PST，忽略 NNUE 权重 |

## 输出格式

数据目录默认 **`nnue_data/`**（仓库根下，gitignore）。常见布局：

- 单文件：`merged.txt`（多批合并）
- 分块：`worker_{id}/chunk_{n}.txt`（`xqwl_gen_nnue` 默认）

每行一条样本，**制表符分隔**，标准 **4 列**：

```text
FEN\t搜索分\tPST\tin_check
```

| 列 | 名称 | 说明 |
|----|------|------|
| 1 | FEN | 局面（含行棋方 `w` / `b`；红在 FEN 底部，大写为红） |
| 2 | 搜索分 | 根节点搜索标签（行棋方视角；NNUE 拟合目标；与 `search_best_detail()['score']` 一致） |
| 3 | PST | 同一局面的 PST 静态分（行棋方视角；`Position.evaluate()`） |
| 4 | in_check | 行棋方是否被将军：`0` 否，`1` 是 |

示例：

```text
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1	8	3	0
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C4NC1/9/RNBAKAB1R b - - 0 1	-2	-6	0
```

训练默认 [`quiet_only: true`](nnue-training.md#静态局面过滤quiet_only)：用第 3/4 列过滤非静态局面（非将杀、未将军、\|搜索分 − PST\| ≤ `quiet_pst_margin`）。

旧版两列 `FEN\tvl` 仍可被加载器读取，但 **`quiet_only: true` 时须先补全第 3/4 列**（见下节）。

### 为旧数据补充 PST（WSL）

若已有仅含 `FEN\tvl` 的文件，在 WSL 中：

```bash
bash scripts/build_linux.sh   # 若尚未编译 xqwlight_core
bash scripts/augment_merged_wsl.sh /path/to/legacy_merged.txt nnue_data/merged.txt
```

也可直接调用 Python：

```bash
python3 scripts/augment_nnue_data_pst.py \
  --input /path/to/legacy_merged.txt \
  --output nnue_data/merged.txt \
  --workers auto
```

## 磁盘与进度

- 1000 万行未压缩约 **0.8–1.0 GB**（均行 ~80 字节）
- 24 worker、默认 chunk-size 2000 时约 **5000 个分块文件**（每块 ~160 KB）
- stderr 每 10000 局面打印进度；行缓冲每行自动刷盘，可配合 `wc -l` 实时查看：

```bash
wc -l nnue_data/worker_*/chunk_*.txt
du -sh nnue_data/
head nnue_data/worker_0/chunk_0.txt
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

输出采用**行缓冲**（`_IOLBF`），每写入一行 `FEN\tscore\n` 即刷盘，便于分块文件在崩溃后仍可保留已写数据。

目录结构：每个 worker 一个子文件夹（`worker_0/`、`worker_1/` …），其内按 `--chunk-size`（默认 2000）写入 `chunk_0.txt`、`chunk_1.txt` … 训练侧默认 glob 为 `worker_*/chunk_*.txt`。

- `--think-ms` 越大，标签噪声越小、生成越慢
- 提高 `--random-pct` 可增加局面多样性，但标签方差也会增大

## 下一步

→ [NNUE 浮点训练](nnue-training.md)
