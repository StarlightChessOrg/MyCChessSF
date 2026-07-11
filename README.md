# MyCChessSF

在浏览器里和 **象棋小巫师（XQWL 0.6）** 对弈：规则核 + Alpha-Beta 搜索来自 `XQWL06.CPP`，**网页棋盘/棋子/音效复用 Win32 原版资源**。

**目标平台：Linux**（编译 `xqwlight_core.so` 与 `xqwl_gen_nnue`）。

## 与 MyCChessRL 的区别

| 项目 | 规则 | AI |
|------|------|-----|
| MyCChessRL | XQWL 局面核 | PyTorch / MCTS |
| **MyCChessSF** | 同上 | **原版小巫师搜索 + 开局库** |

## 依赖（Linux）

```bash
sudo apt install build-essential cmake python3-dev   # Debian/Ubuntu 示例
pip install -r requirements.txt   # includes pybind11
```

`scripts/build_linux.sh` will **`pip install pybind11`** for the same `python3` used by CMake if it is missing.

## 编译与安装

```bash
git clone https://github.com/StarlightChessOrg/MyCChessSF.git
cd MyCChessSF
bash scripts/build_linux.sh
```

编译完成后仓库根目录会有：

- `xqwlight_core*.so` — Python 扩展（规则 + 搜索）
- `xqwl_gen_nnue` — NNUE 训练数据生成器（独立可执行文件，无需 Python）

`build_linux.sh` 默认以 **`-O3`** 编译（可通过 `CXXOPT=-O2 bash scripts/build_linux.sh` 覆盖）。

或手动：

```bash
python3 -m pip install pybind11
cd cpp/build
cmake .. -DPython_EXECUTABLE="$(which python3)" -DCMAKE_CXX_FLAGS="-O3"
cmake --build . -j"$(nproc)"
cp xqwlight_core*.so ../..
cp xqwl_gen_nnue ../..
pip install -e ..
```

## 网页对弈

```bash
mycchess-play-web --host 0.0.0.0 --port 5151
mycchess-play-web --think-ms 2000          # 每步思考毫秒数
mycchess-play-web --no-book-default        # 启动时默认不用开局库
```

- 默认：**红方人类、黑方象棋小巫师**
- **界面**：`mycchess_sf/static/xqwl/`（棋盘 `board.png`、棋子、高亮、WAV 音效，源自 xqbase/xqwlight）
- **开局库**：`data/BOOK.DAT` 已随仓库提供；侧栏可勾选是否启用
- 侧栏可开关 **音效**（走子/吃子/将军/胜负等与 Win32 一致）
- `Engine.search_best_iccs(position, time_ms, use_book=True)` 支持逐步控制
- `Engine.search_best_detail(...)` 返回 `iccs`、`depth`、`score`（根节点搜索分，行棋方视角）、`from_book`

## NNUE 训练数据生成（`xqwl_gen_nnue`）

用于批量生成 **FEN + 搜索评分** 监督数据，供后续 NNUE 模型训练。

### 行为

- 自对弈采样：每个局面记录一行，然后走子进入下一局面；终局后重新 `Startup()`
- 走子策略：**20% 随机合法着** / **80% 引擎搜索结果**（默认，可调）
- **不使用开局库**
- `vl` 为**当前走子方视角**的根节点搜索分（与 `search_best_detail` 的 `score` 一致）
- FEN 编码在 C++ 内联实现（`cpp/xqwl_fen.inc`），便于高速调用
- 多进程并行：`fork()` 启动 worker，默认进程数 = CPU 核数
- 每个 worker 写入独立文件：`{output_dir}/worker_{id}.txt`

### 用法

```bash
# 默认：1000 万局面，CPU 核数个进程，输出到 nnue_data/
./xqwl_gen_nnue

# 自定义
./xqwl_gen_nnue \
  --output-dir ./nnue_data \
  --max-positions 10000000 \
  --think-ms 100 \
  --jobs 24 \
  --random-pct 20
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--output-dir` | `nnue_data` | 输出目录（自动创建） |
| `--max-positions` | `10000000` | 局面总数上限（非局数） |
| `--think-ms` | `100` | 每局面搜索时限（毫秒） |
| `--jobs` | `0`（= CPU 核数） | 并行 worker 数 |
| `--random-pct` | `20` | 随机走子概率（0–100） |

### 输出格式

每行一条样本，制表符分隔：

```text
rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1	7
```

- 左侧：标准象棋 FEN（红方在 FEN 底部，大写为红）
- 右侧：整数搜索分（行棋方视角；开局约 ±1~±15，优势局面可达 ±100+，将杀级接近 ±9800）

### 磁盘占用（估算）

未压缩纯文本，1000 万行约 **0.8–1.0 GB**（平均每行 ~80 字节）。24 个 worker 时每个文件约 **30–40 MB**（按默认配额均分）。

生成过程中 stderr 会打印 worker 启动与进度（每 10000 局面）；输出文件采用行缓冲并定期 `fflush`，可实时查看：

```bash
wc -l nnue_data/worker_*.txt
du -sh nnue_data/
head nnue_data/worker_0.txt
```

小规模验证：

```bash
./xqwl_gen_nnue --max-positions 1000 --jobs 2 --output-dir nnue_test
```

## 项目结构

```
cpp/                      xqwl_extract.inc + xqwl_search.inc + pybind11 绑定
cpp/gen_nnue_main.cpp     NNUE 数据生成器主程序
cpp/xqwl_fen.inc          C++ 内联 FEN 编码
data/BOOK.DAT             开局库（自 xqbase/xqwlight）
mycchess_sf/static/xqwl/  Win32 界面资源（PNG + WAV）
mycchess_sf/              Python 封装与 play_web.py
scripts/                  build_linux.sh, fetch_xqwl_assets.py
```

## 源码出处

- 引擎：`XQWL06.CPP`（elephantbase.net / GPL）
- 界面资源：`xqbase/xqwlight` Win32/RES（GPL）
- 规则抽取：MyCChessRL `xqwl_extract.inc`

## 限制

- 仅标准开局 `reset()`，无任意 FEN 设局
- 单进程 Sanic，多用户请多实例部署
- `xqwl_gen_nnue` 仅支持 Linux（`fork` / POSIX）

## 许可证

XQWL 版权见 elephantbase.net；本仓库见 [LICENSE](LICENSE)。
