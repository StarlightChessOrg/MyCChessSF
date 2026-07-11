# MyCChessSF

在浏览器里和 **象棋小巫师（XQWL 0.6）** 对弈：规则核 + Alpha-Beta 搜索来自 `XQWL06.CPP`，**网页棋盘/棋子/音效复用 Win32 原版资源**。

**目标平台：Linux**（编译 `xqwlight_core.so`）。

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

或手动：

```bash
python3 -m pip install pybind11
cd cpp/build
cmake .. -DPython_EXECUTABLE="$(which python3)"
cmake --build . -j"$(nproc)"
cp xqwlight_core*.so ../..
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

## 项目结构

```
cpp/                    xqwl_extract.inc + xqwl_search.inc + pybind11 绑定
data/BOOK.DAT           开局库（自 xqbase/xqwlight）
mycchess_sf/static/xqwl/  Win32 界面资源（PNG + WAV）
mycchess_sf/            Python 封装与 play_web.py
scripts/                build_linux.sh, fetch_xqwl_assets.py
```

## 源码出处

- 引擎：`XQWL06.CPP`（elephantbase.net / GPL）
- 界面资源：`xqbase/xqwlight` Win32/RES（GPL）
- 规则抽取：MyCChessRL `xqwl_extract.inc`

## 限制

- 仅标准开局 `reset()`，无任意 FEN 设局
- 单进程 Sanic，多用户请多实例部署

## 许可证

XQWL 版权见 elephantbase.net；本仓库见 [LICENSE](LICENSE)。
