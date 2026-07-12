# 网页对弈

**象眸 SF**（*Iris SF*）浏览器对弈入口：`mycchess-play-web`。

## 部署包内启动（无需 pip install -e .）

若已运行 `bash scripts/build_linux.sh` 生成 `deployment/`：

```bash
pip install -r deployment/requirements.txt
./deployment/bin/mycchess-play-web --host 0.0.0.0 --port 5151
```

## 开发模式启动

在仓库根目录 `pip install -e .` 后：

```bash
mycchess-play-web --host 0.0.0.0 --port 5151
mycchess-play-web --think-ms 2000          # 每步思考毫秒数
mycchess-play-web --no-book-default        # 启动时默认不用开局库
mycchess-play-web --no-nnue-default        # 有 NNUE 权重时也默认用 PST
```

## 默认对局设置

- **红方**：人类
- **黑方**：象眸引擎（脱胎小巫师 Alpha-Beta + 可选开局库 + NNUE）

## 界面与资源

- 静态资源目录：`mycchess_sf/static/xqwl/`
- 内容：棋盘 `board.png`、棋子 PNG、高亮图、WAV 音效（源自 xqbase/xqwlight Win32 资源）
- 首次构建若缺少资源，`build_linux.sh` 会调用 `scripts/fetch_xqwl_assets.py` 拉取

## 侧栏选项

| 选项 | 说明 |
|------|------|
| 使用开局库 BOOK.DAT | 见 [开局库](opening-book.md) |
| 使用 NNUE 评估 | 检测到权重时**默认勾选**；无权重时禁用 |
| 悔棋 | 回到**当前人类方**的上一个决策点（人机对弈时撤销己方上一手 + 引擎应手，共 2 步）；**引擎思考/应招期间禁用** |
| 音效 | 走子 / 吃子 / 将军 / 胜负等，与 Win32 一致 |

## Python API（搜索）

```python
import xqwlight_core as xc

engine = xc.Engine()
pos = xc.Position()

# 加载 NNUE（可选）
engine.load_nnue("nnue_qINT8/output/quantized.xqnnue.bin")

iccs = engine.search_best_iccs(pos, time_ms=1000, use_book=True)
detail = engine.search_best_detail(pos, time_ms=1000, use_book=True)
# detail: iccs, depth, score（根节点分，行棋方视角）, from_book
```

NNUE 相关 API 见 [INT8 量化与 C++ 推理](nnue-int8-inference.md)。

## 部署说明

- 单进程 Sanic，适合本地或单机部署
- 多用户并发请多实例 / 多进程部署，而非单进程扛全部连接
