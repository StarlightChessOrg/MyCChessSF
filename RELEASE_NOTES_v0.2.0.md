# 象眸 SF v0.2.0 · Iris SF

聚焦 **搜索质量** 与 **部署精简**：移除易致漏看战术的 staticEval 剪枝，删除相弈桥接实验代码，保留网页对弈 + NNUE 核心。

## 下载

| 文件 | 说明 |
|------|------|
| `MyCChessSF-v0.2.0-linux-x86_64-py3.13.tar.gz` | 完整部署包（解压即用） |
| `MyCChessSF-v0.2.0-linux-x86_64-py3.13.tar.gz.sha256` | 校验和 |

**要求**：Linux x86_64、Python **3.13**（与包内 `.so` 编译版本一致）。

## 快速开始

```bash
tar -xzf MyCChessSF-v0.2.0-linux-x86_64-py3.13.tar.gz
cd deployment
pip install -r requirements.txt
./bin/mycchess-play-web --host 0.0.0.0 --port 5151
```

浏览器访问 `http://<服务器>:5151/`。

## 包内容

- `lib/xqwlight_core*.so` — C++ 引擎（Alpha-Beta + NNUE INT8 + SIMD）
- `lib/mycchess_sf/` — Sanic 网页对弈
- `bin/mycchess-play-web` — 启动脚本
- `bin/xqwl_gen_nnue` — NNUE 训练数据生成器
- `db/BOOK.DAT` — 178 万条开局库
- `model/quantized.xqnnue.bin` — XQWL-PSQ INT8 NNUE

## 相对 v0.1.0 的变更

### 搜索

- **移除** staticEval 驱动剪枝（Razoring、Reverse futility、Quiet futility、LMP、QSearch 吃子 futility、Null 静态分门控）
- **保留** TT staticEval 缓存、惰性 NNUE、LMR、空着剪枝、IID、Singular extension
- 修复 NNUE 静态分噪声导致的「安静着看似合理、实际漏看战术」问题

### 清理

- 删除相弈象棋 HTTP 桥（`:9494`）与 Selenium 自动测试栈
- 删除 `bridge_codec.py`、`tools/auto/` 及相关脚本
- 移除 `XqwlGameState.set_fen()`（桥接专用；C++ `Position.set_fen()` 仍可用于测试）

### 文档

- 更新 [引擎算法技术](docs/engine-algorithms.md) 与 NNUE 相关文档，与当前搜索行为一致

## 从源码构建

```bash
git clone https://github.com/StarlightChessOrg/MyCChessSF.git
cd MyCChessSF
bash scripts/make_release.sh 0.2.0
```

## 许可证

XQWL 版权见 elephantbase.net；本仓库见 [LICENSE](LICENSE)。
