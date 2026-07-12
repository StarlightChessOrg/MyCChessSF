# MyCChessSF v0.1.0

首个正式发布包：Linux x86_64 部署包，含网页对弈、178 万条开局库与 INT8 NNUE 权重。

## 下载

| 文件 | 说明 |
|------|------|
| `MyCChessSF-v0.1.0-linux-x86_64-py3.13.tar.gz` | 完整部署包（解压即用） |
| `MyCChessSF-v0.1.0-linux-x86_64-py3.13.tar.gz.sha256` | 校验和 |

**要求**：Linux x86_64、Python **3.13**（与包内 `.so` 编译版本一致）。

## 快速开始

```bash
tar -xzf MyCChessSF-v0.1.0-linux-x86_64-py3.13.tar.gz
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
- `model/quantized.xqnnue.bin` — XQWL-PSQ INT8 NNUE（epoch 22）

## 本版本亮点

- **NNUE**：XQWL-PSQ 1260 维，INT8 增量推理（单视角 FT + lazy refresh）
- **搜索优化**：TT staticEval、惰性 NNUE、内部 Raw eval
- **剪枝**：Razoring、Reverse futility、Null 门控、Quiet futility、LMP、QSearch 吃子 futility
- **文档**：[引擎算法技术](docs/engine-algorithms.md)

## 从源码构建

```bash
git clone https://github.com/StarlightChessOrg/MyCChessSF.git
cd MyCChessSF
bash scripts/make_release.sh 0.1.0
```

## 许可证

XQWL 版权见 elephantbase.net；本仓库见 [LICENSE](LICENSE)。
