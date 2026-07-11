# MyCChessSF

在浏览器里和 **象棋小巫师（XQWL 0.6）** 对弈：规则核 + Alpha-Beta 搜索来自 `XQWL06.CPP`，网页棋盘 / 棋子 / 音效复用 Win32 原版资源。可选 **NNUE** 静态评估（拟合小巫师搜索分，INT8 增量推理 + SIMD）。

**目标平台：Linux**（编译 `xqwlight_core.so` 与 `xqwl_gen_nnue`）。

## 快速开始

```bash
git clone https://github.com/StarlightChessOrg/MyCChessSF.git
cd MyCChessSF
bash scripts/build_linux.sh
mycchess-play-web --port 5151
```

详细步骤见 [docs/getting-started.md](docs/getting-started.md)。

## 文档

### 入门与使用

| 文档 | 说明 |
|------|------|
| [快速开始](docs/getting-started.md) | 依赖安装、编译、`pip install -e .`、验证 |
| [网页对弈](docs/web-play.md) | `mycchess-play-web`、侧栏选项、搜索 API |
| [开局库](docs/opening-book.md) | `BOOK.DAT`、命令行与 API 开关 |
| [项目结构](docs/project-structure.md) | 目录说明、与 MyCChessRL 的区别 |

### NNUE 管线

| 文档 | 说明 |
|------|------|
| [NNUE 管线总览](docs/nnue-overview.md) | 数据 → 训练 → 量化 → C++ 部署全流程 |
| [NNUE 训练数据生成](docs/nnue-data-generation.md) | `xqwl_gen_nnue` 参数、输出格式、磁盘估算 |
| [NNUE 浮点训练](docs/nnue-training.md) | `nnue_training/`、PyTorch 配置与 checkpoint |
| [INT8 量化与 C++ 推理](docs/nnue-int8-inference.md) | 量化、`.xqnnue.bin`、SIMD、增量累加器、测试 |
| [NNUE 架构调研（Pikafish）](docs/nnue_pikafish_research.md) | 外部 NNUE 参考调研（非本引擎实现） |

### 开发

| 文档 | 说明 |
|------|------|
| [开发与限制](docs/development.md) | 源码出处、许可证、平台限制、提交前检查 |

## 一句话能力

- **对弈**：红人黑机，可选开局库与音效 → [网页对弈](docs/web-play.md)
- **NNUE**：自生成搜索分数据 → 训练 → INT8 → 搜索内评估 → [NNUE 总览](docs/nnue-overview.md)

## 许可证

XQWL 版权见 elephantbase.net；本仓库见 [LICENSE](LICENSE)。
