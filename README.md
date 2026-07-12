# 象眸 SF · Iris SF

> 脱胎象棋小巫师（XQWL 0.6），融合 INT8 NNUE、现代搜索与网页对弈。  
> *Elephant Iris — xiangqi engine rooted in XQWL, sharpened with neural eval.*

在浏览器中与 **象眸** 对弈：规则核 + Alpha-Beta 搜索来自 `XQWL06.CPP`，网页棋盘 / 棋子 / 音效复用 Win32 原版资源；可选 **NNUE** 静态评估（拟合小巫师搜索分，INT8 增量推理 + SIMD）。

GitHub 仓库名仍为 [MyCChessSF](https://github.com/StarlightChessOrg/MyCChessSF)（历史路径，与包名 `mycchess-sf` 一致）。

**目标平台：Linux**（编译 `xqwlight_core.so` 与 `xqwl_gen_nnue`）。

不想从源码编译？可直接下载 [GitHub Releases](https://github.com/StarlightChessOrg/MyCChessSF/releases/latest) 中的 Linux 部署包（`MyCChessSF-v*-linux-x86_64-py3.13.tar.gz`）。

## 快速开始

```bash
git clone --recurse-submodules https://github.com/StarlightChessOrg/MyCChessSF.git
cd MyCChessSF
bash scripts/build_linux.sh
mycchess-play-web --port 5151
```

若已克隆但未拉子模块：`git submodule update --init --recursive`。

详细步骤见 [docs/getting-started.md](docs/getting-started.md)。

## 文档

### 入门与使用

| 文档 | 说明 |
|------|------|
| [快速开始](docs/getting-started.md) | 依赖安装、编译、`pip install -e .`、验证 |
| [网页对弈](docs/web-play.md) | `mycchess-play-web`、侧栏选项、搜索 API |
| [开局库](docs/opening-book.md) | `BOOK.DAT`、命令行与 API 开关 |
| [项目结构](docs/project-structure.md) | 目录说明、与 MyCChessRL 的区别 |

### 引擎算法

| 文档 | 说明 |
|------|------|
| [引擎算法技术](docs/engine-algorithms.md) | 搜索框架、NNUE 增量推理、LMR / 空着剪枝、PST |

### NNUE 管线

| 文档 | 说明 |
|------|------|
| [NNUE 管线总览](docs/nnue-overview.md) | 数据 → 训练 → 量化 → C++ 部署全流程 |
| [NNUE 训练数据生成](docs/nnue-data-generation.md) | `xqwl_gen_nnue`、**4 列 TSV 格式**、旧数据 augment、磁盘估算 |
| [NNUE 浮点训练](docs/nnue-training.md) | `nnue_training/`、PyTorch 配置与 checkpoint |
| [INT8 量化与 C++ 推理](docs/nnue-int8-inference.md) | 量化、`.xqnnue.bin`、SIMD、增量累加器、测试 |
| [NNUE 架构调研（Pikafish）](docs/nnue_pikafish_research.md) | 外部 NNUE 参考调研（非本引擎实现） |

### 开发

| 文档 | 说明 |
|------|------|
| [开发与限制](docs/development.md) | 源码出处、许可证、平台限制、提交前检查 |

## 一句话能力

- **对弈**：红人黑机，可选开局库与音效 → [网页对弈](docs/web-play.md)
- **NNUE**：自生成搜索分数据 → 训练 → INT8 → 搜索内评估 → [NNUE 总览](docs/nnue-overview.md)（数据格式见 [训练数据生成](docs/nnue-data-generation.md#输出格式)）

## 仓库概览

核心目录：`cpp/`（引擎）、`mycchess_sf/`（网页对弈）、`nnue_training/` / `nnue_qINT8/`（NNUE 管线）、`data/`（开局库压缩包与部署用 NNUE 权重）。完整说明见 [项目结构](docs/project-structure.md)。

| 目录 | 说明 |
|------|------|
| `third_party/` | 外部参考 **git submodule**（[Pikafish](https://github.com/official-pikafish/Pikafish)），仅供 NNUE/搜索调研阅读，**不参与本引擎编译**；见 `third_party/README.md` |
| `tmps/` | 本地临时目录，可放量化/推理冒烟测试用的 checkpoint（`*.pt` 已 gitignore，不入库） |
| `deployment/` | `build_linux.sh` 生成的可部署目录（gitignore，Release 包即其 tar 打包） |
| `dist/` | `make_release.sh` 产出的发布压缩包（gitignore） |

## 许可证

XQWL 版权见 elephantbase.net；本仓库见 [LICENSE](LICENSE)。
