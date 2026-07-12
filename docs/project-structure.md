# 项目结构

仓库目录（GitHub：`MyCChessSF`；产品名：**象眸 SF / Iris SF**）：

```
MyCChessSF/
├── cpp/                          C++ 核心
│   ├── include/                  头文件
│   │   ├── xqwl_portable_prefix.h
│   │   ├── xqwl_nnue_simd.h
│   │   └── xqwl_nnue_simd_vnni.h
│   ├── inc/                      内联实现片段（.inc）
│   │   ├── xqwl_extract.inc      规则、走法、PST 评估
│   │   ├── xqwl_preeval*.inc     动态 PST 混合
│   │   ├── xqwl_search.inc       Alpha-Beta 搜索 + 开局库 hook
│   │   ├── xqwl_nnue.inc         INT8 NNUE 推理 + 单视角 FT 累加器
│   │   └── xqwl_fen.inc          内联 FEN 编解码
│   ├── bindings.cpp              pybind11 Python 绑定
│   ├── gen_nnue_main.cpp         NNUE 训练数据生成器
│   ├── xqwl_portable.cpp         引擎核心 umbrella include
│   ├── xqwl_nnue_simd_vnni.cpp   AVX512-VNNI 独立编译单元
│   └── CMakeLists.txt
├── mycchess_sf/                  Python 包
│   ├── play_web.py               Sanic 网页对弈
│   ├── static/xqwl/              Win32 界面资源（PNG + WAV）
│   └── ...
├── data/
│   ├── README.md
│   ├── compressed_files/         开局库与 openbook 压缩包（构建时解压 book.7z）
│   └── nnue_model/               INT8 NNUE 权重（打包源 → deployment/model/）
├── nnue_data/                    xqwl_gen_nnue 输出（FEN\tsearch\tPST\tin_check，通常不入库）
├── nnue_training/                浮点 NNUE PyTorch 训练
├── nnue_qINT8/                   INT8 量化 + .xqnnue.bin 导出
├── third_party/                  外部参考 git submodule，调研用，不参与编译
│   ├── README.md                 子模块说明与更新命令
│   └── Pikafish/                 → official-pikafish/Pikafish
├── tmps/                         本地临时 checkpoint，*.pt 不入库
├── scripts/                      构建、打包、测试脚本
│   ├── build_linux.sh            一键编译 + 打包 deployment/
│   ├── make_release.sh           打包 dist/ 发布压缩包
│   ├── extract_book.py           从 book.7z 解压 BOOK.DAT 到 deployment/db/
│   ├── test_nnue_wsl.py          C++ vs Python NNUE 对比测试
│   └── fetch_xqwl_assets.py      拉取网页 UI 资源
├── .github/workflows/            CI（推送 v* 标签时自动构建 Release）
├── deployment/                   build_linux.sh 产出（gitignore，可部署目录）
├── dist/                         make_release.sh 产出（gitignore，Release 压缩包）
└── docs/                         文档（本目录）
```

## 核心模块职责

| 模块 | 职责 |
|------|------|
| `xqwlight_core.so` | 局面、走法、PST、`Engine.search_*`、NNUE 加载与评估 |
| `xqwl_gen_nnue` | 多进程 fork 采样 FEN + 根节点搜索分 |
| `nnue_training/` | XQWL-PSQ 特征、EmbeddingBag + FC 头、拟合搜索分 |
| `nnue_qINT8/` | 对称 INT8 量化、校准 input scale、导出二进制权重 |
| `mycchess_sf/play_web.py` | 浏览器对弈前端 + Sanic 后端 |
| `data/nnue_model/` | 部署用 INT8 权重源（`quantized.xqnnue.bin` 等） |
| `data/compressed_files/` | 开局库等 7z 压缩包，`build_linux.sh` 解压 `book.7z` |

## 辅助与生成目录

与 [根目录 README](../README.md#仓库概览) 对应；下列路径多为本地生成或可选，**不影响从源码编译对弈**。

| 目录 | 说明 |
|------|------|
| `third_party/` | [Pikafish](https://github.com/official-pikafish/Pikafish) **git submodule**（`third_party/Pikafish`）。仅供 NNUE 特征、搜索剪枝等**调研对照**，CMake **不**编译、链接其中任何代码。克隆后需 `git submodule update --init`；说明见 [third_party/README.md](../third_party/README.md) |
| `tmps/` | 开发者本地 scratch。可将 `best.pt` 等 checkpoint 复制到此，跑 `nnue_qINT8/quantize.py` 或推理冒烟；`*.pt` 已在 `.gitignore`，不入库。说明见 `tmps/README.md` |
| `deployment/` | `bash scripts/build_linux.sh` 生成的**可部署目录**（`lib/`、`bin/`、`db/`、`model/`）。已在 `.gitignore`；[GitHub Releases](https://github.com/StarlightChessOrg/MyCChessSF/releases/latest) 上的 tar 包即此目录的打包 |
| `dist/` | `bash scripts/make_release.sh <版本>` 输出的 `MyCChessSF-v*-linux-x86_64-py*.tar.gz` 及 `.sha256`。已在 `.gitignore` |
| `nnue_data/` | `xqwl_gen_nnue` 生成的训练语料（4 列 TSV，见 README），体积大，通常不入库 |
| 根目录 `xqwlight_core*.so` / `xqwl_gen_nnue` | 编译产物拷贝，便于 `PYTHONPATH=.` 测试；`.gitignore` 忽略 |

## 发布相关

| 路径 | 说明 |
|------|------|
| `scripts/make_release.sh` | 本地：构建 `deployment/` 并打 tar 包到 `dist/` |
| `.github/workflows/release.yml` | 推送 `v*` 标签时 CI 自动构建并上传 Release 附件 |
| `RELEASE_NOTES_v*.md` | 对应版本的 Release 说明正文（如 `RELEASE_NOTES_v0.2.0.md`） |

本地打包示例：

```bash
bash scripts/make_release.sh 0.2.0
# 产物：dist/MyCChessSF-v0.2.0-linux-x86_64-py3.13.tar.gz
```

更多见 [开发与限制 — Release 发布](development.md#release-发布)。

## 与 MyCChessRL 的区别

| 项目 | 规则核 | AI |
|------|--------|-----|
| MyCChessRL | XQWL 局面核 | PyTorch / MCTS |
| **象眸 SF** | 同上 | **脱胎小巫师 Alpha-Beta + 开局库 + NNUE（Iris SF）** |

算法细节见 [引擎算法技术](engine-algorithms.md)。NNUE 训练特征采用 **XQWL-PSQ**（1260 维）；Pikafish HalfKA 调研见 [NNUE 架构调研（Pikafish）](nnue_pikafish_research.md)（仅供参考，当前管线未启用）。
