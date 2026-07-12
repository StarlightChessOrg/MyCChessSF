# 项目结构

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
├── deployment/                   build_linux.sh 产出（lib/ + bin/ + db/ + model/ + requirements.txt）
├── nnue_data/                    xqwl_gen_nnue 输出（FEN\tvl，通常不入库）
├── nnue_training/                浮点 NNUE PyTorch 训练
├── nnue_qINT8/                   INT8 量化 + .xqnnue.bin 导出
├── third_party/                  外部参考（如 Pikafish），调研用，不参与编译
├── tmps/                         本地临时 checkpoint，*.pt 不入库
├── scripts/
│   ├── build_linux.sh            一键编译 + 打包 deployment/
│   ├── make_release.sh           打包 dist/ 发布压缩包
│   ├── extract_book.py           从 book.7z 解压 BOOK.DAT 到 deployment/db/
│   ├── test_nnue_wsl.py          C++ vs Python NNUE 对比测试
│   └── fetch_xqwl_assets.py      拉取网页 UI 资源
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

## 与 MyCChessRL 的区别

| 项目 | 规则核 | AI |
|------|--------|-----|
| MyCChessRL | XQWL 局面核 | PyTorch / MCTS |
| **MyCChessSF** | 同上 | **原版小巫师 Alpha-Beta + 开局库 + 可选 NNUE** |

算法细节见 [引擎算法技术](engine-algorithms.md)。NNUE 训练特征采用 **XQWL-PSQ**（1260 维）；Pikafish HalfKA 调研见 [NNUE 架构调研（Pikafish）](nnue_pikafish_research.md)（仅供参考，当前管线未启用）。
