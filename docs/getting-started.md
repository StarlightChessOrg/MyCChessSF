# 快速开始

MyCChessSF 的目标平台是 **Linux**：在此编译 `xqwlight_core.so`（Python 扩展）与 `xqwl_gen_nnue`（NNUE 数据生成器）。

## 依赖

Debian / Ubuntu 示例：

```bash
sudo apt install build-essential cmake python3-dev
pip install -r requirements.txt   # 含 pybind11、Sanic 等
```

`scripts/build_linux.sh` 会用当前 `python3` 自动 `pip install pybind11`（若尚未安装）。

NNUE 训练与量化还需额外依赖：

```bash
pip install -r nnue_training/requirements.txt
pip install -r nnue_qINT8/requirements.txt
```

## 一键编译

```bash
git clone --recurse-submodules https://github.com/StarlightChessOrg/MyCChessSF.git
cd MyCChessSF
bash scripts/build_linux.sh
```

已克隆但未初始化子模块时：

```bash
git submodule update --init --recursive
```

编译完成后，仓库根目录会出现：

| 产物 | 说明 |
|------|------|
| `xqwlight_core*.so` | Python 扩展：规则核 + 搜索 + NNUE |
| `xqwl_gen_nnue` | NNUE 训练数据生成器（独立可执行，无需 Python） |

同时生成 **`deployment/` 部署包**（可直接拷贝到服务器）：

| 路径 | 说明 |
|------|------|
| `deployment/lib/` | `xqwlight_core*.so` + `mycchess_sf/`（网页对弈） |
| `deployment/bin/` | `mycchess-play-web`、`xqwl_gen_nnue` |
| `deployment/db/BOOK.DAT` | 开局库 |
| `deployment/model/` | INT8 NNUE（来自 `data/nnue_model/`） |
| `deployment/requirements.txt` | 对弈服务 Python 依赖 |
| `deployment/README.md` | 部署说明 |

`build_linux.sh` 默认以 **`-O3`** 编译；可覆盖：

```bash
CXXOPT=-O2 bash scripts/build_linux.sh
```

## 手动编译

```bash
python3 -m pip install pybind11
cd cpp/build
cmake .. -DPython_EXECUTABLE="$(which python3)" -DCMAKE_CXX_FLAGS="-O3"
cmake --build . -j"$(nproc)"
cp xqwlight_core*.so ../..
cp xqwl_gen_nnue ../..
pip install -e ..
```

## 验证安装

```bash
python3 -c "import xqwlight_core as xc; print('ok', xc.Engine())"
mycchess-play-web --help
```

若已量化 NNUE，可运行对比测试（需 WSL / Linux 与 torch）：

```bash
PYTHONPATH=. python3 scripts/test_nnue_wsl.py
```

## 下一步

- [网页对弈](web-play.md)
- [NNUE 管线总览](nnue-overview.md)
