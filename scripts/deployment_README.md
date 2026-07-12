# MyCChessSF 部署包

本目录由 `bash scripts/build_linux.sh` 自动生成，可直接拷贝到 Linux 服务器使用。

## 目录结构

```
deployment/
├── README.md              本说明
├── requirements.txt       网页对弈 Python 依赖（numpy、sanic）
├── lib/
│   ├── xqwlight_core*.so      C++ 引擎扩展（规则 + 搜索 + NNUE）
│   └── mycchess_sf/            网页对弈 Python 包（含 static/xqwl 界面资源）
├── bin/
│   ├── mycchess-play-web      网页对弈启动脚本
│   └── xqwl_gen_nnue          NNUE 训练数据生成器
├── db/
│   └── BOOK.DAT               开局库
└── model/
    └── quantized.xqnnue.bin   INT8 NNUE 权重（来自 data/nnue_model/）
```

## 网页对弈（人类 vs 小巫师）

```bash
pip install -r deployment/requirements.txt
./deployment/bin/mycchess-play-web --host 0.0.0.0 --port 5151
```

默认加载 `deployment/db/BOOK.DAT`；浏览器访问 `http://<服务器>:5151/`。

常用参数：

```bash
./deployment/bin/mycchess-play-web --think-ms 2000
./deployment/bin/mycchess-play-web --no-book-default
./deployment/bin/mycchess-play-web --book /path/to/other/BOOK.DAT
```

## Python 引擎 API

```python
import sys
sys.path.insert(0, "deployment/lib")
import xqwlight_core as xc

engine = xc.Engine()
engine.load_book("deployment/db/BOOK.DAT")
engine.load_nnue("deployment/model/quantized.xqnnue.bin")
```

## NNUE 数据生成

```bash
./deployment/bin/xqwl_gen_nnue --output-dir nnue_data --max-positions 10000 --jobs 4
```

## 说明

- **对弈程序**即 `bin/mycchess-play-web` + `lib/mycchess_sf/`，不是单独的二进制，需 Python 3.10+ 与 `requirements.txt` 中的依赖。
- `lib/` 放 `.so` 与 Python 包；`bin/` 放可执行脚本/程序。
- `xqwlight_core*.so` 与构建所用 Python 版本绑定，换 Python 需重新运行 `build_linux.sh`。
- 完整开发流程见仓库 `docs/`。
