# MyCChessSF 部署包

本目录由 `bash scripts/build_linux.sh` 自动生成，可直接拷贝到 Linux 服务器使用。

## 目录结构

```
deployment/
├── README.md          本说明
├── lib/
│   └── xqwlight_core*.so   Python 扩展（规则 + 搜索 + NNUE）
├── bin/
│   └── xqwl_gen_nnue       NNUE 训练数据生成器（独立可执行）
└── db/
    └── BOOK.DAT            开局库（自 data/compressed_files/book.7z 解压，约 178 万条）
```

## 快速使用

### Python 引擎 / 网页对弈

在仓库根目录已 `pip install -e .` 的前提下：

```bash
export PYTHONPATH="$(pwd)/deployment/lib:${PYTHONPATH:-}"
mycchess-play-web --host 0.0.0.0 --port 5151 --book "$(pwd)/deployment/db/BOOK.DAT"
```

或在 Python 中：

```python
import sys
sys.path.insert(0, "deployment/lib")
import xqwlight_core as xc

engine = xc.Engine()
engine.load_book("deployment/db/BOOK.DAT")
```

### NNUE 数据生成

```bash
./deployment/bin/xqwl_gen_nnue --output-dir nnue_data --max-positions 10000 --jobs 4
```

无 NNUE 权重时自动使用 PST 评估；量化后可自动加载 `nnue_qINT8/output/quantized.xqnnue.bin`。

## 说明

- `lib/` 放 Python 扩展 `.so`，`bin/` 放独立可执行程序，二者职责不同。
- `BOOK.DAT` 源文件为仓库内 `data/compressed_files/book.7z`，构建时解压，**勿**在 `data/` 下单独存放解压后的库。
- `xqwlight_core*.so` 与构建所用 Python 版本绑定，换 Python 需重新运行 `build_linux.sh`。
- 完整开发与 NNUE 流程见仓库 `docs/` 与根目录 `README.md`。
