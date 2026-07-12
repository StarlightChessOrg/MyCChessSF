# 数据资产

本目录包含 **`compressed_files/`**（压缩包）、**`nnue_model/`**（INT8 NNUE 部署源）与本说明。运行时解压产物（如 `BOOK.DAT`）不存放在 `compressed_files/`，由构建脚本写入 **`deployment/`**。

## compressed_files/

源自 Chess98 `tools/openbook` 的 7z 压缩包：

| 文件 | 说明 |
|------|------|
| `book.7z` | 完整开局库（约 178 万条，14 MB 解压） |
| `make_book.7z` | 制库工具 |
| `master_cbf.7z` / `master_pgn.7z` | 大师棋谱源数据 |
| `mxq_to_fen.7z` | 棋谱转 FEN 工具 |

旧版 xqbase 小库（约 1.2 万条、96 KB）已弃用。

## 运行时 BOOK.DAT

`bash scripts/build_linux.sh` 会从 `compressed_files/book.7z` 解压到 **`deployment/db/BOOK.DAT`**。也可单独解压：

```bash
pip install py7zr
python scripts/extract_book.py
```

详细用法见 [docs/opening-book.md](../docs/opening-book.md)。

## nnue_model/

存放 INT8 量化后的 NNUE 权重（`quantized.xqnnue.bin` 等）。`bash scripts/build_linux.sh` 会将其复制到 **`deployment/model/`**。详见 [nnue_model/README.md](nnue_model/README.md)。
