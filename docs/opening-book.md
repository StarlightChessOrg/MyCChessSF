# 开局库

压缩包 **`data/compressed_files/book.7z`** 源自 Chess98 openbook（约 **178 万条**，14 MB 解压；旧版 xqbase 小库约 1.2 万条已替换）。

制库工具与棋谱源见同目录下其他 7z（`make_book.7z`、大师棋谱等），清单见 **`data/README.md`**。

## 部署包中的 BOOK.DAT

`bash scripts/build_linux.sh` 会：

1. 从 `data/compressed_files/book.7z` 解压 → **`deployment/db/BOOK.DAT`**
2. 将 `xqwlight_core*.so` 放入 **`deployment/lib/`**，`xqwl_gen_nnue` 放入 **`deployment/bin/`**
3. 生成 **`deployment/README.md`** 简要说明

也可单独解压（输出同样到 `deployment/db/`）：

```bash
pip install py7zr
python scripts/extract_book.py
```

## 网页侧栏

勾选 **「小巫师使用开局库」** 时，搜索优先查库；取消勾选则每步纯 Alpha-Beta，不走库。

## 命令行

启动时默认关闭开局库（网页仍可手动勾选）：

```bash
mycchess-play-web --no-book-default
```

指定开局库（默认查找 `deployment/db/BOOK.DAT`）：

```bash
mycchess-play-web --book deployment/db/BOOK.DAT
```

## 搜索 API

```python
engine.search_best_iccs(pos, time_ms=1000, use_book=True)   # 默认走库
engine.search_best_iccs(pos, time_ms=1000, use_book=False)  # 禁用开局库
```

`search_best_detail` 返回的 `from_book=True` 表示本步着法来自开局库。

## 与 NNUE 的关系

- **开局库着法**：仍由传统搜索 + 库表决定
- **空着剪枝（Null Move）**：使用 PST 子力评估，**不使用 NNUE**（避免 NNUE 在空着局面下失真）
- 生成 NNUE 训练数据时 `xqwl_gen_nnue` **默认不用开局库**，以保证分数标签来自搜索而非库表
