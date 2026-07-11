# 开局库

仓库已随附象棋小巫师 Win32 开局库 **`data/BOOK.DAT`**（约 96 KB，12081 条项）。

## 网页侧栏

勾选 **「小巫师使用开局库」** 时，搜索优先查库；取消勾选则每步纯 Alpha-Beta，不走库。

## 命令行

启动时默认关闭开局库（网页仍可手动勾选）：

```bash
mycchess-play-web --no-book-default
```

指定其他开局库文件：

```bash
mycchess-play-web --book /path/to/BOOK.DAT
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
