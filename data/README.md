# 开局库（来自象棋小巫师 Win32/BOOK.DAT）

仓库已包含 `BOOK.DAT`（约 96KB，12081 条开局库项）。

网页侧栏可勾选 **「小巫师使用开局库」**；取消勾选则每步仅 Alpha-Beta 搜索、不走开局库。

也可在启动时默认关闭（网页仍可勾选）：

```bash
mycchess-play-web --no-book-default
```

指定其他开局库文件：

```bash
mycchess-play-web --book /path/to/BOOK.DAT
```
