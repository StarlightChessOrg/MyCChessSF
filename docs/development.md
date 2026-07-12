# 开发与限制

## 源码出处

| 组件 | 来源 |
|------|------|
| 搜索引擎 | `XQWL06.CPP`（elephantbase.net / GPL） |
| 界面资源 | xqbase/xqwlight Win32/RES（GPL） |
| 规则抽取 | MyCChessRL `xqwl_extract.inc` 同源思路 |
| NNUE | XQWL-PSQ 动态特征（1260 维） |

## 许可证

XQWL 版权见 elephantbase.net；本仓库见根目录 [LICENSE](../LICENSE)。

## 已知限制

| 限制 | 说明 |
|------|------|
| 设局 | 网页默认标准开局 `reset()`；Python 侧支持 `Position.set_fen()` |
| 平台 | 主目标 **Linux**；`xqwl_gen_nnue` 依赖 `fork()` / POSIX |
| 网页 | 单进程 Sanic，高并发请多实例 |
| NNUE bin | `output/quantized.xqnnue.bin` 需本地运行 `quantize.py` 生成，通常不入库 |
| Windows | 建议在 **WSL** 编译 `.so` 并跑测试 |

## 常用脚本

| 脚本 | 用途 |
|------|------|
| `scripts/build_linux.sh` | 编译 + `pip install -e .` |
| `scripts/test_nnue_wsl.py` | C++ / Python NNUE 对比 |
| `scripts/fetch_xqwl_assets.py` | 下载网页 UI 资源 |
| `scripts/regen_extract_en.py` | 维护用：规则抽取相关 |

## 提交前检查

```bash
bash scripts/build_linux.sh
cd nnue_qINT8 && python3 quantize.py --max-samples 2000
cd .. && PYTHONPATH=. python3 scripts/test_nnue_wsl.py
```

## 文档索引

- [引擎算法技术](engine-algorithms.md)
- 返回 [README 文档索引](../README.md#文档)
