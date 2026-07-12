# INT8 NNUE 模型（部署源）

本目录存放 **量化后的 NNUE 权重**（**XQWL-PSQ，1260 维**），供 `scripts/build_linux.sh` 打包进部署包。

> 若 `quantized.xqnnue.bin` 为旧版 HalfKA（16536 维），C++ 会拒绝加载；请重新训练/量化，或关闭网页对弈中的「NNUE 评估」改用动态 PST。

## 文件

| 文件 | 说明 |
|------|------|
| `quantized.xqnnue.bin` | C++ / 搜索引擎加载的二进制权重（**部署必需**） |
| `quantized.xqint8.pt` | Python 量化 checkpoint（可选，便于校验与再导出） |

## 生成与更新

```bash
python nnue_qINT8/quantize.py --checkpoint tmps/best.pt \
  --data-dir ../nnue_data --data-pattern "2/worker_0/chunk_0.txt"

cp nnue_qINT8/output/quantized.xqnnue.bin data/nnue_model/
cp nnue_qINT8/output/quantized.xqint8.pt data/nnue_model/   # 可选
```

Windows PowerShell：

```powershell
Copy-Item nnue_qINT8\output\quantized.xqnnue.bin data\nnue_model\
```

## 部署与搜索

**仓库源目录**：`data/nnue_model/` 存放量化后的 canonical 权重（版本管理 / 迭代入库用）。

**运行时目录**：`bash scripts/build_linux.sh` 将 `data/nnue_model/*.xqnnue.bin` 复制到 **`deployment/model/`**。部署包、网页对弈与 `xqwl_gen_nnue` **优先加载 `deployment/model/`**（Release 包内仅有此路径）。

### NNUE 迭代流程

```bash
# 1. 量化后更新仓库源
cp nnue_qINT8/output/quantized.xqnnue.bin data/nnue_model/

# 2. 同步进部署目录（build 会自动复制；也可手动）
cp data/nnue_model/quantized.xqnnue.bin deployment/model/
# 或：bash scripts/build_linux.sh

# 3. 用 deployment/model 中的权重生成下一轮数据
./deployment/bin/xqwl_gen_nnue --output-dir nnue_data
```

`*.bin` / `*.pt` 体积较大，默认 **不入库**（见根目录 `.gitignore`）。
