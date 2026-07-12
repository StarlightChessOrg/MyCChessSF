# INT8 量化与 C++ 推理

目录：`nnue_qINT8/`。将浮点 checkpoint 量化为 INT8，导出 C++ 可读二进制，并在搜索中增量推理。

## 依赖

```bash
pip install -r nnue_qINT8/requirements.txt
```

## 量化与导出

```bash
cd nnue_qINT8
python3 quantize.py
# 可选：--checkpoint ../nnue_training/checkpoints/best.pt
#       --max-samples 10000
```

流程：

1. 加载浮点 `best.pt`
2. 对称 INT8 量化（FT 按列，FC 按输出通道）
3. **校准 FC input scale**（在 nnue_data 上扫描各层输入最大值）
4. 评估 float vs int8 指标
5. 保存 `output/quantized.xqint8.pt` 与 `output/quantized.xqnnue.bin`

单独导出 bin：

```bash
python3 export_bin.py --input output/quantized.xqint8.pt --output output/quantized.xqnnue.bin
```

## 二进制格式（`.xqnnue.bin`）

| 字段 | 说明 |
|------|------|
| magic | `XQNNUE01` |
| version | `2`（v1 仍可读，缺 input scale 时用保守默认） |
| label_mean / label_std | 与训练 checkpoint 一致 |
| n_features, l1, l2, l3 | 1260, 512, 32, 32 |
| ft_clip, act_clip | 激活截断 |
| fc0/1/2_input_scale | **v2 新增**，FC 层输入量化步长 |
| 权重 | FT + FC0/1/2 的 int8 权重、int32 bias、float scales |

## INT8 FC 公式

```
q = clamp(round(x / input_scale), 0, 127)
y = weight_scale * bias_int32 + weight_scale * dot(w_int8, q) * input_scale
```

Python 参考：`QuantizedNNUE.forward_norm_int8()`（`int8_nnue.py`）。

## C++ 推理架构

| 组件 | 说明 |
|------|------|
| FT 累加器 | int32 列累加，**单视角增量**（每步更新 `acc[1-sdPlayer]`，`acc_valid[]` lazy refresh） |
| 增量更新 | 搜索 `MakeMove`/`UndoMakeMove` 维护累加器；排序/探测不触发 NNUE |
| 评估模式 | 内部节点 Raw；根/QSearch Full（含 PST 将杀修正） |
| FC | 纯 int8×uint8 点积 + input scale |
| SIMD | 运行时派发：Scalar / **NEON**（ARM）/ **AVX2** / **AVX512-VNNI**（独立 TU） |
| 将杀修正 | PST 极大时采用 PST；将军/大分歧时与 PST 混合 |

查看当前 SIMD 后端：

```python
engine = xc.Engine()
engine.load_nnue("nnue_qINT8/output/quantized.xqnnue.bin")
print(engine.nnue_simd_backend())  # 如 AVX2、NEON、AVX512_VNNI
```

## Python API

```python
import xqwlight_core as xc

engine = xc.Engine()
engine.load_nnue("nnue_qINT8/output/quantized.xqnnue.bin")

pos = xc.Position()
engine.evaluate_nnue(pos)       # 含将杀修正，搜索用
engine.evaluate_nnue_raw(pos)   # 纯 NNUE 输出
engine.verify_nnue_incremental(pos)  # 增量 vs 全量，应返回 0

engine.clear_nnue()
```

## 一致性测试

```bash
# 需已编译 xqwlight_core.so，且已运行 quantize.py
PYTHONPATH=. python3 scripts/test_nnue_wsl.py
```

通过标准（摘要）：

- C++ vs Python **int8** 路径：max \|diff\| ≤ 1.0
- 增量 vs 全量：`incremental_drift == 0`
- Python float vs int8：量化固有误差（校准集 MAE 接近）

## 编译注意

- x86：主模块 `-mavx2 -mfma`；`xqwl_nnue_simd_vnni.cpp` 单独 `-mavx512f -mavx512vnni -mavx512vl`
- AVX512 路径需 **CPUID + xgetbv** 双重检测，避免在 WSL 等环境误启用
- ARM：自动启用 NEON 路径

## 相关文档

- [引擎算法技术](engine-algorithms.md) — 搜索、LMR / 空着剪枝、NNUE 集成
- [NNUE 管线总览](nnue-overview.md)
- [NNUE 架构调研（Pikafish）](nnue_pikafish_research.md)
