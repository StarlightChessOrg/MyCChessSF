# 引擎算法技术

本文档单独介绍 **象眸 SF**（*Iris SF*）的 **搜索、评估、NNUE 推理** 等核心算法，与 [NNUE 管线总览](nnue-overview.md)（训练/量化/部署流程）互补。

## 总览

```
┌─────────────────────────────────────────────────────────────┐
│  SearchMain / SearchRoot / SearchFull / SearchQuiesc        │
│    Alpha-Beta + TT + aspiration + LMR + 空着/IID/Singular      │
└───────────────────────────┬─────────────────────────────────┘
                            │ 静态评估
              ┌─────────────┴─────────────┐
              ▼                           ▼
     NNUE evaluate_vl           PST Evaluate()
     (INT8 增量)                 (动态 PST，无 NNUE 时)
              │
              ▼
     XQWL-PSQ 1260 维稀疏特征
     FT int32 累加器 → FC int8 前向
```

| 层次 | 实现位置 | 要点 |
|------|----------|------|
| 搜索 | `cpp/inc/xqwl_search.inc` | 小巫师 Alpha-Beta 骨架 + LMR / 空着 / IID |
| NNUE | `cpp/inc/xqwl_nnue.inc` | XQWL-PSQ、单视角 FT 增量、SIMD FC |
| PST | `cpp/inc/xqwl_preeval.inc` | 动态中残局 PST 混合 |
| 规则核 | `cpp/inc/xqwl_extract.inc` | 走法生成、Zobrist、重复局面 |

---

## 搜索框架

基于 **象棋小巫师 XQWL 0.6** 的 `SearchMain → SearchRoot → SearchFull → SearchQuiesc` 递归结构（象眸 SF 的搜索骨架），在保持兼容性的前提下叠加若干现代技巧。

### 核心常量（节选）

定义于 `cpp/include/xqwl_portable_prefix.h`：

| 常量 | 典型值 | 用途 |
|------|--------|------|
| `MATE_VALUE` | 10000 | 将杀分尺度 |
| `WIN_VALUE` | 9800 | 胜势区边界 |
| `HASH_SIZE` | 2²⁰ | 置换表条目数 |
| `NULL_DEPTH` | 2 | 空着剪枝深度削减 |
| `LIMIT_DEPTH` | 64 | 最大 ply |

### 置换表（TT）

每条 `HashItem` 存储：

| 字段 | 说明 |
|------|------|
| `ucDepth` | 存储深度 |
| `ucFlag` | `HASH_ALPHA` / `HASH_BETA` / `HASH_PV` |
| `svl` | 搜索分值（含将杀距离编码） |
| **`sEval`** | **静态评估**（行棋方视角，centipawn） |
| `wmv` | 最佳着法 |
| `dwLock0/1` | Zobrist 锁 |

`ProbeHash` / `RecordHash` 在命中/回填时读写 `sEval`。QSearch 条目（depth=0）同样携带静态分，供后续节点复用，避免重复 NNUE 前向。

哨兵值 `HASH_EVAL_NONE = -32768` 表示 TT 中无有效静态分。

### 迭代加深与 Aspiration Windows

`SearchMain` 从浅到深迭代：

1. 根节点初始化 NNUE 累加器与 `staticEval[0]`
2. 深度 `< ASPIRATION_MIN_DEPTH(4)`：全窗口 `[-MATE, +MATE]`
3. 更深时以上一轮分数为中心开窄窗口，失败则扩窗重搜（最多 `ASPIRATION_MAX_FAILS` 次）
4. `SearchUnique` 检测唯一着法时可提前终止

### 着法排序

`SortStruct` 分阶段输出：

1. TT 最佳着
2. Killer 着（两层）
3. 历史启发排序的其余着法

**将军局面**：预生成合法应着，按 TT / 历史分排序；仅一应着时标记 `singleReply`（不减深度）。

**惰性合法化**：排序与探测阶段只做 `LegalMove`，**不** `MakeMove`，避免无效分支触发 NNUE 增量。

### 已有搜索扩展（第一梯队之前）

| 技巧 | 条件摘要 |
|------|----------|
| **空着剪枝** | 非将军、PST 子力 `NullOkay()`、有攻击子；浅验证 |
| **LMR** | 深度 ≥3、非吃非将、着法序号 > 阈值；先减深再必要时重搜 |
| **IID** | 深度 ≥4、TT 无着；内部迭代加深以获 hash move |
| **Singular extension** | 深度 ≥6、TT 有着；其它着法均无法在降深窗口 refute 时延长 TT 着深度 |
| **QSearch** | 停着 + 吃子；MvvLva 排序 + 低价值吃子裁剪 |

---

## 评估与 NNUE 集成

加载 NNUE 时，搜索全程使用 **纯 NNUE 静态分**（`evaluate_vl` / `nnue_score_vl`）；未加载 NNUE 时使用 **纯 PST**（`Position::Evaluate()`）。二者不混合。

`XqwlEvalOrTt(ply, ttEval)` 优先使用 TT 中的 `sEval`；未命中再计算并写入 `staticEval[ply]`。

### 空着剪枝与 NNUE 分离

空着剪枝的 `NullOkay()` / `NullSafe()` 仅看 **PST 子力**（`vlWhite` / `vlBlack`），不用 NNUE。原因：Pass 局面无真实对弈语义，NNUE 静态分易失真。

---

## NNUE 推理（C++）

特征：**XQWL-PSQ**，1260 维 = 己方/对方 × 7 子力 × 90 格。结构见 [NNUE 管线总览](nnue-overview.md)。

### 单视角 FT 累加器

维护 `acc[2][L1]` 两个 int32 缓冲，但搜索树中 **每步只增量更新一个视角**：

- 根节点 `acc_init` 刷新 `acc[sdPlayer]`
- 每步 `acc_apply_move` 更新 **走子后视角** `persp = 1 - sdPlayer`（走子前）
- 评估时只读 `acc[sdPlayer]`，FC 前向 **只算一次**

`acc_valid[2]` 跟踪各视角是否与当前局面同步。若目标视角未初始化，先 `acc_refresh_perspective` 再应用列 delta；走子后标记对侧失效。`acc_undo` 回滚 delta 并恢复 valid 标记。

增量一致性：`engine.verify_nnue_incremental(pos)` 应返回 **0**（根 + 所有一步合法着）。

### 惰性 NNUE（搜索路径）

| 操作 | NNUE 行为 |
|------|-----------|
| `XqwlSearchMakeMove` | `acc_apply_move` + 棋盘 `MakeMove` |
| `XqwlSearchUndoMakeMove` | `acc_undo` + `UndoMakeMove` |
| 排序 / `LegalMove` 探测 | **不**触发 NNUE |
| 节点 eval | 纯 NNUE（或纯 PST）+ TT `sEval` 缓存 |

### SIMD 后端

运行时派发：Scalar / NEON / AVX2 / AVX512-VNNI。热路径：`acc_add_column`（FT 增量）、`dot_i8_u8`（FC int8 点积）。

---

## 动态 PST

`RefreshPst()`（搜索入口调用一次）按局面子力与过河攻击指数在 **XQWL 静态 `cucvlPiecePos`** 上做中残局混合，写入 `vlWhite` / `vlBlack`。

用途：

1. 无 NNUE 时的 **PST 评估**（`Evaluate()`）
2. 空着剪枝子力判断（`NullOkay()` / `NullSafe()`）

NNUE 加载后搜索评估 **不**读取 PST 分值；PST 与 NNUE 互不混合。

---

## 开局库

178 万条 `BOOK.DAT`（Chess98 openbook），Zobrist `dwLock1` 二分查找。命中后按权重随机抽着；可镜像局面查库。

与搜索关系：库着直接返回，不走 NNUE 搜索；纯搜索模式每步 `SearchMain` 完整迭代。

详见 [开局库](opening-book.md)。

---

## 训练标签与将杀处理

NNUE 监督标签来自 `xqwl_gen_nnue` 根搜索分。训练管线（`nnue_training/labels.py`）对 **\|vl\| ≥ 9800** 的将杀样本：

- 不参与 z-score 统计
- 不参与 loss
- quiet 指标单独统计

避免极端将杀分拉偏 NNUE 回归。部署模型见 `data/nnue_model/`。

---

## 源码索引

| 主题 | 文件 |
|------|------|
| 搜索 + 剪枝 | `cpp/inc/xqwl_search.inc` |
| NNUE 推理 | `cpp/inc/xqwl_nnue.inc` |
| SIMD | `cpp/include/xqwl_nnue_simd.h`, `xqwl_nnue_simd_vnni.h` |
| 常量 | `cpp/include/xqwl_portable_prefix.h` |
| PST | `cpp/inc/xqwl_preeval.inc` |
| Python 绑定 | `cpp/bindings.cpp` |
| 特征定义 | `nnue_training/features/xqwl_psq.py` |

## 相关文档

- [NNUE 管线总览](nnue-overview.md) — 数据 → 训练 → 量化流程
- [INT8 量化与 C++ 推理](nnue-int8-inference.md) — bin 格式、API、parity 测试
- [NNUE 架构调研（Pikafish）](nnue_pikafish_research.md) — 外部参考（HalfKA 等，**未启用**）
- [开发与限制](development.md) — 平台与提交前检查
