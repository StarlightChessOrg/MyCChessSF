# third_party

外部参考依赖，**不参与象眸 SF 编译**。

| 路径 | 来源 | 说明 |
|------|------|------|
| `Pikafish/` | [official-pikafish/Pikafish](https://github.com/official-pikafish/Pikafish)（git submodule） | NNUE / 搜索调研对照；见 [docs/nnue_pikafish_research.md](../docs/nnue_pikafish_research.md) |

当前锁定提交见主仓库中 `third_party/Pikafish` 的 gitlink（`git submodule status`）。

克隆本仓库后初始化子模块：

```bash
git submodule update --init --recursive
# 或首次克隆：git clone --recurse-submodules https://github.com/StarlightChessOrg/MyCChessSF.git
```

更新 Pikafish 到上游新提交（可选）：

```bash
cd third_party/Pikafish
git fetch origin
git checkout <commit-or-tag>
cd ../..
git add third_party/Pikafish
git commit -m "Bump Pikafish submodule"
```
