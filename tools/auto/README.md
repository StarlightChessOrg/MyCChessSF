# 相弈象棋自动对弈测试

通过 **Selenium + HTTP 桥**，让 **象眸 SF** 引擎在 [相弈象棋](https://play.xiangqi.com/) 上与指定等级的人机对战，用于强度回归与实战冒烟测试。

本目录移植自 [Chess98 `tools/auto`](https://github.com/StarlightChessOrg/Chess98)，协议与 Chess98 UI 模式（`:9494`）兼容；引擎侧由 `mycchess-xiangqi-bridge` 替代 `Chess98.exe`。

## 架构

```
象眸 SF (xqwlight_core + NNUE)
        ↕  HTTP :9494  （Chess98 兼容）
mycchess-xiangqi-bridge
        ↕  /computer、/move?playermove=
tools/auto/z.js  (Selenium)
        ↕  DOM 点击
play.xiangqi.com  （相弈指定等级 bot）
```

## 环境要求

| 组件 | 说明 |
|------|------|
| **Python** | 已安装 `mycchess-sf`，且编译好 `xqwlight_core` |
| **Node.js** | ≥ 18 |
| **Microsoft Edge** | Windows；需与 `selenium-webdriver` 驱动匹配 |
| **网络** | 可访问 `play.xiangqi.com` |

> 主仓库目标平台为 Linux；**相弈自动测试目前仅 Windows + Edge**（与原 Chess98 脚本一致）。桥接服务本身可在 WSL/Linux 运行，Selenium 端需在 Windows 执行。

## 快速开始

### 1. 启动引擎桥（终端 A）

**部署包**（推荐）：

```bash
./deployment/bin/mycchess-xiangqi-bridge --think-ms 1000
```

**开发树**（`pip install -e .` 后）：

```bash
mycchess-xiangqi-bridge --think-ms 1000
# 可选：--engine-color red|black  --no-book  --no-nnue-default
```

默认：红方引擎、开局库（若已 build）、NNUE（若存在 `deployment/model/`）。

### 2. 安装并启动 Selenium 桥（终端 B）

```powershell
cd deployment\tools\auto   # 或仓库 tools\auto
npm install
$env:OPPONENT_LEVEL = "9"   # 相弈人机等级 1–9
$env:USER_PROFILE_DIR = "$env:LOCALAPPDATA\Microsoft\Edge\User Data"
npm start
```

脚本会打开相弈「人机对战」，选择对应等级 bot，随后自动代引擎走子。

**注意**：当前 `z.js` 流程与 Chess98 一致，要求引擎执**红先行**（`mycchess-xiangqi-bridge` 默认 `--engine-color red`）。若引擎执黑，需等相弈 bot 先走子，脚本需另行扩展。

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `OPPONENT_LEVEL` | `9` | 相弈 bot 等级（1–9） |
| `BRIDGE_HOST` | `127.0.0.1` | 引擎桥地址 |
| `BRIDGE_PORT` | `9494` | 引擎桥端口 |
| `USER_PROFILE_DIR` | Edge 用户目录 | 浏览器 Profile（登录态） |
| `KILL_EDGE` | `0` | 设为 `1` 时启动前强杀 Edge（慎用） |

## HTTP 协议（与 Chess98 兼容）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/computer` | 引擎着法（Chess98 `move.id` 字符串） |
| GET | `/move?playermove=<4位>` | 相弈对手着法（y1x1y2x2） |
| GET | `/boardcode` | 棋盘编码（占位，可扩展） |
| GET | `/undo` | 悔棋两回合 |

着法编解码见 `mycchess_sf/bridge_codec.py`（4 位桥接码 ↔ ICCS）。

## 文件

| 文件 | 说明 |
|------|------|
| `z.js` | Selenium 主脚本（相弈网页 ↔ :9494） |
| `package.json` | Node 依赖 `selenium-webdriver` |

Python 桥接：`mycchess_sf/xiangqi_auto_bridge.py`  
编解码：`mycchess_sf/bridge_codec.py`

## 常见问题

- **桥接无着法**：确认 `xqwlight_core` 已编译、BOOK/NNUE 路径正确；看终端 `[bridge] engine ->` 日志。
- **网页着法失败**：相弈 DOM 变更会导致选择器失效，需更新 `z.js` 中 `#game-grid` 相关 CSS。
- **Edge 打不开**：检查 `USER_PROFILE_DIR`；勿与其他 Edge 实例共用 Profile，或关闭 `KILL_EDGE`。

## 与本地网页对弈的区别

| | `mycchess-play-web` | `tools/auto` |
|--|---------------------|----------------|
| 对手 | 本地引擎 / 人类 | 相弈网站 bot |
| 平台 | Linux 浏览器 | Windows + Edge + Selenium |
| 用途 | 日常对弈、演示 | 对外部 AI 等级线回归测试 |
