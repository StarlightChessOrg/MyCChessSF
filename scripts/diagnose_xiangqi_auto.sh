#!/usr/bin/env bash
# Diagnose 象眸 SF ↔ 相弈 auto-play stack (bridge :9494 + Selenium client).
set -euo pipefail
HOST="${BRIDGE_HOST:-127.0.0.1}"
PORT="${BRIDGE_PORT:-9494}"
BASE="http://${HOST}:${PORT}"

echo "=== 象眸 SF 相弈自动对弈诊断 ==="
echo ""
echo "[1] 桥接 HTTP (${BASE})"
if curl -sf --max-time 3 "${BASE}/computer" >/tmp/xq_bridge_computer.txt 2>/dev/null; then
  mv=$(tr -d '\r\n' </tmp/xq_bridge_computer.txt)
  echo "    OK  GET /computer -> '${mv}'"
  if [[ "$mv" == "null" || -z "$mv" ]]; then
    echo "    提示: 尚无引擎着法（正常；Selenium 客户端拉取后会更新）"
  else
    echo "    提示: 引擎已有待执行着法，等待 Selenium 轮询 /computer"
  fi
else
  echo "    FAIL 无法连接 ${BASE}"
  echo "    请先启动: ./deployment/bin/mycchess-xiangqi-bridge"
  exit 1
fi

echo ""
echo "[2] 架构说明（常见误解）"
echo "    mycchess-xiangqi-bridge 不会打开 play.xiangqi.com。"
echo "    它只在 :9494 等待 selenium_xiangqi_com_auto.js (Selenium) 来取着 / 回传对手着法。"
echo ""
echo "[3] 你必须再开 **Windows PowerShell** 终端运行 Selenium："
echo "    cd deployment\\tools\\auto    # 或仓库 tools\\auto"
echo "    npm install"
echo "    \$env:OPPONENT_LEVEL = \"9\""
echo "    \$env:BRIDGE_HOST = \"127.0.0.1\"   # WSL 桥接；连不上时见 README"
echo "    npm start"
echo ""
echo "[4] 相弈网址: https://play.xiangqi.com/computer"
echo "    Selenium 脚本会打开 Edge，访问 play.xiangqi.com/computer，选等级、执红、开始对局。"
echo ""
echo "[5] WSL 桥 + Windows Selenium 时若 Windows 连不上 127.0.0.1:9494："
echo "    在 WSL 执行: hostname -I | awk '{print \$1}'"
echo "    在 PowerShell: \$env:BRIDGE_HOST = \"<WSL的IP>\""
echo ""
