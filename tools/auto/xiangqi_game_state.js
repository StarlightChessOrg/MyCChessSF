/**
 * 从 play.xiangqi.com 人机局读取权威局面（localStorage + gamePlayData）。
 * 源码键名见 main bundle: "xiangqi.botGameState"
 * Bot 着手经 Socket.IO bot.move / bot.move.success（chunk 4406）。
 */

const BOT_STATE_KEY = "xiangqi.botGameState"

/** 在浏览器内执行 */
function readBotGameStateInBrowser() {
  const raw = localStorage.getItem("xiangqi.botGameState")
  if (!raw) return null
  let state
  try {
    state = JSON.parse(raw)
  } catch {
    return null
  }
  const gp = state.gamePlayData
  if (!gp || !gp.currentFen) return null

  const moves = Array.isArray(gp.moves) ? gp.moves : []
  const uciList = moves.map((m) => (typeof m === "string" ? m : m && m.uci)).filter(Boolean)

  const parts = String(gp.currentFen).trim().split(/\s+/)
  const ranks = parts[0]
  let sideToMove = parts[1] === "b" ? "b" : parts[1] === "w" ? "w" : null
  if (!sideToMove) {
    sideToMove = gp.turnNow === "black" ? "b" : "w"
  }
  const fen = parts.length >= 2 ? String(gp.currentFen).trim() : `${ranks} ${sideToMove} - - 0 1`

  const playerSide = state.playerSide === "black" ? "b" : "w"

  return {
    fen,
    ranks,
    sideToMove,
    playerSide,
    moveCount: Number(gp.moveCount || uciList.length),
    uciList,
    lastUci: uciList.length ? uciList[uciList.length - 1] : null,
    stateCode: Number(gp.state),
  }
}

function uciToSquares(uci) {
  const m = String(uci).match(/^([a-i])(\d)([a-i])(\d)$/i)
  if (!m) return null
  return {
    from: { file: m[1].charCodeAt(0) - 97, rank: Number(m[2]) },
    to: { file: m[3].charCodeAt(0) - 97, rank: Number(m[4]) },
  }
}

/** ICCS (x,y) 0-index → 相弈 UCI（a-i + rank） */
function iccsToUci(x1, y1, x2, y2) {
  const f = (x) => String.fromCharCode(97 + x)
  return `${f(x1)}${y1}${f(x2)}${y2}`
}

function bridgeTokenToUci(token) {
  const d = String(token).replace(/\D/g, "").slice(-4)
  if (d.length < 4) return null
  const y1 = Number(d.charAt(0))
  const x1 = Number(d.charAt(1))
  const y2 = Number(d.charAt(2))
  const x2 = Number(d.charAt(3))
  return iccsToUci(x1, y1, x2, y2)
}

function uciToSquareClass(file, rank, userSide = "w") {
  let pageR
  let pageCol
  if (userSide === "b") {
    pageR = 10 - rank
    pageCol = String.fromCharCode("a".charCodeAt(0) + (8 - file))
  } else {
    pageR = rank + 1
    pageCol = String.fromCharCode("a".charCodeAt(0) + file)
  }
  return `${pageR}-${pageCol}`
}

async function readBotGameState(driver) {
  return driver.executeScript(readBotGameStateInBrowser)
}

module.exports = {
  BOT_STATE_KEY,
  readBotGameState,
  uciToSquares,
  iccsToUci,
  bridgeTokenToUci,
  uciToSquareClass,
}
