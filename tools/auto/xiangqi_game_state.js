/**
 * 从 play.xiangqi.com 人机局读取权威局面。
 *
 * 重要：localStorage['xiangqi.botGameState'] 仅在对局 END 时写入；
 * 进行中的局面在 React 组件 state（gamePlayData）里，需走 Fiber 扫描。
 */

const BOT_STATE_KEY = "xiangqi.botGameState"

/** 在浏览器内执行（Selenium executeScript） */
function readBotGameStateInBrowser() {
  function normalizeSide(v) {
    if (v === "black" || v === "b" || v === 2) return "b"
    return "w"
  }

  function packState(raw, source) {
    const gp = raw.gamePlayData
    if (!gp) return null
    const ranksRaw = gp.currentFen || gp.initFen
    if (!ranksRaw) return null

    const parts = String(ranksRaw).trim().split(/\s+/)
    const ranks = parts[0]
    let sideToMove = parts[1] === "b" ? "b" : parts[1] === "w" ? "w" : null
    if (!sideToMove) sideToMove = normalizeSide(gp.turnNow)

    const fen =
      parts.length >= 2 ? String(ranksRaw).trim() : `${ranks} ${sideToMove} - - 0 1`

    const moves = Array.isArray(gp.moves) ? gp.moves : []
    const uciList = moves
      .map((m) => (typeof m === "string" ? m : m && (m.uci || m.move)))
      .filter(Boolean)

    const playerSide = normalizeSide(raw.playerSide)

    return {
      fen,
      ranks,
      sideToMove,
      playerSide,
      moveCount: Number(gp.moveCount != null ? gp.moveCount : uciList.length),
      uciList,
      lastUci: uciList.length ? uciList[uciList.length - 1] : null,
      stateCode: Number(gp.state),
      source,
    }
  }

  function getReactFiber(dom) {
    if (!dom) return null
    const key = Object.keys(dom).find(
      (k) => k.startsWith("__reactFiber$") || k.startsWith("__reactInternalInstance$")
    )
    return key ? dom[key] : null
  }

  function stateFromHookChain(fiber) {
    let hook = fiber && fiber.memoizedState
    while (hook) {
      const st = hook.memoizedState
      if (st && typeof st === "object" && st.gamePlayData && st.gamePlayData.currentFen) {
        return st
      }
      hook = hook.next
    }
    return null
  }

  function readFromReactFiber() {
    const app = document.getElementById("app")
    const grid = document.getElementById("game-grid")
    const seeds = []
    if (app) seeds.push(getReactFiber(app))
    if (grid) {
      let el = grid
      while (el) {
        const f = getReactFiber(el)
        if (f) seeds.push(f)
        el = el.parentElement
      }
    }

    const seen = new Set()
    const queue = seeds.filter(Boolean)
    let steps = 0
    while (queue.length && steps < 8000) {
      steps++
      const fiber = queue.shift()
      if (!fiber || seen.has(fiber)) continue
      seen.add(fiber)

      const st = stateFromHookChain(fiber)
      if (st) return st

      const sn = fiber.stateNode
      if (sn) {
        if (sn.state && sn.state.gamePlayData) return sn.state
        if (sn.gamePlayData) return sn
      }

      if (fiber.child) queue.push(fiber.child)
      if (fiber.sibling) queue.push(fiber.sibling)
      if (fiber.return) queue.push(fiber.return)
    }
    return null
  }

  function readFromLocalStorage() {
    try {
      const raw = localStorage.getItem("xiangqi.botGameState")
      if (!raw) return null
      return JSON.parse(raw)
    } catch {
      return null
    }
  }

  const reactRaw = readFromReactFiber()
  if (reactRaw) {
    const packed = packState(reactRaw, "react")
    if (packed) return packed
  }

  const lsRaw = readFromLocalStorage()
  if (lsRaw) {
    const packed = packState(lsRaw, "localStorage")
    if (packed) return packed
  }

  return null
}

function parseBridge4(token) {
  const d = String(token).replace(/\D/g, "").slice(-4)
  return {
    y1: Number(d.charAt(0)),
    x1: Number(d.charAt(1)),
    y2: Number(d.charAt(2)),
    x2: Number(d.charAt(3)),
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
  parseBridge4,
  uciToSquares,
  iccsToUci,
  bridgeTokenToUci,
  uciToSquareClass,
}
