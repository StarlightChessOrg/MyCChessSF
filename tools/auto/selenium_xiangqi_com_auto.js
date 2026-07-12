/**
 * 相弈象棋 (play.xiangqi.com) 自动对弈桥
 *
 * 依赖象眸 SF 桥接服务（Chess98 兼容 HTTP :9494）：
 *   mycchess-xiangqi-bridge --think-ms 1000
 *
 * 局面来源：localStorage['xiangqi.botGameState']（网页人机局权威状态，含 currentFen / moves[].uci）
 * 不再从 DOM 格子 diff 猜测对手着法。
 *
 * 环境变量（可选）：
 *   XIANGQI_URL         默认 https://play.xiangqi.com/computer
 *   OPPONENT_LEVEL      相弈人机等级 1–9（默认 9）
 *   BRIDGE_HOST         默认 127.0.0.1（WSL 桥接失败时改为 WSL IP）
 *   BRIDGE_PORT         默认 9494
 *   USER_PROFILE_DIR    Edge 用户数据目录（默认独立目录，避免与已打开的 Edge 冲突）
 *   KILL_EDGE           设为 1 时启动前 taskkill msedge（默认 0）
 */

const path = require("path")
const fs = require("fs")

const OPPONENT_LEVEL = Math.max(1, Math.min(9, Number(process.env.OPPONENT_LEVEL || 9)))
const XIANGQI_URL = process.env.XIANGQI_URL || "https://play.xiangqi.com/computer"
const BRIDGE_HOST = process.env.BRIDGE_HOST || "127.0.0.1"
const BRIDGE_PORT = Number(process.env.BRIDGE_PORT || 9494)
const DEFAULT_PROFILE_DIR = path.join(
  process.env.LOCALAPPDATA ||
    path.join(process.env.USERPROFILE || process.env.HOME || ".", "AppData", "Local"),
  "MyCChessSF",
  "edge-selenium-profile"
)
const USER_PROFILE_DIR = process.env.USER_PROFILE_DIR || DEFAULT_PROFILE_DIR
const USE_SYSTEM_EDGE_PROFILE =
  process.env.USER_PROFILE_DIR &&
  /[\\/]Microsoft[\\/]Edge[\\/]User Data$/i.test(process.env.USER_PROFILE_DIR)
const KILL_EDGE = process.env.KILL_EDGE === "1"
const HTTP_TIMEOUT_MS = Number(process.env.BRIDGE_HTTP_TIMEOUT || 15000)

const { Builder, By, until } = require("selenium-webdriver")
const edge = require("selenium-webdriver/edge")
const http = require("http")
const { exec } = require("child_process")
const {
  readBotGameState,
  bridgeTokenToUci,
  uciToSquares,
  uciToSquareClass,
} = require("./xiangqi_game_state")
const { parseBridge4 } = require("./xiangqi_web_fen")

const BRIDGE_BASE = `http://${BRIDGE_HOST}:${BRIDGE_PORT}`

async function syncBridgeFen(fen) {
  const body = await httpGet(`/sync?fen=${encodeURIComponent(fen)}`)
  return body.trim()
}

async function findSquareByClass(driver, sqClass) {
  const xp = `//*[@id='game-grid']//*[contains(@class,'square') and contains(concat(' ', normalize-space(@class), ' '), ' ${sqClass} ')]`
  return driver.findElement(By.xpath(xp))
}

async function findPieceWrapperAtSquare(driver, sqClass) {
  const sq = await findSquareByClass(driver, sqClass)
  const sr = await sq.getRect()
  const sx = sr.x + sr.width / 2
  const sy = sr.y + sr.height / 2
  const wrappers = await driver.findElements(
    By.css('#game-grid .pieces-container [class*="PieceWrapper"]')
  )
  let best = null
  let bestD = Infinity
  for (const w of wrappers) {
    const r = await w.getRect()
    const d = (r.x + r.width / 2 - sx) ** 2 + (r.y + r.height / 2 - sy) ** 2
    if (d < bestD) {
      bestD = d
      best = w
    }
  }
  if (best && bestD < (sr.width * sr.width) / 2) return best
  return null
}

async function playMoveOnWebSquares(driver, token, userSide) {
  const { x1, y1, x2, y2 } = parseBridge4(token)
  const uci = bridgeTokenToUci(token)
  const fromClass = uciToSquareClassFromUci(uci, true, userSide)
  const toClass = uciToSquareClassFromUci(uci, false, userSide)
  console.log(
    `[auto] 网页走子 ICCS (${x1},${y1})->(${x2},${y2}) UCI=${uci} square ${fromClass}->${toClass} bridge=${token}`
  )

  const fromPiece = await findPieceWrapperAtSquare(driver, fromClass)
  const toSquare = await findSquareByClass(driver, toClass)
  if (!fromPiece || !toSquare) {
    throw new Error(`定位失败 from=${fromClass} to=${toClass}`)
  }

  await focusGameWindow(driver)
  const dragEl = fromPiece
  const fr = await dragEl.getRect()
  const tr = await toSquare.getRect()
  const fx = fr.x + fr.width / 2
  const fy = fr.y + fr.height / 2
  const tx = tr.x + tr.width / 2
  const ty = tr.y + tr.height / 2

  await driver.executeScript(
    `const [dragEl, toSquare, fx, fy, tx, ty] = arguments;
     const dt = new DataTransfer();
     try { dt.effectAllowed = 'move'; } catch {}
     const fire = (el, type, x, y) => {
       const ev = new DragEvent(type, {
         bubbles: true, cancelable: true, composed: true, view: window,
         clientX: x, clientY: y, button: 0, dataTransfer: dt,
       });
       try { Object.defineProperty(ev, 'dataTransfer', { value: dt }); } catch {}
       el.dispatchEvent(ev);
     };
     const mouse = (el, type, x, y) => {
       el.dispatchEvent(new MouseEvent(type, {
         bubbles: true, cancelable: true, view: window, button: 0, clientX: x, clientY: y,
       }));
     };
     mouse(dragEl, 'mousedown', fx, fy);
     fire(dragEl, 'dragstart', fx, fy);
     const dropTarget = document.elementFromPoint(tx, ty) || toSquare;
     fire(dropTarget, 'dragenter', tx, ty);
     fire(dropTarget, 'dragover', tx, ty);
     fire(dropTarget, 'drop', tx, ty);
     fire(dragEl, 'dragend', tx, ty);
     mouse(dropTarget, 'mouseup', tx, ty);`,
    dragEl,
    toSquare,
    fx,
    fy,
    tx,
    ty
  )
  await sleep(500)
}

function uciToSquareClassFromUci(uci, isFrom, userSide) {
  const sq = uciToSquares(uci)
  if (!sq) throw new Error(`Invalid UCI ${uci}`)
  const { file, rank } = isFrom ? sq.from : sq.to
  return uciToSquareClass(file, rank, userSide)
}

async function botGameLoop(driver) {
  await isEndGame(driver)
  const gs = await readBotGameState(driver)
  if (!gs) return

  const sig = `${gs.fen}|${gs.moveCount}|${gs.uciList.join(",")}`
  if (lastGameState && lastGameState._sig === sig) return

  if (gs.moveCount === 0 && lastGameState && lastGameState.moveCount > 0) {
    console.log("[auto] 新对局")
    await resetBridgeSession()
  }

  console.log(
    `[auto] 局面 ply=${gs.moveCount} side=${gs.sideToMove} player=${gs.playerSide} last=${gs.lastUci || "-"}`
  )

  const moveToken = await syncBridgeFen(gs.fen)

  if (gs.sideToMove === "w" && moveToken && moveToken !== "null" && moveToken !== "____") {
    const token = moveToken.replace(/\D/g, "").slice(-4)
    if (token.length === 4 && token !== "0000" && token !== engineLastMove) {
      console.log("[auto] 引擎着法", token)
      engineLastMove = token
      try {
        await playMoveOnWebSquares(driver, token, gs.playerSide)
      } catch (err) {
        console.error("[auto] 网页走子失败:", err.message || err)
      }
    }
  }

  gs._sig = sig
  lastGameState = gs
  state++
  console.log("==========================")
}

let lastGameState = null
let engineLastMove = "0000"
let webLastBoard = [
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 1, 0],
  [1, 0, 1, 0, 1, 0, 1, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 0, 1, 0, 1, 0, 1, 0, 1],
  [0, 1, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 1, 1, 1, 1, 1, 1, 1, 1],
]
let state = 0

function httpGet(path, timeoutMs = HTTP_TIMEOUT_MS) {
  return new Promise((resolve, reject) => {
    const req = http.get(`${BRIDGE_BASE}${path}`, (res) => {
      let data = ""
      res.on("data", (chunk) => {
        data += chunk
      })
      res.on("end", () => resolve(data.trim()))
    })
    req.on("error", reject)
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`timeout GET ${path}`))
    })
  })
}

async function httpGetRetry(path, { retries = 3, timeoutMs = HTTP_TIMEOUT_MS } = {}) {
  let lastErr = null
  for (let i = 0; i < retries; i++) {
    try {
      return await httpGet(path, timeoutMs)
    } catch (err) {
      lastErr = err
      const msg = String(err.message || err)
      if (!/timeout|ECONNREFUSED|ECONNRESET|socket hang up/i.test(msg)) {
        throw err
      }
      await sleep(400 * (i + 1))
    }
  }
  throw lastErr
}

function httpNotify(path) {
  return new Promise((resolve, reject) => {
    const req = http
      .request(`${BRIDGE_BASE}${path}`, { method: "GET" }, (res) => {
        res.on("data", () => {})
        res.on("end", resolve)
      })
      .on("error", reject)
    req.setTimeout(HTTP_TIMEOUT_MS, () => {
      req.destroy(new Error(`timeout notify ${path}`))
    })
    req.end()
  })
}

async function pingBridge() {
  console.log(`[auto] 检测桥接 ${BRIDGE_BASE} ...`)
  try {
    const raw = await httpGet("/computer")
    console.log(`[auto] 桥接 OK  /computer -> '${raw}'`)
    if (raw !== "null" && raw !== "____" && /\d{4}/.test(raw)) {
      console.warn(
        "[auto] 桥接含上一局着法，将在网页开局后调用 /reset 同步"
      )
    }
    return true
  } catch (err) {
    console.error("[auto] 无法连接象眸桥接服务:", err.message || err)
    console.error("[auto] 请先在 WSL/Linux 启动: ./deployment/bin/mycchess-xiangqi-bridge")
    console.error("[auto] 若桥在 WSL、本脚本在 Windows，设置: set BRIDGE_HOST=<WSL的IP>")
    console.error("[auto] WSL IP: wsl hostname -I")
    return false
  }
}

async function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms))
}

async function clickFirst(driver, selectors, label) {
  for (const sel of selectors) {
    try {
      const el = await driver.findElement(By.css(sel))
      await el.click()
      console.log(`[auto] 点击 ${label}: ${sel}`)
      return true
    } catch {
      /* try next */
    }
  }
  return false
}

async function clickByXPath(driver, xpaths, label) {
  for (const xp of xpaths) {
    try {
      const el = await driver.findElement(By.xpath(xp))
      await el.click()
      console.log(`[auto] 点击 ${label}: ${xp}`)
      return true
    } catch {
      /* try next */
    }
  }
  return false
}

async function clickSideRed(driver) {
  return (
    (await clickByXPath(
      driver,
      [
        "//*[normalize-space(text())='Red']/ancestor-or-self::button[1]",
        "//*[normalize-space(text())='Red']/ancestor-or-self::*[@role='button'][1]",
        "//*[contains(@class,'side') and .//*[normalize-space(text())='Red']]",
        "//button[contains(.,'Red')]",
        "//*[self::button or self::div][normalize-space(text())='Red']",
      ],
      "执红"
    )) ||
    (await clickFirst(
      driver,
      ["[class*='side' i] [class*='red' i]", "[data-side='red']"],
      "执红"
    ))
  )
}

async function clickStartPlay(driver) {
  return (
    (await clickByXPath(
      driver,
      [
        "//*[normalize-space(text())='Play' and not(starts-with(normalize-space(.),'Play Computer'))]",
        "//button[contains(.,'Play')]",
        "//*[self::button or self::div or self::span][normalize-space(text())='Play']",
      ],
      "开始对局"
    )) ||
    (await clickFirst(
      driver,
      [
        ".button-wrapper button:nth-child(1)",
        "button[class*='play' i]",
        "[class*='play-button' i]",
      ],
      "开始对局"
    ))
  )
}

async function syncWebBoardFromPage(driver) {
  webLastBoard = await readWebBoard(driver)
  console.log(`[auto] 已从网页同步棋盘状态（${countOccupied(webLastBoard)} 个有子格）`)
}

function diffBoardMove(before, after) {
  const from = []
  const to = []
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 9; j++) {
      if (before[i][j] === 1 && after[i][j] === 0) from.push({ x1: i, y1: j })
      if (before[i][j] === 0 && after[i][j] === 1) to.push({ x2: i, y2: j })
    }
  }
  if (from.length === 1 && to.length === 1) {
    return { x1: from[0].x1, y1: from[0].y1, x2: to[0].x2, y2: to[0].y2 }
  }
  return null
}

function moveToBridgeToken(move) {
  return String(move.x1) + String(move.y1) + String(move.x2) + String(move.y2)
}

function moveToIccsLabel(move) {
  return `${move.y1}${move.x1}-${move.y2}${move.x2}`
}

function isValidMove(move) {
  return (
    move &&
    move.x1 >= 0 &&
    move.x1 <= 9 &&
    move.y1 >= 0 &&
    move.y1 <= 8 &&
    move.x2 >= 0 &&
    move.x2 <= 9 &&
    move.y2 >= 0 &&
    move.y2 <= 8 &&
    !(move.x1 === move.x2 && move.y1 === move.y2)
  )
}

async function waitForStableOpponentBoard(driver, sinceBoard, timeoutMs = 45000) {
  const since = sinceBoard.toString()
  const deadline = Date.now() + timeoutMs
  console.log("[auto] 等待相弈应手…")
  let candidate = null
  let candidateKey = ""
  let stableSince = 0
  while (Date.now() < deadline) {
    const cur = await readWebBoard(driver)
    const key = cur.toString()
    if (key === since) {
      candidate = null
      candidateKey = ""
      stableSince = 0
    } else if (key === candidateKey) {
      if (Date.now() - stableSince >= 700) return candidate
    } else {
      candidate = cur
      candidateKey = key
      stableSince = Date.now()
    }
    await sleep(200)
  }
  if (candidate) return candidate
  throw new Error("等待相弈应手超时")
}

async function resetBridgeSession() {
  console.log("[auto] 重置桥接局面（新对局）...")
  await httpNotify("/reset")
  engineLastMove = "0000"
  lastGameState = null
  await sleep(400)
}

async function waitForBridgeEngineMove(prevToken, timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    let raw
    try {
      raw = (await httpGetRetry("/computer", { retries: 2 })).trim()
    } catch (err) {
      console.warn("[auto] 桥接轮询失败，继续重试:", err.message || err)
      await sleep(1000)
      continue
    }
    if (raw === "null" || raw === "____") {
      await sleep(400)
      continue
    }
    const digits = raw.replace(/\D/g, "")
    const data = digits.length >= 4 ? digits.slice(-4) : digits
    if (data.length === 4 && data !== "0000" && data !== prevToken) {
      console.log(`[auto] 桥接下一着就绪 -> '${data}'`)
      return data
    }
    await sleep(400)
  }
  console.warn("[auto] 桥接下一着等待超时，继续轮询")
}

async function resolveOpponentMove(driver, before, after, pieceBefore, pieceAfter) {
  const boardMove = parseMoveFromBoardDiff(before, after)
  if (boardMove && !boardMove.capture && isValidMove(boardMove)) {
    console.log(
      `[auto] 解着(盘差) ICCS ${moveToIccsLabel(boardMove)} bridge ${moveToBridgeToken(boardMove)}`
    )
    return boardMove
  }

  const pieceMove = inferMoveFromPieceGrids(pieceBefore, pieceAfter)
  if (isValidMove(pieceMove)) {
    console.log(
      `[auto] 解着(棋子) ICCS ${moveToIccsLabel(pieceMove)} bridge ${moveToBridgeToken(pieceMove)}`
    )
    return pieceMove
  }

  if (boardMove && boardMove.capture) {
    const destFromPieces = inferMoveFromPieceGrids(pieceBefore, pieceAfter)
    if (isValidMove(destFromPieces)) {
      console.log(
        `[auto] 解着(吃子) ICCS ${moveToIccsLabel(destFromPieces)} bridge ${moveToBridgeToken(destFromPieces)}`
      )
      return destFromPieces
    }
  }

  const dom = await readLastMoveFromDOM(driver)
  if (dom && dom.from && dom.to) {
    const move = {
      x1: dom.from.row,
      y1: dom.from.col,
      x2: dom.to.row,
      y2: dom.to.col,
    }
    if (isValidMove(move)) {
      console.log(
        `[auto] 解着(DOM) ICCS ${moveToIccsLabel(move)} bridge ${moveToBridgeToken(move)}`
      )
      return move
    }
  }

  const legacy = await getWebMoveLegacy(driver, before, after, boardMove)
  if (isValidMove(legacy)) {
    console.log(
      `[auto] 解着 ICCS ${moveToIccsLabel(legacy)} bridge ${moveToBridgeToken(legacy)}`
    )
    return legacy
  }

  return null
}

async function findGridSquare(driver, row, col, preferPiece = true) {
  const rowSel = `#game-grid > div:nth-child(${row}) > div:nth-child(${col})`
  const variants = []
  if (row === 1) {
    if (preferPiece) variants.push(`${rowSel} div.square-has-piece`)
    variants.push(`${rowSel} div.square`, `${rowSel} > div:last-child`, `${rowSel} > div`)
  } else if (row === 10) {
    if (preferPiece) variants.push(`${rowSel} div.square-has-piece`)
    variants.push(`${rowSel} div.square`, `${rowSel} > div:first-child`, `${rowSel} > div`)
  } else {
    if (preferPiece) variants.push(`${rowSel} div.square-has-piece`, `${rowSel} div.square`)
    variants.push(`${rowSel} > div`, `${rowSel} div`)
  }
  for (const sel of variants) {
    try {
      const el = await driver.findElement(By.css(sel))
      if (await el.isDisplayed()) return el
    } catch {
      /* try next */
    }
  }
  throw new Error(`找不到格子 row=${row} col=${col}`)
}

async function focusGameWindow(driver) {
  await driver.executeScript("window.focus();")
  try {
    const board = await driver.findElement(By.css("#game-grid"))
    await board.click()
  } catch {
    /* optional focus target */
  }
  await sleep(200)
}

async function listPiecesDebug(driver) {
  try {
    return await driver.executeScript(() => {
      const out = []
      document.querySelectorAll(".pieces-container [r]").forEach((el) => {
        out.push({
          r: el.getAttribute("r"),
          c: el.getAttribute("c"),
          f: el.getAttribute("f"),
          cls: el.className,
        })
      })
      return out.slice(0, 40)
    })
  } catch {
    return []
  }
}

async function getGridMetrics(driver) {
  const grid = await driver.findElement(By.css("#game-grid"))
  const rect = await grid.getRect()
  return { rect, cellW: rect.width / 9, cellH: rect.height / 10 }
}

async function rectToGridCell(_driver, el, metrics) {
  const box = await el.getRect()
  const cx = box.x + box.width / 2 - metrics.rect.x
  const cy = box.y + box.height / 2 - metrics.rect.y
  const col = Math.max(1, Math.min(9, Math.floor(cx / metrics.cellW) + 1))
  const row = Math.max(1, Math.min(10, Math.floor(cy / metrics.cellH) + 1))
  return { row, col }
}

function countOccupied(board) {
  let n = 0
  for (const row of board) for (const v of row) if (v === 1) n++
  return n
}

async function snapshotPieceGrid(driver) {
  let metrics
  try {
    metrics = await getGridMetrics(driver)
  } catch {
    return []
  }
  const elements = await driver.findElements(By.css(".pieces-container [r]"))
  const cells = []
  for (const el of elements) {
    const { row, col } = await rectToGridCell(driver, el, metrics)
    cells.push([row - 1, col - 1])
  }
  cells.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  return cells
}

function pickClosest(fromList, toList) {
  if (fromList.length === 0 || toList.length === 0) return null
  let bestFrom = fromList[0]
  let bestTo = toList[0]
  let bestDist = Infinity
  for (const [fx, fy] of fromList) {
    for (const [tx, ty] of toList) {
      const d = Math.abs(fx - tx) + Math.abs(fy - ty)
      if (d > 0 && d < bestDist) {
        bestDist = d
        bestFrom = [fx, fy]
        bestTo = [tx, ty]
      }
    }
  }
  if (bestDist === Infinity) return null
  return { x1: bestFrom[0], y1: bestFrom[1], x2: bestTo[0], y2: bestTo[1] }
}

function inferMoveFromPieceGrids(pieceBefore, pieceAfter) {
  const key = (r, c) => `${r},${c}`
  const toMap = (cells) => {
    const m = new Map()
    for (const [r, c] of cells) {
      const k = key(r, c)
      m.set(k, (m.get(k) || 0) + 1)
    }
    return m
  }
  const bMap = toMap(pieceBefore)
  const aMap = toMap(pieceAfter)
  const from = []
  const to = []
  for (const [k, n] of bMap) {
    const diff = n - (aMap.get(k) || 0)
    for (let i = 0; i < diff; i++) from.push(k.split(",").map(Number))
  }
  for (const [k, n] of aMap) {
    const diff = n - (bMap.get(k) || 0)
    for (let i = 0; i < diff; i++) to.push(k.split(",").map(Number))
  }
  if (from.length === 1 && to.length === 1) {
    return { x1: from[0][0], y1: from[0][1], x2: to[0][0], y2: to[0][1] }
  }
  if (from.length >= 1 && to.length >= 1) {
    return pickClosest(from, to)
  }
  // 吃子：仅起点变空，终点仍显示有子
  if (from.length === 1 && to.length === 0 && pieceAfter.length === pieceBefore.length - 1) {
    const [fx, fy] = from[0]
    const afterSet = new Set(pieceAfter.map(([r, c]) => key(r, c)))
    const beforeSet = new Set(pieceBefore.map(([r, c]) => key(r, c)))
    const added = [...afterSet].filter((k) => !beforeSet.has(k))
    if (added.length === 1) {
      const [tr, tc] = added[0].split(",").map(Number)
      return { x1: fx, y1: fy, x2: tr, y2: tc }
    }
    const candidates = pieceAfter.filter(([r, c]) => !(r === fx && c === fy))
    if (candidates.length === 1) {
      return { x1: fx, y1: fy, x2: candidates[0][0], y2: candidates[0][1] }
    }
    const near = pickClosest(
      [from[0]],
      candidates.filter(([r, c]) => Math.abs(r - fx) + Math.abs(c - fy) <= 4)
    )
    if (near) return near
  }
  return null
}

function parseMoveFromBoardDiff(lastBoard, currentBoard) {
  const from = []
  const to = []
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 9; j++) {
      if (lastBoard[i][j] === 1 && currentBoard[i][j] === 0) from.push([i, j])
      if (lastBoard[i][j] === 0 && currentBoard[i][j] === 1) to.push([i, j])
    }
  }
  if (from.length === 1 && to.length === 1) {
    return { x1: from[0][0], y1: from[0][1], x2: to[0][0], y2: to[0][1] }
  }
  if (from.length >= 1 && to.length >= 1) {
    return pickClosest(from, to)
  }
  if (from.length === 1 && to.length === 0) {
    return { x1: from[0][0], y1: from[0][1], x2: -1, y2: -1, capture: true }
  }
  return null
}

async function readLastMoveFromDOM(driver) {
  try {
    return await driver.executeScript(() => {
      const out = { from: null, to: null, marks: [] }
      const grid = document.querySelector("#game-grid")
      if (!grid) return out
      for (let ri = 0; ri < grid.children.length; ri++) {
        const row = grid.children[ri]
        for (let ci = 0; ci < row.children.length; ci++) {
          const sq = row.children[ci]
          const blob =
            (sq.className || "") +
            " " +
            [...sq.querySelectorAll("*")]
              .slice(0, 12)
              .map((el) => el.className || "")
              .join(" ")
          if (/last.?move|move.?from|move.?to|move.?marker|prev.?move|dest.?square|from.?square|to.?square|moved.?piece|move.?indicator|move.?highlight|last.?from|last.?to/i.test(blob)) {
            const isFrom = /from|prev|source|start/i.test(blob) && !/to|dest|target/i.test(blob)
            const isTo = /to|dest|target|moved|end/i.test(blob) && !/from|prev|source/i.test(blob)
            out.marks.push({ row: ri, col: ci, isFrom, isTo, cls: blob.slice(0, 80) })
            if (isFrom && !out.from) out.from = { row: ri, col: ci }
            if (isTo && !out.to) out.to = { row: ri, col: ci }
            if (!isFrom && !isTo && !out.to) out.to = { row: ri, col: ci }
            if (!isFrom && !isTo && !out.from) out.from = { row: ri, col: ci }
          }
        }
      }
      return out
    })
  } catch {
    return null
  }
}

async function findPieceAtGrid(driver, gridRow, gridCol) {
  let metrics
  try {
    metrics = await getGridMetrics(driver)
  } catch {
    return null
  }
  const targetX = metrics.rect.x + (gridCol - 0.5) * metrics.cellW
  const targetY = metrics.rect.y + (gridRow - 0.5) * metrics.cellH
  const pieces = await driver.findElements(By.css(".pieces-container [r]"))
  let best = null
  let bestDist = Infinity
  for (const el of pieces) {
    const box = await el.getRect()
    const cx = box.x + box.width / 2
    const cy = box.y + box.height / 2
    const dist = (cx - targetX) ** 2 + (cy - targetY) ** 2
    if (dist < bestDist) {
      bestDist = dist
      best = el
    }
  }
  const maxDist = metrics.cellW ** 2 + metrics.cellH ** 2
  if (best && bestDist <= maxDist * 2) {
    console.log(`[auto] 棋子(屏幕坐标) grid (${gridRow},${gridCol})`)
    return best
  }
  return null
}

async function performMoveDrag(driver, start, end) {
  for (const el of [start, end]) {
    await driver.executeScript(
      "arguments[0].scrollIntoView({block:'center',inline:'center'});",
      el
    )
  }
  await sleep(150)
  await driver
    .actions()
    .move({ origin: start, x: 0, y: 0 })
    .press()
    .pause(300)
    .move({ origin: end, x: 0, y: 0 })
    .pause(200)
    .release()
    .perform()
}

async function performMoveClick(driver, start, end) {
  for (const el of [start, end]) {
    await driver.executeScript(
      "arguments[0].scrollIntoView({block:'center',inline:'center'});",
      el
    )
  }
  await sleep(150)
  await dispatchPointerClick(driver, start)
  await sleep(350)
  await dispatchPointerClick(driver, end)
}

async function dispatchPointerClick(driver, el) {
  try {
    await driver.actions().move({ origin: el, x: 0, y: 0 }).click().perform()
    return
  } catch {
    /* fallback below */
  }
  await driver.executeScript(
    `const el = arguments[0];
     const box = el.getBoundingClientRect();
     const opts = {
       bubbles: true, cancelable: true, view: window,
       clientX: box.left + box.width / 2,
       clientY: box.top + box.height / 2,
     };
     for (const t of ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click']) {
       el.dispatchEvent(new MouseEvent(t, opts));
     }`,
    el
  )
}

/** bridge4 y1x1y2x2（ICCS 的 y=行、x=列）→ 相弈 DOM */
function bridgeToSite(iccsX1, iccsY1, iccsX2, iccsY2) {
  const fromRow = iccsY1 + 1
  const fromCol = iccsX1 + 1
  const toRow = iccsY2 + 1
  const toCol = iccsX2 + 1
  return {
    from: {
      iccsX: iccsX1,
      iccsY: iccsY1,
      gridRow: fromRow,
      gridCol: fromCol,
      r: 11 - fromRow,
      c: fromCol,
    },
    to: {
      iccsX: iccsX2,
      iccsY: iccsY2,
      gridRow: toRow,
      gridCol: toCol,
      r: 11 - toRow,
      c: toCol,
    },
  }
}

async function findPieceAtSite(driver, r, c) {
  const xpaths = [
    `//div[contains(@class,'pieces-container')]//div[@r='${r}' and @c='${c}']`,
    `//div[contains(@class,'pieces-container')]//div[@r='${r}' and @c='${c}']//div[contains(@class,'piece')]`,
    `//div[contains(@class,'pieces-container')]//*[@r='${r}' and @c='${c}']`,
  ]
  for (const xp of xpaths) {
    try {
      const el = await driver.findElement(By.xpath(xp))
      if (await el.isDisplayed()) {
        console.log(`[auto] 棋子 r=${r} c=${c}`)
        return el
      }
    } catch {
      /* try next */
    }
  }
  return null
}

async function findMoveTarget(driver, site, preferPiece) {
  if (preferPiece) {
    const byRc = await findPieceAtSite(driver, site.r, site.c)
    if (byRc) return byRc
    const byGrid = await findPieceAtGrid(driver, site.gridRow, site.gridCol)
    if (byGrid) return byGrid
  }
  return findGridSquare(driver, site.gridRow, site.gridCol, preferPiece)
}

async function waitForBoardDiff(driver, beforeBoard, timeoutMs = 4000) {
  const since = beforeBoard.toString()
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const cur = await readWebBoard(driver)
    if (cur.toString() !== since) return cur
    await sleep(200)
  }
  return readWebBoard(driver)
}

async function detectMoveApplied(driver, beforeBoard, toSite) {
  const afterBoard = await waitForBoardDiff(driver, beforeBoard, 5000)
  if (afterBoard.toString() !== beforeBoard.toString()) return afterBoard

  // 相弈棋子无 c 属性；若 #game-grid 读盘滞后，检查目标格是否已有子
  try {
    await driver.findElement(
      By.css(
        `#game-grid > div:nth-child(${toSite.gridRow}) > div:nth-child(${toSite.gridCol}) > div.square-has-piece`
      )
    )
    if (beforeBoard[toSite.gridRow - 1][toSite.gridCol - 1] === 0) {
      console.log("[auto] 目标格出现棋子，判定走子成功")
      return readWebBoard(driver)
    }
  } catch {
    /* not yet */
  }
  return null
}

async function applyMoveOnPage(driver, start, end, attempt) {
  await focusGameWindow(driver)
  if (attempt % 2 === 0) {
    console.log("[auto] 走子方式: 拖拽")
    await performMoveDrag(driver, start, end)
  } else {
    console.log("[auto] 走子方式: 点选")
    await performMoveClick(driver, start, end)
  }
}

async function waitForComputerSetupUi(driver, timeoutMs = 90000) {
  console.log("[auto] 等待相弈 React UI（Edge 控制台 SSL 报错多为广告/统计，可忽略）...")
  await driver.wait(async () => {
    try {
      const body = await driver.findElement(By.css("body")).getText()
      return /Play Computer|Level \d+|(^|\s)Play(\s|$)/i.test(body)
    } catch {
      return false
    }
  }, timeoutMs, "相弈 computer 页 UI 未在时限内出现")
  console.log(`[auto] UI 就绪  title=${await driver.getTitle()}  url=${await driver.getCurrentUrl()}`)
}

async function selectBotLevel(driver, level) {
  const tag = `(Level ${level})`
  const botXpaths = [
    `//*[contains(normalize-space(.),'${tag}') and not(self::script)]`,
    `//*[contains(@class,'bot') and contains(.,'Level ${level}')]`,
    `//*[contains(.,'${tag}') and not(self::script)]`,
  ]
  if (await clickByXPath(driver, botXpaths, `bot ${tag}`)) return true

  const botLegacy = `.all-bots :nth-child(${level})`
  if (await clickFirst(driver, [botLegacy], `bot L${level}`)) return true

  console.log(`[auto] 轮播切换到 ${tag} ...`)
  const nextSelectors = [
    "button[class*='next' i]",
    "[class*='carousel'] [class*='next' i]",
    "[class*='arrow'][class*='right' i]",
    "[aria-label*='next' i]",
  ]
  const nextXpaths = [
    "//button[contains(@class,'next')]",
    "//*[contains(@class,'arrow') and contains(@class,'right')]",
    "//button[contains(@class,'right') and .//*[name()='svg']]",
  ]
  for (let i = 0; i < 12; i++) {
    if (await clickByXPath(driver, botXpaths, `bot ${tag}`)) return true
    let advanced = await clickFirst(driver, nextSelectors, "bot-next")
    if (!advanced) advanced = await clickByXPath(driver, nextXpaths, "bot-next")
    if (!advanced) break
    await sleep(400)
  }
  return false
}

async function waitForBoard(driver, timeoutMs = 120000) {
  console.log("[auto] 等待棋盘 #game-grid ...")
  await driver.wait(until.elementLocated(By.css("#game-grid")), timeoutMs)
  await sleep(800)
}

/** 新版 https://play.xiangqi.com/computer */
async function openComputerPage(driver) {
  console.log(`[auto] 打开 ${XIANGQI_URL}`)
  try {
    await driver.get(XIANGQI_URL)
  } catch (err) {
    const msg = String(err.message || err)
    if (!/timeout|timed out/i.test(msg)) throw err
    console.warn("[auto] 页面加载超时（第三方资源 SSL 失败常见），继续等待 UI ...")
  }
  await waitForComputerSetupUi(driver)

  if (!(await selectBotLevel(driver, OPPONENT_LEVEL))) {
    console.warn(`[auto] 未找到等级 ${OPPONENT_LEVEL}，使用页面默认 bot`)
  }

  await sleep(400)
  if (!(await clickSideRed(driver))) {
    console.warn("[auto] 未点到 Red，可能默认已是红方")
  } else {
    await sleep(400)
  }

  await sleep(300)
  if (!(await clickStartPlay(driver))) {
    throw new Error("找不到 Play 按钮；相弈页面 DOM 可能已变更")
  }

  await waitForBoard(driver)
  await sleep(1200)
  await focusGameWindow(driver)
}

/** 旧版首页入口（回退） */
async function openLegacyHome(driver) {
  console.log("[auto] 回退旧版首页流程 play.xiangqi.com/")
  await driver.get("https://play.xiangqi.com/")
  await driver.findElement(By.css("div.btn-list > div:nth-child(2)")).click()
  await driver.findElement(By.css(`.all-bots :nth-child(${OPPONENT_LEVEL})`)).click()
  await driver.findElement(By.css(".button-wrapper button:nth-child(1)")).click()
  const waitLoading = async () => {
    try {
      await driver.findElement(By.css(".body"))
      await sleep(200)
      await waitLoading()
    } catch {
      return
    }
  }
  await waitLoading()
  await waitForBoard(driver)
  await sleep(1200)
  await focusGameWindow(driver)
}

async function openXiangqiGame(driver) {
  try {
    await openComputerPage(driver)
  } catch (err) {
    console.warn("[auto] /computer 流程失败:", err.message || err)
    await openLegacyHome(driver)
  }
  await syncWebBoardFromPage(driver)
  await resetBridgeSession()
  for (let i = 0; i < 30; i++) {
    lastGameState = await readBotGameState(driver)
    if (lastGameState) {
      console.log(`[auto] botGameState OK ply=${lastGameState.moveCount} fen=${lastGameState.fen}`)
      await syncBridgeFen(lastGameState.fen)
      break
    }
    await sleep(500)
  }
  if (!lastGameState) {
    console.error("[auto] 无法读取 localStorage.xiangqi.botGameState，请确认已进入人机对局")
  }
}

async function isEndGame(driver) {
  try {
    const elements = await driver.findElements(By.css(".game-end-widget"))
    if (elements.length > 0) {
      console.log("[auto] 对局结束")
      process.exit(0)
    }
  } catch (err) {
    if (/no such window|web view not found/i.test(String(err.message || err))) {
      throw err
    }
  }
}

async function getEngineMove(driver) {
  await isEndGame(driver)
  let raw
  try {
    raw = await httpGet("/computer")
  } catch (err) {
    console.error("[auto] 桥接请求失败:", err.message || err)
    await sleep(1000)
    return
  }
  const digits = raw.replace(/\D/g, "")
  const data = digits.length >= 4 ? digits.slice(-4) : digits
  if (data !== engineLastMove && data.length === 4 && data !== "0000") {
    console.log("[auto] 引擎着法", data)
    const playedToken = data
    engineLastMove = data
    try {
      await doMoveOnWeb(driver)
    } catch (err) {
      console.error("[auto] 网页走子放弃:", err.message || err)
      return
    }

    const boardAfterEngine = webLastBoard
    const pieceBefore = await snapshotPieceGrid(driver)
    let boardAfterOpponent
    try {
      boardAfterOpponent = await waitForStableOpponentBoard(driver, boardAfterEngine)
    } catch (err) {
      console.error("[auto]", err.message || err)
      webLastBoard = await readWebBoard(driver)
      return
    }

    let move = null
    for (let attempt = 0; attempt < 6; attempt++) {
      const pieceAfter = await snapshotPieceGrid(driver)
      const curBoard =
        attempt === 0 ? boardAfterOpponent : await readWebBoard(driver)
      move = await resolveOpponentMove(
        driver,
        boardAfterEngine,
        curBoard,
        pieceBefore,
        pieceAfter
      )
      if (isValidMove(move)) {
        boardAfterOpponent = curBoard
        break
      }
      if (attempt < 5) {
        console.warn(`[auto] 解着失败，${attempt + 1}/6 次重试…`)
        await sleep(700)
      }
    }

    if (!isValidMove(move)) {
      const dom = await readLastMoveFromDOM(driver)
      console.error(
        "[auto] 无法解析相弈应手",
        JSON.stringify({
          pieceBefore: pieceBefore.length,
          pieceAfter: (await snapshotPieceGrid(driver)).length,
          dom,
        })
      )
      webLastBoard = await readWebBoard(driver)
      return
    }
    console.log("[auto] 相弈应手", move)
    webLastBoard = boardAfterOpponent
    await sendOpponentMove(move)
    await waitForBridgeEngineMove(playedToken)
    state++
    console.log("==========================")
  } else {
    await sleep(200)
  }
}

async function doMoveOnWeb(driver, attempt = 0) {
  const MAX_ATTEMPTS = 8
  const { x1, y1, x2, y2 } = parseBridge4(engineLastMove)
  const { from, to } = bridgeToSite(x1, y1, x2, y2)

  console.log(
    `[auto] 网页走子 ICCS (${x1},${y1})->(${x2},${y2})  grid (${from.gridRow},${from.gridCol})->(${to.gridRow},${to.gridCol})  r/c (${from.r},${from.c})->(${to.r},${to.c})  bridge=${engineLastMove} attempt=${attempt + 1}`
  )

  if (attempt >= MAX_ATTEMPTS) {
    throw new Error(`网页走子失败已达 ${MAX_ATTEMPTS} 次: ${engineLastMove}`)
  }

  const beforeBoard = await readWebBoard(driver)

  let start
  let end
  try {
    start = await findMoveTarget(driver, from, true)
    end = await findMoveTarget(driver, to, false)
  } catch (err) {
    console.error("[auto] 定位格子失败:", err.message || err)
    await sleep(500)
    return doMoveOnWeb(driver, attempt + 1)
  }

  await driver.executeScript("arguments[0].style.outline='3px solid red';", start)
  await driver.executeScript("arguments[0].style.outline='3px solid orange';", end)

  try {
    await applyMoveOnPage(driver, start, end, attempt)
  } catch (err) {
    console.warn("[auto] 走子交互失败，尝试备用点击:", err.message || err)
    await dispatchPointerClick(driver, start)
    await sleep(300)
    await dispatchPointerClick(driver, end)
  }

  await driver.executeScript("arguments[0].style.outline='';", start)
  await driver.executeScript("arguments[0].style.outline='';", end)

  const afterBoard = await detectMoveApplied(driver, beforeBoard, to)
  if (!afterBoard) {
    const pieces = await listPiecesDebug(driver)
    if (pieces.length) console.warn("[auto] 当前棋子样本", JSON.stringify(pieces.slice(0, 8)))
    console.error("[auto] 网页着法失败，重试")
    await sleep(500)
    return doMoveOnWeb(driver, attempt + 1)
  }

  webLastBoard = afterBoard
  console.log("[auto] 网页着法成功，棋盘已同步")
}

async function getWebBoardFromPieces(driver) {
  let metrics
  try {
    metrics = await getGridMetrics(driver)
  } catch {
    return null
  }
  const pieces = await driver.findElements(By.css(".pieces-container [r]"))
  if (pieces.length === 0) return null
  const board = Array.from({ length: 10 }, () => Array(9).fill(0))
  for (const el of pieces) {
    const { row, col } = await rectToGridCell(driver, el, metrics)
    board[row - 1][col - 1] = 1
  }
  return board
}

async function readWebBoard(driver) {
  const pieceBoard = await getWebBoardFromPieces(driver)
  const gridBoard = await getWebBoard(driver)
  const pieceN = pieceBoard ? countOccupied(pieceBoard) : 0
  const gridN = countOccupied(gridBoard)
  if (pieceBoard && pieceN >= 16 && pieceN >= gridN) return pieceBoard
  if (gridN >= 16) return gridBoard
  if (pieceBoard && pieceN > gridN) return pieceBoard
  return gridBoard
}

async function getWebBoard(driver) {
  const currentBoard = []
  for (let i = 1; i <= 10; i++) {
    for (let j = 1; j <= 9; j++) {
      currentBoard[i - 1] = currentBoard[i - 1] || []
      try {
        await driver.findElement(
          By.css(`#game-grid > div:nth-child(${i}) > div:nth-child(${j}) > div.square-has-piece`)
        )
        currentBoard[i - 1][j - 1] = 1
      } catch {
        currentBoard[i - 1][j - 1] = 0
      }
    }
  }
  return currentBoard
}

async function getWebMoveLegacy(driver, lastBoard, currentBoard, boardMoveHint) {
  const strict = boardMoveHint || parseMoveFromBoardDiff(lastBoard, currentBoard)
  if (strict && !strict.capture && isValidMove(strict)) return strict

  const fromSquares = []
  const toSquares = []
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 9; j++) {
      if (lastBoard[i][j] === 1 && currentBoard[i][j] === 0) fromSquares.push({ x1: i, y1: j })
      if (lastBoard[i][j] === 0 && currentBoard[i][j] === 1) toSquares.push({ x2: i, y2: j })
    }
  }
  if (fromSquares.length === 0) return null

  let from = fromSquares[0]
  if (fromSquares.length > 1 && toSquares.length >= 1) {
    const picked = pickClosest(
      fromSquares.map((s) => [s.x1, s.y1]),
      toSquares.map((s) => [s.x2, s.y2])
    )
    if (picked) from = { x1: picked.x1, y1: picked.y1 }
  }

  const move = { x1: from.x1, y1: from.y1, x2: -1, y2: -1 }

  if (toSquares.length === 1) {
    move.x2 = toSquares[0].x2
    move.y2 = toSquares[0].y2
    return isValidMove(move) ? move : null
  }

  if (toSquares.length > 1) {
    const picked = pickClosest(
      [[move.x1, move.y1]],
      toSquares.map((s) => [s.x2, s.y2])
    )
    if (picked) {
      move.x2 = picked.x2
      move.y2 = picked.y2
      return isValidMove(move) ? move : null
    }
  }

  try {
    const metrics = await getGridMetrics(driver)
    const elements = await driver.findElements(By.css(".pieces-container [r]"))
    for (const el of elements) {
      try {
        const child = await el.findElement(By.css(":scope > div > div > div"))
        const cls = await child.getAttribute("class")
        if (!cls || !cls.includes("moved-piece")) continue
        const { row, col } = await rectToGridCell(driver, el, metrics)
        move.x2 = row - 1
        move.y2 = col - 1
        console.log(`[auto] moved-piece 落点 grid (${row},${col})`)
        return isValidMove(move) ? move : null
      } catch {
        continue
      }
    }
  } catch {
    /* ignore */
  }

  return null
}

async function sendOpponentMove(move) {
  const moveString = moveToBridgeToken(move)
  if (moveString.includes("-") || moveString.length !== 4) {
    console.error("[auto] 非法着法坐标", moveString)
    return false
  }
  console.log(`[auto] 回传桥接 ${moveString} (ICCS ${moveToIccsLabel(move)})`)
  await httpNotify(`/move?playermove=${moveString}`)
  await sleep(300)
  return true
}

async function initDriver() {
  fs.mkdirSync(USER_PROFILE_DIR, { recursive: true })
  const options = new edge.Options()
  options.setPageLoadStrategy("eager")
  const args = [
    `--user-data-dir=${USER_PROFILE_DIR}`,
    "--log-level=3",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "--remote-allow-origins=*",
    "--start-maximized",
    "--disable-session-crashed-bubble",
  ]
  if (USE_SYSTEM_EDGE_PROFILE) {
    args.push("--profile-directory=Default")
  }
  options.addArguments(...args)
  console.log(`[auto] Edge profile: ${USER_PROFILE_DIR}`)
  if (USE_SYSTEM_EDGE_PROFILE) {
    console.warn("[auto] 使用系统 Edge Profile；请先关闭所有 Edge 窗口，否则会启动失败")
  }
  try {
    const driver = await new Builder().forBrowser("MicrosoftEdge").setEdgeOptions(options).build()
    await driver.manage().setTimeouts({ pageLoad: 45000, script: 30000, implicit: 0 })
    return driver
  } catch (err) {
    const msg = String(err.message || err)
    if (/DevToolsActivePort|session not created|crashed/i.test(msg)) {
      console.error("[auto] Edge 启动失败。常见原因：")
      console.error("[auto]   1) 系统 Edge 已打开且占用了同一 Profile — 请全部关闭 Edge 后重试")
      console.error("[auto]   2) 不要设置 USER_PROFILE_DIR 为系统目录（默认已用独立 Profile）")
      console.error("[auto]   3) 或在 PowerShell: $env:KILL_EDGE='1'; npm start  （会强杀 Edge）")
    }
    throw err
  }
}

async function run() {
  const startDriver = async () => {
    if (!(await pingBridge())) {
      process.exit(1)
    }
    console.log(`[auto] 相弈等级=${OPPONENT_LEVEL}  url=${XIANGQI_URL}`)
    const driver = await initDriver()
    try {
      await openXiangqiGame(driver)
    } catch (err) {
      console.error("[auto] 无法打开相弈象棋:", err.message || err)
      await driver.quit()
      process.exit(1)
    }
    console.log("[auto] 对局已开始，轮询 botGameState（网页权威局面）…")
    while (true) {
      await botGameLoop(driver)
      await sleep(350)
    }
  }

  if (KILL_EDGE) {
    exec("taskkill /F /IM msedge.exe", () => startDriver())
  } else {
    await startDriver()
  }
}

run().catch((err) => {
  console.error("[auto] fatal", err)
  process.exit(1)
})
