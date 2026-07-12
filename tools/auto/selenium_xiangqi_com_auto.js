/**
 * 相弈象棋 (play.xiangqi.com) 自动对弈桥
 *
 * 依赖象眸 SF 桥接服务（Chess98 兼容 HTTP :9494）：
 *   mycchess-xiangqi-bridge --think-ms 1000
 *
 * **桥接不会打开浏览器**；本脚本 (Windows + Edge) 负责网页自动化。
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

const { Builder, By, until } = require("selenium-webdriver")
const edge = require("selenium-webdriver/edge")
const http = require("http")
const { exec } = require("child_process")

const BRIDGE_BASE = `http://${BRIDGE_HOST}:${BRIDGE_PORT}`

let engineLastMove = "null"
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

function httpGet(path) {
  return new Promise((resolve, reject) => {
    const req = http.get(`${BRIDGE_BASE}${path}`, (res) => {
      let data = ""
      res.on("data", (chunk) => {
        data += chunk
      })
      res.on("end", () => resolve(data.trim()))
    })
    req.on("error", reject)
    req.setTimeout(5000, () => {
      req.destroy(new Error(`timeout GET ${path}`))
    })
  })
}

function httpNotify(path) {
  return new Promise((resolve, reject) => {
    const req = http
      .request(`${BRIDGE_BASE}${path}`, { method: "GET" }, (res) => {
        res.on("data", () => {})
        res.on("end", resolve)
      })
      .on("error", reject)
    req.setTimeout(5000, () => {
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
  console.log("[auto] 已从网页同步棋盘状态")
}

function diffBoardMove(before, after) {
  let from = null
  let to = null
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < 9; j++) {
      if (before[i][j] === 1 && after[i][j] === 0) from = { x1: i, y1: j }
      if (before[i][j] === 0 && after[i][j] === 1) to = { x2: i, y2: j }
    }
  }
  if (!from || !to) return null
  return { x1: from.x1, y1: from.y1, x2: to.x2, y2: to.y2 }
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

async function waitForOpponentBoard(driver, sinceBoard, timeoutMs = 45000) {
  const since = sinceBoard.toString()
  const deadline = Date.now() + timeoutMs
  console.log("[auto] 等待相弈应手…")
  while (Date.now() < deadline) {
    const cur = await readWebBoard(driver)
    if (cur.toString() !== since) return cur
    await sleep(250)
  }
  throw new Error("等待相弈应手超时")
}

async function waitForBridgeEngineMove(prevToken, timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const raw = (await httpGet("/computer")).trim()
    if (raw === "null" || raw === "____") {
      await sleep(300)
      continue
    }
    const digits = raw.replace(/\D/g, "")
    const data = digits.length >= 4 ? digits.slice(-4) : digits
    if (data.length === 4 && data !== "0000" && data !== prevToken) {
      console.log(`[auto] 桥接下一着就绪 -> '${data}'`)
      return data
    }
    await sleep(300)
  }
  console.warn("[auto] 桥接下一着等待超时，继续轮询")
}

async function resolveOpponentMove(driver, before, after) {
  const diff = diffBoardMove(before, after)
  if (isValidMove(diff)) return diff
  const legacy = await getWebMove(driver, before, after)
  if (isValidMove(legacy)) return legacy
  return null
}

async function findGridSquare(driver, row, col, preferPiece = true) {
  const rowSel = `#game-grid > div:nth-child(${row}) > div:nth-child(${col})`
  const variants = []
  if (row === 1) {
    if (preferPiece) variants.push(`${rowSel} div.square-has-piece`)
    variants.push(`${rowSel} > div:last-child`, `${rowSel} > div`)
  } else if (row === 10) {
    if (preferPiece) variants.push(`${rowSel} div.square-has-piece`)
    variants.push(`${rowSel} > div:first-child`, `${rowSel} > div`)
  } else {
    if (preferPiece) variants.push(`${rowSel} div.square-has-piece`)
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
    const piece = await findPieceAtSite(driver, site.r, site.c)
    if (piece) return piece
  }
  return findGridSquare(driver, site.gridRow, site.gridCol, preferPiece)
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
  }

  await sleep(300)
  if (!(await clickStartPlay(driver))) {
    throw new Error("找不到 Play 按钮；相弈页面 DOM 可能已变更")
  }

  await waitForBoard(driver)
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
}

async function openXiangqiGame(driver) {
  try {
    await openComputerPage(driver)
  } catch (err) {
    console.warn("[auto] /computer 流程失败:", err.message || err)
    await openLegacyHome(driver)
  }
  await syncWebBoardFromPage(driver)
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
    let boardAfterOpponent
    try {
      boardAfterOpponent = await waitForOpponentBoard(driver, boardAfterEngine)
    } catch (err) {
      console.error("[auto]", err.message || err)
      webLastBoard = await readWebBoard(driver)
      return
    }

    const move = await resolveOpponentMove(driver, boardAfterEngine, boardAfterOpponent)
    if (!isValidMove(move)) {
      console.error("[auto] 无法解析相弈应手，重新同步棋盘", move)
      webLastBoard = boardAfterOpponent
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
  const iccsY1 = Number(engineLastMove.charAt(0))
  const iccsX1 = Number(engineLastMove.charAt(1))
  const iccsY2 = Number(engineLastMove.charAt(2))
  const iccsX2 = Number(engineLastMove.charAt(3))
  const { from, to } = bridgeToSite(iccsX1, iccsY1, iccsX2, iccsY2)

  console.log(
    `[auto] 网页走子 ICCS (${iccsX1},${iccsY1})->(${iccsX2},${iccsY2})  grid (${from.gridRow},${from.gridCol})->(${to.gridRow},${to.gridCol})  r/c (${from.r},${from.c})->(${to.r},${to.c})  bridge=${engineLastMove} attempt=${attempt + 1}`
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
    await performMoveClick(driver, start, end)
  } catch (err) {
    console.warn("[auto] 指针点击失败，尝试 JS click:", err.message || err)
    await dispatchPointerClick(driver, start)
    await sleep(250)
    await dispatchPointerClick(driver, end)
  }

  await driver.executeScript("arguments[0].style.outline='';", start)
  await driver.executeScript("arguments[0].style.outline='';", end)
  await sleep(600)

  const afterBoard = await readWebBoard(driver)
  if (afterBoard.toString() === beforeBoard.toString()) {
    console.error("[auto] 网页着法失败，重试")
    await sleep(400)
    return doMoveOnWeb(driver, attempt + 1)
  }

  webLastBoard = afterBoard
  console.log("[auto] 网页着法成功，棋盘已同步")
}

async function getWebBoardFromPieces(driver) {
  const board = Array.from({ length: 10 }, () => Array(9).fill(0))
  const pieces = await driver.findElements(
    By.css(".pieces-container > div[r], .pieces-container > div[data-r]")
  )
  if (pieces.length === 0) return null
  for (const el of pieces) {
    const r = Number((await el.getAttribute("r")) || (await el.getAttribute("data-r")))
    const c = Number((await el.getAttribute("c")) || (await el.getAttribute("data-c")))
    if (!Number.isFinite(r) || !Number.isFinite(c)) continue
    const gridRow = 11 - r
    if (gridRow >= 1 && gridRow <= 10 && c >= 1 && c <= 9) {
      board[gridRow - 1][c - 1] = 1
    }
  }
  return board
}

async function readWebBoard(driver) {
  const fromPieces = await getWebBoardFromPieces(driver)
  if (fromPieces) return fromPieces
  return getWebBoard(driver)
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

async function getWebMove(_driver, lastBoard, currentBoard) {
  const diff = diffBoardMove(lastBoard, currentBoard)
  if (isValidMove(diff)) return diff

  const move = { x1: -1, y1: -1, x2: -1, y2: -1 }
  for (let i = 0; i < lastBoard.length; i++) {
    for (let j = 0; j < lastBoard[i].length; j++) {
      if (lastBoard[i][j] === 1 && currentBoard[i][j] === 0) {
        move.x1 = i
        move.y1 = j
      }
    }
  }
  const elements = await _driver.findElements(By.css(".pieces-container > div[r]"))
  let moved = { x: -1, index: -1 }
  const pieces = []
  for (const el of elements) {
    let elChild
    try {
      elChild = await el.findElement(By.css(":scope > div > div > div"))
    } catch {
      continue
    }
    const rowNum = 11 - Number(await el.getAttribute("r"))
    pieces[rowNum] = pieces[rowNum] || []
    pieces[rowNum].push(el)
    const cls = await elChild.getAttribute("class")
    if (cls && cls.includes("moved-piece")) {
      moved.x = rowNum
      moved.index = pieces[rowNum].length - 1
    }
  }
  if (moved.x < 1) return move
  move.x2 = moved.x - 1
  let count = 0
  for (const k in currentBoard[moved.x - 1]) {
    if (currentBoard[moved.x - 1][k] === 1) {
      if (count === moved.index) {
        move.y2 = Number(k)
        break
      }
      count++
    }
  }
  return move
}

async function sendOpponentMove(move) {
  // move.x* = 网页行(0–9)=ICCS y；move.y* = 网页列(0–8)=ICCS x
  const moveString =
    String(move.x1) + String(move.y1) + String(move.x2) + String(move.y2)
  if (moveString.includes("-")) {
    console.error("[auto] 非法着法坐标", moveString)
    return
  }
  console.log("[auto] 回传桥接", moveString)
  await httpNotify(`/move?playermove=${moveString}`)
  await sleep(300)
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
    console.log("[auto] 对局已开始，轮询引擎着法…")
    while (true) {
      const prev = state
      await getEngineMove(driver)
      while (state === prev) {
        await sleep(200)
        await getEngineMove(driver)
      }
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
