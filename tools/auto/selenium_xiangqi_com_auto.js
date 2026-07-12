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

async function waitForBoard(driver, timeoutMs = 120000) {
  console.log("[auto] 等待棋盘 #game-grid ...")
  await driver.wait(until.elementLocated(By.css("#game-grid")), timeoutMs)
  await sleep(800)
}

/** 新版 https://play.xiangqi.com/computer */
async function openComputerPage(driver) {
  console.log(`[auto] 打开 ${XIANGQI_URL}`)
  await driver.get(XIANGQI_URL)
  await sleep(1500)

  const botLegacy = `.all-bots :nth-child(${OPPONENT_LEVEL})`
  if (await clickFirst(driver, [botLegacy], `bot L${OPPONENT_LEVEL}`)) {
    /* legacy list */
  } else {
    const levelRe = `Level ${OPPONENT_LEVEL}`
    const okBot = await clickByXPath(
      driver,
      [
        `//*[contains(@class,'bot') and contains(.,'${levelRe}')]`,
        `//*[contains(.,'(${levelRe}') or contains(.,'${levelRe}')][not(self::script)]`,
      ],
      `bot ${levelRe}`
    )
    if (!okBot) {
      console.warn(`[auto] 未找到等级 ${OPPONENT_LEVEL}，使用页面默认 bot`)
    }
  }

  await sleep(400)
  const okRed = await clickByXPath(
    driver,
    [
      "//button[contains(.,'Red')]",
      "//*[self::button or self::div][normalize-space(text())='Red']",
      "//label[contains(.,'Red')]",
    ],
    "执红"
  )
  if (!okRed) {
    console.warn("[auto] 未点到 Red，可能默认已是红方")
  }

  await sleep(300)
  const okPlay = await clickByXPath(
    driver,
    [
      "//button[normalize-space()='Play']",
      "//button[contains(.,'Play')]",
      ".button-wrapper button:nth-child(1)",
    ],
    "开始对局"
  )
  if (!okPlay) {
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
}

async function isEndGame(driver) {
  const elements = await driver.findElements(By.css(".game-end-widget"))
  if (elements.length > 0) {
    console.log("[auto] 对局结束")
    process.exit(0)
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
    engineLastMove = data
    await doMoveOnWeb(driver)

    const waitBoardChange = async () => {
      const currentBoard = await getWebBoard(driver)
      if (currentBoard.toString() !== webLastBoard.toString()) return
      await sleep(200)
      await waitBoardChange()
    }
    await waitBoardChange()

    const currentBoard = await getWebBoard(driver)
    const move = await getWebMove(driver, webLastBoard, currentBoard)
    console.log("[auto] 相弈应手", move)
    webLastBoard = currentBoard
    await sendOpponentMove(move)
    state++
    console.log("==========================")
  } else {
    await sleep(200)
  }
}

async function doMoveOnWeb(driver) {
  const x1 = Number(engineLastMove.charAt(1))
  const y1 = Number(engineLastMove.charAt(0))
  const x2 = Number(engineLastMove.charAt(3))
  const y2 = Number(engineLastMove.charAt(2))
  const _x1 = 10 - x1
  const _y1 = y1 + 1
  const _x2 = 10 - x2
  const _y2 = y2 + 1

  webLastBoard[9 - x1][y1] = 0
  webLastBoard[9 - x2][y2] = 1

  let start = await driver.findElement(
    By.css(`#game-grid > div:nth-child(${_x1}) > div:nth-child(${_y1}) > div`)
  )
  let end = await driver.findElement(
    By.css(`#game-grid > div:nth-child(${_x2}) > div:nth-child(${_y2}) > div`)
  )
  if (_x1 === 1)
    start = await driver.findElement(
      By.css(`#game-grid > div:nth-child(${_x1}) > div:nth-child(${_y1}) > div:last-child`)
    )
  if (_x2 === 1)
    end = await driver.findElement(
      By.css(`#game-grid > div:nth-child(${_x2}) > div:nth-child(${_y2}) > div:last-child`)
    )
  if (_x1 === 10)
    start = await driver.findElement(
      By.css(`#game-grid > div:nth-child(${_x1}) > div:nth-child(${_y1}) > div:first-child`)
    )
  if (_x2 === 10)
    end = await driver.findElement(
      By.css(`#game-grid > div:nth-child(${_x2}) > div:nth-child(${_y2}) > div:first-child`)
    )

  await driver.executeScript("arguments[0].style.border='3px solid red';", start)
  await driver.executeScript("arguments[0].style.border='3px solid red';", end)

  const startRect = await start.getRect()
  const endRect = await end.getRect()
  const startX = Math.ceil(startRect.x + startRect.width / 2)
  const startY = Math.ceil(startRect.y + startRect.height / 2)
  const endX = Math.ceil(endRect.x + endRect.width / 2)
  const endY = Math.ceil(endRect.y + endRect.height / 2)
  await driver
    .actions({ bridge: true })
    .move({ x: startX, y: startY })
    .click()
    .move({ x: endX, y: endY, duration: 300 })
    .click()
    .perform()

  await driver.executeScript("arguments[0].style.border='';", start)
  await driver.executeScript("arguments[0].style.border='';", end)

  if ((await getWebBoard(driver)).toString() !== webLastBoard.toString()) {
    webLastBoard[9 - x1][y1] = 1
    webLastBoard[9 - x2][y2] = 0
    console.error("[auto] 网页着法失败，重试")
    await sleep(400)
    await doMoveOnWeb(driver)
  }
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
  const move = { x1: -1, y1: -1, x2: -1, y2: -1 }
  for (let i = 0; i < lastBoard.length; i++) {
    for (let j = 0; j < lastBoard[i].length; j++) {
      if (lastBoard[i][j] === 1 && currentBoard[i][j] === 0) {
        move.x1 = i
        move.y1 = j
      }
    }
  }
  const elements = await _driver.findElements(By.css(".pieces-container > div"))
  let moved = { x: -1, index: -1 }
  const pieces = []
  for (const el of elements) {
    const elChild = await el.findElement(By.css(".pieces-container > div > div > div"))
    const rowNum = 11 - Number(await el.getAttribute("r"))
    pieces[rowNum] = pieces[rowNum] || []
    pieces[rowNum].push(el)
    if ((await elChild.getAttribute("class")).match("moved-piece")) {
      moved.x = rowNum
      moved.index = pieces[rowNum].length - 1
    }
  }
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
  const x1 = String(9 - move.x1)
  const y1 = String(move.y1)
  const x2 = String(9 - move.x2)
  const y2 = String(move.y2)
  const moveString = y1 + x1 + y2 + x2
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
  const args = [
    `--user-data-dir=${USER_PROFILE_DIR}`,
    "--log-level=3",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "--remote-allow-origins=*",
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
    return await new Builder().forBrowser("MicrosoftEdge").setEdgeOptions(options).build()
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
