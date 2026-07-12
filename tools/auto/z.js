/**
 * 相弈象棋 (play.xiangqi.com) 自动对弈桥
 *
 * 依赖象眸 SF 桥接服务（Chess98 兼容 HTTP :9494）：
 *   mycchess-xiangqi-bridge --think-ms 1000
 *
 * 环境变量（可选）：
 *   OPPONENT_LEVEL      相弈人机等级 1–9（默认 9）
 *   BRIDGE_HOST         默认 127.0.0.1
 *   BRIDGE_PORT         默认 9494
 *   USER_PROFILE_DIR    Edge 用户数据目录（Windows）
 *   KILL_EDGE           设为 1 时启动前 taskkill msedge（默认 0）
 */

const OPPONENT_LEVEL = Math.max(1, Math.min(9, Number(process.env.OPPONENT_LEVEL || 9)))
const BRIDGE_HOST = process.env.BRIDGE_HOST || "127.0.0.1"
const BRIDGE_PORT = Number(process.env.BRIDGE_PORT || 9494)
const USER_PROFILE_DIR =
  process.env.USER_PROFILE_DIR ||
  `${process.env.LOCALAPPDATA || process.env.HOME}/Microsoft/Edge/User Data`
const KILL_EDGE = process.env.KILL_EDGE === "1"

const { Builder, By } = require("selenium-webdriver")
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
    http.get(`${BRIDGE_BASE}${path}`, (res) => {
      let data = ""
      res.on("data", (chunk) => {
        data += chunk
      })
      res.on("end", () => resolve(data.trim()))
    }).on("error", reject)
  })
}

function httpNotify(path) {
  return new Promise((resolve, reject) => {
    http
      .request(`${BRIDGE_BASE}${path}`, { method: "GET" }, (res) => {
        res.on("data", () => {})
        res.on("end", resolve)
      })
      .on("error", reject)
      .end()
  })
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
  const raw = await httpGet("/computer")
  const digits = raw.replace(/\D/g, "")
  const data = digits.length >= 4 ? digits.slice(-4) : digits
  if (data !== engineLastMove && data.length === 4 && data !== "0000") {
    console.log("[auto] 引擎着法", data)
    engineLastMove = data
    await doMoveOnWeb(driver)

    const waitBoardChange = async () => {
      const currentBoard = await getWebBoard(driver)
      if (currentBoard.toString() !== webLastBoard.toString()) return
      await driver.sleep(200)
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
    await driver.sleep(200)
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
    await driver.sleep(400)
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
  await new Promise((r) => setTimeout(r, 300))
}

async function initDriver() {
  const options = new edge.Options()
  options.addArguments(
    `--user-data-dir=${USER_PROFILE_DIR}`,
    "--profile-directory=Default",
    "--log-level=3"
  )
  return new Builder().forBrowser("MicrosoftEdge").setEdgeOptions(options).build()
}

async function openXiangqiGame(driver) {
  await driver.get("https://play.xiangqi.com/")
  await driver.findElement(By.css("div.btn-list > div:nth-child(2)")).click()
  await driver.findElement(By.css(`.all-bots :nth-child(${OPPONENT_LEVEL})`)).click()
  await driver.findElement(By.css(".button-wrapper button:nth-child(1)")).click()
  const waitLoading = async () => {
    try {
      await driver.findElement(By.css(".body"))
      await driver.sleep(200)
      await waitLoading()
    } catch {
      return
    }
  }
  await waitLoading()
  await driver.sleep(500)
}

async function run() {
  const startDriver = async () => {
    console.log(`[auto] 相弈等级=${OPPONENT_LEVEL}  bridge=${BRIDGE_BASE}`)
    const driver = await initDriver()
    try {
      await openXiangqiGame(driver)
    } catch (err) {
      console.error("[auto] 无法打开相弈象棋", err)
      await driver.quit()
      process.exit(1)
    }
    while (true) {
      const prev = state
      await getEngineMove(driver)
      while (state === prev) {
        await driver.sleep(200)
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
