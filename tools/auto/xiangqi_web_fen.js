/**
 * 从 play.xiangqi.com 读取真实局面并转 FEN（棋子 img.alt + .square 格坐标）。
 * 思路参考开源 XiangqiAnalysisHelper，避免 occupancy diff 猜着法。
 */

const STARTPOS_RANKS =
  "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

/** 在浏览器内执行的读盘函数（executeScript 注入） */
function readBoardInBrowser() {
  const PIECE_MAP = {
    king: "k",
    advisor: "a",
    elephant: "b",
    horse: "n",
    rook: "r",
    cannon: "c",
    pawn: "p",
  }

  function parsePieceAlt(alt) {
    const m = alt && alt.match(/^(king|advisor|elephant|horse|rook|cannon|pawn)-(red|brown)-zh$/)
    if (!m) return null
    const ch = PIECE_MAP[m[1]]
    return m[2] === "red" ? ch.toUpperCase() : ch
  }

  function detectUserSide() {
    const pieces = document.querySelectorAll('#game-grid .pieces-container [class*="PieceWrapper"]')
    let bestY = -Infinity
    let bestAlt = null
    for (const wrap of pieces) {
      const img = wrap.querySelector("img.img-holder")
      if (!img || !img.alt || !img.alt.startsWith("king-")) continue
      const rect = wrap.getBoundingClientRect()
      if (rect.top > bestY) {
        bestY = rect.top
        bestAlt = img.alt
      }
    }
    if (bestAlt && bestAlt.startsWith("king-red-")) return "w"
    if (bestAlt && bestAlt.startsWith("king-brown-")) return "b"
    return "w"
  }

  const grid = document.querySelector("#game-grid")
  if (!grid) return null

  const squares = [...grid.querySelectorAll(".square")]
  if (squares.length !== 90) return null

  const squareCenters = squares
    .map((sq) => {
      const m = sq.className.match(/(\d+)-([a-i])/)
      if (!m) return null
      const pageR = parseInt(m[1], 10)
      const pageColIdx = m[2].charCodeAt(0) - "a".charCodeAt(0)
      const rect = sq.getBoundingClientRect()
      return {
        pageR,
        pageColIdx,
        cx: rect.left + rect.width / 2,
        cy: rect.top + rect.height / 2,
      }
    })
    .filter(Boolean)

  const userSide = detectUserSide()
  const board = Array.from({ length: 10 }, () => Array(9).fill(null))
  const pieces = grid.querySelectorAll('.pieces-container [class*="PieceWrapper"]')

  for (const wrap of pieces) {
    const img = wrap.querySelector("img.img-holder")
    const ch = parsePieceAlt(img && img.alt)
    if (!ch) continue
    const r = wrap.getBoundingClientRect()
    const px = r.left + r.width / 2
    const py = r.top + r.height / 2
    let best = null
    let bestD = Infinity
    for (const sc of squareCenters) {
      const d = (sc.cx - px) ** 2 + (sc.cy - py) ** 2
      if (d < bestD) {
        bestD = d
        best = sc
      }
    }
    if (!best) continue
    let rank
    let file
    if (userSide === "b") {
      rank = best.pageR - 1
      file = 8 - best.pageColIdx
    } else {
      rank = 10 - best.pageR
      file = best.pageColIdx
    }
    board[rank][file] = ch
  }

  const ranks = board
    .map((row) => {
      let s = ""
      let empty = 0
      for (const c of row) {
        if (c === null) {
          empty++
        } else {
          if (empty) {
            s += empty
            empty = 0
          }
          s += c
        }
      }
      if (empty) s += empty
      return s
    })
    .join("/")

  return { board, ranks, userSide }
}

function boardDiffMover(prevBoard, currBoard) {
  if (!prevBoard) return { mover: null, changed: 0 }
  let changed = 0
  let mover = null
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 9; c++) {
      const a = prevBoard[r][c]
      const b = currBoard[r][c]
      if (a !== b) {
        changed++
        if (b && b !== a) mover = b === b.toUpperCase() ? "w" : "b"
      }
    }
  }
  return { mover, changed }
}

function inferSideToMove(ranks, prevRanks, prevBoard, board, userSide) {
  if (ranks === STARTPOS_RANKS) return "w"
  const diff = boardDiffMover(prevBoard, board)
  if (diff.mover) return diff.mover === "w" ? "b" : "w"
  return userSide === "w" ? "w" : "b"
}

async function readWebSnapshot(driver, prev) {
  const raw = await driver.executeScript(readBoardInBrowser)
  if (!raw || !raw.ranks) return null

  const sideToMove = inferSideToMove(
    raw.ranks,
    prev && prev.ranks,
    prev && prev.board,
    raw.board,
    raw.userSide
  )
  const fen = `${raw.ranks} ${sideToMove} - - 0 1`
  return {
    fen,
    ranks: raw.ranks,
    board: raw.board,
    userSide: raw.userSide,
    sideToMove,
  }
}

/** ICCS x,y → 相弈 .square 选择器（userSide=w 即红在下方） */
function iccsToSquareClass(x, y, userSide = "w") {
  let pageR
  let pageCol
  if (userSide === "b") {
    pageR = 10 - y
    pageCol = String.fromCharCode("a".charCodeAt(0) + (8 - x))
  } else {
    pageR = y + 1
    pageCol = String.fromCharCode("a".charCodeAt(0) + x)
  }
  return `${pageR}-${pageCol}`
}

module.exports = {
  STARTPOS_RANKS,
  readWebSnapshot,
  iccsToSquareClass,
  parseBridge4: (token) => {
    const d = String(token).replace(/\D/g, "").slice(-4)
    return {
      y1: Number(d.charAt(0)),
      x1: Number(d.charAt(1)),
      y2: Number(d.charAt(2)),
      x2: Number(d.charAt(3)),
    }
  },
}
