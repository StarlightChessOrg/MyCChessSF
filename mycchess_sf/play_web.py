"""Web play (Sanic): rules and AI from ``xqwlight_core`` (XQWL06)."""
from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

_cwd = str(Path.cwd().resolve())
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

import numpy as np

from mycchess_sf.chess.session import GamePlay
from mycchess_sf.iccs_util import parse_move_squares
from mycchess_sf.xqwl_state import REP_RULE_VALUE_DRAWISH_ABS

STRATEGY_HUMAN = "人类"
STRATEGY_XQWL = "象棋小巫师"
STRATEGIES = (STRATEGY_HUMAN, STRATEGY_XQWL)

_PIECE_CHAR = {
    "R": "车",
    "N": "马",
    "B": "相",
    "A": "仕",
    "K": "帅",
    "C": "炮",
    "P": "兵",
    "r": "车",
    "n": "马",
    "b": "象",
    "a": "士",
    "k": "将",
    "c": "炮",
    "p": "卒",
}


def _piece_side(ch: str | None) -> str | None:
    if not ch:
        return None
    return "red" if ch.isupper() else "black"


def _board_view_y_to_iccs_y(iy_view: int) -> int:
    return 9 - iy_view


def _iccs_y_to_board_view_y(iy_iccs: int) -> int:
    return 9 - iy_iccs


class XqwlWebSession:
    def __init__(self, engine, *, think_ms: int = 1000, book_available: bool = False) -> None:
        self._lock = threading.Lock()
        self._engine = engine
        self._think_ms = max(50, int(think_ms))
        self._book_available = bool(book_available)
        self._use_book = bool(book_available)
        self.game = GamePlay()
        self.sel_from: tuple[int, int] | None = None
        self.last_move: tuple[int, int, int, int] | None = None
        self.strategy_red = STRATEGY_HUMAN
        self.strategy_black = STRATEGY_XQWL
        self._toasts: list[dict[str, str]] = []
        self._ai_busy = False
        self._ai_log: list[str] = []
        self._ai_thinking: str = ""

    def _raw_board(self) -> np.ndarray:
        return self.game.board_view()

    def _legal_strings(self) -> set[str]:
        return set(self.game.legal_moves_iccs_str())

    def _apply_move(self, mv: str) -> None:
        if mv not in self._legal_strings():
            return
        self.game.make_move_iccs(mv)
        x1, y1, x2, y2 = parse_move_squares(mv)
        self.last_move = (x1, y1, x2, y2)
        self._check_terminal()

    def _check_terminal(self) -> None:
        t, r = self.game.terminal()
        if not t:
            return
        if r == "checkmate":
            stm = "红方" if self.game.red_to_move else "黑方"
            self._toasts.append({"kind": "info", "title": "终局", "body": f"{stm} 被将死。"})
        elif r == "repetition_rule":
            v = int(self.game.rep_value_if_any())
            if abs(v) <= REP_RULE_VALUE_DRAWISH_ABS:
                self._toasts.append({"kind": "info", "title": "终局", "body": "重复局面（和棋）。"})
            elif v > 0:
                w = "红方" if self.game.red_to_move else "黑方"
                self._toasts.append(
                    {"kind": "info", "title": "终局", "body": f"重复判例：{w} 胜（对方犯规判负）。"}
                )
            else:
                w = "黑方" if self.game.red_to_move else "红方"
                self._toasts.append(
                    {"kind": "info", "title": "终局", "body": f"重复判例：{w} 胜（对方犯规判负）。"}
                )
        else:
            self._toasts.append({"kind": "info", "title": "终局", "body": r})

    def snapshot(self) -> dict:
        with self._lock:
            arr = self._raw_board()
            rows: list[list[dict[str, str | None]]] = []
            for iy in range(10):
                row: list[dict[str, str | None]] = []
                for ix in range(9):
                    ch = arr[iy, ix]
                    if not ch:
                        row.append({"ch": None, "side": None, "label": ""})
                    else:
                        s = str(ch)
                        row.append({"ch": s, "side": _piece_side(s), "label": _PIECE_CHAR.get(s, "?")})
                rows.append(row)
            side = self.game.get_side()
            lm_view: list[int] | None = None
            if self.last_move:
                x1, y1, x2, y2 = self.last_move
                lm_view = [x1, _iccs_y_to_board_view_y(y1), x2, _iccs_y_to_board_view_y(y2)]
            return {
                "board": rows,
                "visual_sig": "|".join(str(arr[iy, ix] or ".") for iy in range(10) for ix in range(9)),
                "side_to_move": side,
                "sel_from": list(self.sel_from) if self.sel_from else None,
                "last_move": lm_view,
                "strategy_red": self.strategy_red,
                "strategy_black": self.strategy_black,
                "strategies": list(STRATEGIES),
                "book_available": self._book_available,
                "use_book": self._use_book,
                "book_size": int(self._engine.book_size()) if self._book_available else 0,
                "status_text": f"轮到 {'红方' if side == 'red' else '黑方'} 走棋",
                "ai_busy": self._ai_busy,
                "current_strategy": self.strategy_red if side == "red" else self.strategy_black,
                "ai_thinking": self._ai_thinking,
                "ai_log": list(self._ai_log[-48:]),
            }

    def pop_client_messages(self) -> dict:
        with self._lock:
            t, self._toasts = self._toasts, []
            return {"toasts": t, "ai_errors": []}

    def set_strategies(self, red: str, black: str) -> dict | None:
        if red not in STRATEGIES or black not in STRATEGIES:
            return {"error": "非法策略"}
        with self._lock:
            self.strategy_red = red
            self.strategy_black = black
        self.maybe_ai()
        return None

    def set_use_book(self, enabled: bool) -> dict | None:
        if not self._book_available:
            if enabled:
                return {"error": "未加载开局库 BOOK.DAT"}
            with self._lock:
                self._use_book = False
            return None
        with self._lock:
            self._use_book = bool(enabled)
        return None

    def new_game(self) -> dict | None:
        with self._lock:
            self.game.reset()
            self.sel_from = None
            self.last_move = None
            self._ai_thinking = ""
            self._ai_log.append("—— 新局 ——")
            if len(self._ai_log) > 200:
                self._ai_log[:] = self._ai_log[-120:]
        self.maybe_ai()
        return None

    def click_cell(self, ix: int, iy: int) -> dict | None:
        with self._lock:
            if self._ai_busy:
                return {"error": "小巫师思考中"}
            side = self.game.get_side()
            strat = self.strategy_red if side == "red" else self.strategy_black
            if strat != STRATEGY_HUMAN:
                return {"error": "当前非人类行棋"}
            if not (0 <= ix <= 8 and 0 <= iy <= 9):
                return {"error": "坐标越界"}
            arr = self._raw_board()
            ch = arr[iy, ix]
            if self.sel_from is None:
                if _piece_side(str(ch) if ch else None) == side:
                    self.sel_from = (ix, iy)
                return None
            fx, fy = self.sel_from
            y1e, y2e = _board_view_y_to_iccs_y(fy), _board_view_y_to_iccs_y(iy)
            mv = f"{fx}{y1e}-{ix}{y2e}"
            if mv not in self._legal_strings():
                if _piece_side(str(ch) if ch else None) == side:
                    self.sel_from = (ix, iy)
                else:
                    self.sel_from = None
                return None
            self._apply_move(mv)
            self.sel_from = None
        self.maybe_ai()
        return None

    def maybe_ai(self) -> None:
        with self._lock:
            if self._ai_busy:
                return
            side = self.game.get_side()
            strat = self.strategy_red if side == "red" else self.strategy_black
            if strat != STRATEGY_XQWL:
                return
            if self.game.terminal()[0]:
                return
            if not self.game.legal_moves_iccs_str():
                return
            self._ai_busy = True
            g_copy = self.game.copy()
            think_ms = self._think_ms
            engine = self._engine
            use_book = self._use_book and self._book_available

        def worker() -> None:
            log_line = ""
            try:
                with self._lock:
                    mode = "开局库+搜索" if use_book else "纯搜索"
                    self._ai_thinking = f"象棋小巫师思考中（{mode}，约 {think_ms} ms）…"
                t0 = time.perf_counter()
                pos = g_copy.raw_position()
                mv = engine.search_best_iccs(pos, think_ms, use_book)
                elapsed = (time.perf_counter() - t0) * 1000.0
                log_line = f"象棋小巫师 | {'库' if use_book else '搜'} | 着 {mv} | {elapsed:.0f} ms"
            except Exception as e:
                with self._lock:
                    self._ai_busy = False
                    self._ai_thinking = ""
                    self._toasts.append({"kind": "info", "title": "小巫师错误", "body": str(e)})
                return
            with self._lock:
                self._ai_busy = False
                self._ai_thinking = ""
                if log_line:
                    self._ai_log.append(log_line)
                    if len(self._ai_log) > 200:
                        self._ai_log[:] = self._ai_log[-120:]
                if mv not in self._legal_strings():
                    self._toasts.append({"kind": "info", "title": "小巫师", "body": f"非法着法 {mv}"})
                    return
                self._apply_move(mv)
            self.maybe_ai()

        threading.Thread(target=worker, daemon=True).start()


def _html_page() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>MyCChessSF · 象棋小巫师</title>
  <style>
    :root { --bg0:#1a1510; --bg1:#2d2419; --panel:#352a22; --panel2:#2a2218; --line:rgba(74,50,37,.45);
      --board-bg0:#f2e8d4; --board-bg1:#e5d3b6; --red:#c62828; --black:#1565c0; --text:#f2ebe3;
      --muted:rgba(242,235,227,.72); --accent:#ff9800; --sel:#ffeb3b; --radius:14px; }
    *{box-sizing:border-box} body{margin:0;font-family:"Microsoft YaHei","PingFang SC",sans-serif;color:var(--text);
      min-height:100vh;background:radial-gradient(120% 80% at 50% 0%,var(--bg1) 0%,var(--bg0) 55%,#120e0a 100%)}
    .shell{max-width:1480px;margin:0 auto;min-height:100vh;padding:clamp(14px,2.2vw,28px);
      display:grid;grid-template-columns:minmax(200px,280px) minmax(0,1fr) minmax(240px,300px);
      gap:clamp(14px,2.2vw,24px);align-items:start}
    @media(max-width:960px){.shell{grid-template-columns:1fr;}.ai-log-panel{order:3;max-height:220px;}.board-wrap{order:1;}.sidepanel{order:2;}}
    .board-wrap{display:flex;justify-content:center;align-items:center}
    .board-card{background:linear-gradient(145deg,#faf3e6 0%,var(--board-bg1) 100%);border-radius:var(--radius);
      padding:clamp(10px,1.4vw,16px);box-shadow:0 4px 0 rgba(62,39,35,.35),0 18px 48px rgba(0,0,0,.45);
      border:1px solid rgba(62,39,35,.25)}
    .board{--cs:clamp(42px,min((100vw - 48px)/9.6,(100vh - 120px)/10.2),76px);display:grid;
      grid-template-columns:repeat(9,var(--cs));grid-template-rows:repeat(10,var(--cs));
      width:calc(9 * var(--cs));height:calc(10 * var(--cs));
      background:linear-gradient(180deg,var(--board-bg0) 0%,var(--board-bg1) 100%);border-radius:10px;overflow:hidden}
    .cell{border:1px solid var(--line);cursor:pointer;display:flex;align-items:center;justify-content:center;user-select:none}
    .cell:hover{background:rgba(255,255,255,.14)}
    .piece-red,.piece-black{color:#fff;border-radius:50%;width:calc(var(--cs)*.78);height:calc(var(--cs)*.78);
      min-width:32px;min-height:32px;display:flex;align-items:center;justify-content:center;border:2px solid #3e2723;
      font-size:clamp(15px,calc(var(--cs)*.38),30px);font-weight:700;box-shadow:0 2px 6px rgba(0,0,0,.22)}
    .piece-red{background:linear-gradient(165deg,#e53935 0%,var(--red) 55%,#8b0000 100%)}
    .piece-black{background:linear-gradient(165deg,#42a5f5 0%,var(--black) 55%,#0d47a1 100%)}
    .sel{outline:3px solid var(--sel);outline-offset:-3px;border-radius:4px}
    .last-from,.last-to{box-shadow:inset 0 0 0 3px var(--accent)}
    .sidepanel{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);padding:clamp(16px,2vw,22px);
      border-radius:var(--radius);border:1px solid rgba(255,255,255,.06);box-shadow:0 12px 40px rgba(0,0,0,.35)}
    h1{font-size:clamp(1.05rem,2.2vw,1.25rem);margin:0 0 6px;font-weight:600}
    .subtitle{font-size:12px;color:var(--muted);margin-bottom:14px}
    label{display:block;margin-top:10px;font-size:13px;color:var(--muted)}
    .check-row{display:flex;align-items:center;gap:10px;margin-top:14px;font-size:14px;color:var(--text)}
    .check-row input{width:18px;height:18px;cursor:pointer}
    .check-row.disabled{opacity:.45;pointer-events:none}
    select{width:100%;padding:10px 12px;margin-top:6px;border-radius:10px;border:1px solid rgba(93,78,58,.6);
      background:rgba(0,0,0,.25);color:var(--text);font-size:14px}
    button{margin-top:16px;padding:12px 16px;border:none;border-radius:10px;
      background:linear-gradient(180deg,#8d6e63 0%,#6d4c41 100%);color:#fff;font-size:15px;font-weight:600;cursor:pointer;width:100%}
    button.btn-secondary{margin-top:10px;background:linear-gradient(180deg,#5d6b7a 0%,#455a64 100%)}
    #status{margin-top:16px;white-space:pre-wrap;font-size:13px;padding:12px 14px;background:rgba(0,0,0,.22);border-radius:10px;min-height:4.5em}
    .ai-busy .board{opacity:.92;pointer-events:none}
    .ai-log-panel{align-self:start}
    .ai-thinking{min-height:2.1em;font-size:12px;color:#ffcc80;margin-bottom:8px;white-space:pre-wrap;word-break:break-word}
    #ai-log-body{margin:0;font-family:ui-monospace,Consolas,"Courier New",monospace;font-size:11px;line-height:1.45;
      max-height:min(560px,calc(100vh - 200px));overflow:auto;padding:10px 12px;background:rgba(0,0,0,.22);
      border-radius:10px;color:rgba(242,235,227,.92);border:1px solid rgba(255,255,255,.06)}
  </style>
</head>
<body>
  <div class="shell" id="shell">
    <div class="sidepanel ai-log-panel">
      <h1 style="font-size:clamp(0.95rem,1.8vw,1.1rem)">小巫师日志</h1>
      <div class="subtitle">每步搜索耗时与着法 ICCS</div>
      <div id="ai-thinking" class="ai-thinking"></div>
      <pre id="ai-log-body"></pre>
    </div>
    <div class="board-wrap"><div class="board-card"><div class="board" id="board"></div></div></div>
    <div class="sidepanel">
      <h1>MyCChessSF · 象棋小巫师</h1>
      <div class="subtitle">XQWL06 规则与搜索 · 默认红方人类、黑方小巫师</div>
      <label>红方策略</label><select id="sel-red"></select>
      <label>黑方策略</label><select id="sel-black"></select>
      <div id="book-row" class="check-row disabled"><label for="chk-book"><input type="checkbox" id="chk-book" disabled/> 小巫师使用开局库（BOOK.DAT）</label></div>
      <button type="button" id="btn-new">新局</button>
      <button type="button" class="btn-secondary" id="btn-flip" title="默认红方在下面">翻转棋盘（黑方视角）</button>
      <div id="status"></div>
    </div>
  </div>
<script>
(function(){
  const shell=document.getElementById("shell"),boardEl=document.getElementById("board"),statusEl=document.getElementById("status");
  const selRed=document.getElementById("sel-red"),selBlack=document.getElementById("sel-black"),btnNew=document.getElementById("btn-new");
  const btnFlip=document.getElementById("btn-flip");
  const chkBook=document.getElementById("chk-book"),bookRow=document.getElementById("book-row");
  const aiThinkingEl=document.getElementById("ai-thinking"),aiLogBody=document.getElementById("ai-log-body");
  let viewFlipY=true,pollTimer=null,lastSnap=null;
  function showAlert(t,b){alert(t+"\\n\\n"+b);}
  function fillStrategiesOnce(strategies){
    if(selRed.options.length>0)return;
    strategies.forEach(function(t){var o=document.createElement("option");o.value=o.textContent=t;selRed.appendChild(o);});
    strategies.forEach(function(t){var o=document.createElement("option");o.value=o.textContent=t;selBlack.appendChild(o);});
  }
  function srvY(iyVis){return viewFlipY?(9-iyVis):iyVis;}
  function renderCells(snap){
    var lm=snap.last_move,sf=snap.sel_from,frag=document.createDocumentFragment();
    for(var iyVis=0;iyVis<10;iyVis++)for(var ix=0;ix<9;ix++){
      var iy=srvY(iyVis);
      var cell=document.createElement("div");cell.className="cell";cell.dataset.ix=String(ix);cell.dataset.iy=String(iy);
      if(lm&&ix===lm[0]&&iy===lm[1])cell.classList.add("last-from");
      if(lm&&ix===lm[2]&&iy===lm[3])cell.classList.add("last-to");
      if(sf&&ix===sf[0]&&iy===sf[1])cell.classList.add("sel");
      var sq=snap.board[iy][ix];
      if(sq.ch){var span=document.createElement("span");span.textContent=sq.label||"?";
        span.className=sq.side==="red"?"piece-red":"piece-black";cell.appendChild(span);}
      frag.appendChild(cell);
    }
    boardEl.replaceChildren(frag);
  }
  function applySnap(snap){
    fillStrategiesOnce(snap.strategies||[]);selRed.value=snap.strategy_red;selBlack.value=snap.strategy_black;
    if(chkBook&&bookRow){
      var avail=!!snap.book_available;
      bookRow.classList.toggle("disabled",!avail);
      chkBook.disabled=!avail;
      if(avail){ chkBook.checked=!!snap.use_book; }
    }
    statusEl.textContent=snap.status_text||"";
    if(aiThinkingEl) aiThinkingEl.textContent=snap.ai_thinking||"";
    if(aiLogBody){ aiLogBody.textContent=(snap.ai_log||[]).join("\\n"); aiLogBody.scrollTop=aiLogBody.scrollHeight; }
    renderCells(snap);
    shell.classList.toggle("ai-busy",!!snap.ai_busy);
  }
  function handleMessages(msg){
    (msg.toasts||[]).forEach(function(t){if(t.kind==="info")showAlert(t.title||"提示",t.body||"");});
  }
  function armPoll(ms){if(pollTimer)clearTimeout(pollTimer);pollTimer=setTimeout(onePoll,ms);}
  function onePoll(){
    pollTimer=null;
    Promise.all([fetch("/api/state",{cache:"no-store"}).then(function(r){return r.json();}),
      fetch("/api/messages",{cache:"no-store"}).then(function(r){return r.json();})])
      .then(function(pair){lastSnap=pair[0];applySnap(pair[0]);handleMessages(pair[1]);armPoll(pair[0].ai_busy?160:380);})
      .catch(function(){armPoll(700);});
  }
  boardEl.addEventListener("click",function(ev){
    var cell=(ev.target.closest&&ev.target.closest(".cell"))||null;
    if(!cell||shell.classList.contains("ai-busy"))return;
    var ix=parseInt(cell.dataset.ix,10),iy=parseInt(cell.dataset.iy,10);
    fetch("/api/click",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ix:ix,iy:iy})})
      .then(function(r){return r.json();}).then(function(j){if(j.error)showAlert("走子",j.error);else armPoll(25);});
  });
  selRed.addEventListener("change",function(){
    fetch("/api/strategies",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({red:selRed.value,black:selBlack.value})}).then(function(r){return r.json();})
      .then(function(j){if(j.error)showAlert("策略",j.error);else armPoll(25);});
  });
  selBlack.addEventListener("change",function(){
    fetch("/api/strategies",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({red:selRed.value,black:selBlack.value})}).then(function(r){return r.json();})
      .then(function(j){if(j.error)showAlert("策略",j.error);else armPoll(25);});
  });
  btnNew.addEventListener("click",function(){
    fetch("/api/new_game",{method:"POST"}).then(function(r){return r.json();}).then(function(){armPoll(25);});
  });
  btnFlip.addEventListener("click",function(){
    viewFlipY=!viewFlipY;
    btnFlip.textContent=viewFlipY?"翻转棋盘（黑方视角）":"还原红方在下面";
    if(lastSnap)renderCells(lastSnap);
  });
  if(chkBook){
    chkBook.addEventListener("change",function(){
      fetch("/api/book",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({enabled:chkBook.checked})}).then(function(r){return r.json();})
        .then(function(j){if(j.error)showAlert("开局库",j.error);else armPoll(25);});
    });
  }
  onePoll();
})();
</script>
</body>
</html>"""


def _default_book_path() -> Path | None:
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "BOOK.DAT",
        Path.cwd() / "data" / "BOOK.DAT",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def main() -> None:
    try:
        from sanic import Sanic
        from sanic.response import html, json
    except ImportError as e:
        raise SystemExit("请安装: pip install 'sanic>=23.12'") from e

    from xqwlight_core import Engine

    p = argparse.ArgumentParser(description="MyCChessSF 网页对弈（象棋小巫师 XQWL06）")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--think-ms", type=int, default=1000, help="小巫师每步思考时间（毫秒）")
    p.add_argument(
        "--book",
        type=Path,
        default=None,
        help="开局库 BOOK.DAT（默认 data/BOOK.DAT）",
    )
    p.add_argument(
        "--no-book-default",
        action="store_true",
        help="启动时默认关闭开局库（网页仍可勾选开启）",
    )
    args = p.parse_args()

    engine = Engine()
    book = args.book
    if book is None:
        book = _default_book_path()
    book_available = False
    if book is not None and book.is_file():
        engine.load_book(str(book))
        book_available = engine.book_size() > 0
        print(f"[play] 已加载开局库: {book} ({engine.book_size()} 项)", flush=True)
    else:
        print("[play] 未找到 BOOK.DAT", flush=True)

    use_book_default = book_available and not args.no_book_default
    session = XqwlWebSession(engine, think_ms=int(args.think_ms), book_available=book_available)
    session.set_use_book(use_book_default)

    app = Sanic("mycchess_sf_play_web")
    app.config.RESPONSE_TIMEOUT = max(60, int(args.think_ms) // 500 + 30)

    @app.get("/")
    async def _index(_request):
        return html(_html_page())

    @app.get("/api/state")
    async def _api_state(_request):
        return json(session.snapshot())

    @app.get("/api/messages")
    async def _api_messages(_request):
        return json(session.pop_client_messages())

    @app.post("/api/click")
    async def _api_click(request):
        data = request.json
        if not isinstance(data, dict):
            return json({"error": "无效 JSON"}, status=400)
        try:
            ix = int(data.get("ix", -1))
            iy = int(data.get("iy", -1))
        except (TypeError, ValueError):
            return json({"error": "坐标无效"}, status=400)
        err = session.click_cell(ix, iy)
        return json(err or {})

    @app.post("/api/strategies")
    async def _api_strategies(request):
        data = request.json
        if not isinstance(data, dict):
            return json({"error": "参数无效"}, status=400)
        red, black = data.get("red"), data.get("black")
        if not isinstance(red, str) or not isinstance(black, str):
            return json({"error": "参数无效"}, status=400)
        err = session.set_strategies(red, black)
        return json(err or {})

    @app.post("/api/new_game")
    async def _api_new_game(_request):
        err = session.new_game()
        return json(err or {})

    @app.post("/api/book")
    async def _api_book(request):
        data = request.json
        if not isinstance(data, dict):
            return json({"error": "参数无效"}, status=400)
        enabled = data.get("enabled")
        if not isinstance(enabled, bool):
            return json({"error": "enabled 须为布尔值"}, status=400)
        err = session.set_use_book(enabled)
        return json(err or {})

    @app.after_server_start
    async def _kick_ai(_app, _loop):
        await asyncio.sleep(0.2)
        session.maybe_ai()

    print(f"[play] http://{args.host}:{args.port}/  (象棋小巫师 XQWL06)", flush=True)
    app.run(host=str(args.host), port=int(args.port), single_process=True, access_log=False, motd=False)


if __name__ == "__main__":
    main()
