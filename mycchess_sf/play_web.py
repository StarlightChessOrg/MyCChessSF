"""Web play (Sanic): XQWL06 rules/search with Win32-style board UI."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
from pathlib import Path

_cwd = str(Path.cwd().resolve())
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)

from mycchess_sf.chess.session import GamePlay
from mycchess_sf.iccs_util import parse_move_squares
from mycchess_sf.move_desc import think_log_entry
from mycchess_sf.xqwl_assets import (
    BOARD_EDGE,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    PIECE_SPRITE,
    SOUND_NAMES,
    SQUARE_SIZE,
    STATIC_DIR,
    assets_available,
)
from mycchess_sf.xqwl_state import REP_RULE_VALUE_DRAWISH_ABS

STRATEGY_HUMAN = "人类"
STRATEGY_XQWL = "象棋小巫师"
STRATEGIES = (STRATEGY_HUMAN, STRATEGY_XQWL)
AI_THINK_DELAY_SEC = 0.45
AI_VS_AI_MIN_STEP_SEC = 0.5


def _piece_side(ch: str | None) -> str | None:
    if not ch:
        return None
    return "red" if ch.isupper() else "black"


def _board_view_y_to_iccs_y(iy_view: int) -> int:
    return iy_view


def _iccs_y_to_board_view_y(iy_iccs: int) -> int:
    return iy_iccs


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
        self._sounds: list[str] = []
        self._ai_busy = False
        self._ai_pending = False
        self._think_log: list[dict[str, object]] = []
        self._ai_thinking: str = ""
        self._ai_timer: threading.Timer | None = None

    def _cancel_ai_timer(self) -> None:
        if self._ai_timer is not None:
            self._ai_timer.cancel()
            self._ai_timer = None
        self._ai_pending = False

    def _schedule_ai_after_human_move(self) -> None:
        self._cancel_ai_timer()
        with self._lock:
            side = self.game.get_side()
            strat = self.strategy_red if side == "red" else self.strategy_black
            if strat == STRATEGY_XQWL:
                self._ai_pending = True
        self._ai_timer = threading.Timer(AI_THINK_DELAY_SEC, self._ai_timer_fire)
        self._ai_timer.daemon = True
        self._ai_timer.start()

    def _ai_timer_fire(self) -> None:
        with self._lock:
            self._ai_timer = None
        self.maybe_ai()

    def _raw_board(self):
        return self.game.board_view()

    def _legal_strings(self) -> set[str]:
        return set(self.game.legal_moves_iccs_str())

    def _queue_sound(self, name: str) -> None:
        if name in SOUND_NAMES:
            self._sounds.append(name)

    def _terminal_sound(self, *, for_ai: bool) -> None:
        t, r = self.game.terminal()
        if not t:
            return
        if r == "checkmate":
            self._queue_sound("loss" if for_ai else "win")
            return
        if r == "repetition_rule":
            try:
                from xqwlight_core import WIN_VALUE
            except ImportError:
                WIN_VALUE = 9800
            vl = int(self.game.rep_value_if_any())
            if abs(vl) <= REP_RULE_VALUE_DRAWISH_ABS:
                self._queue_sound("draw")
            elif for_ai:
                self._queue_sound("loss" if vl < -WIN_VALUE else "win" if vl > WIN_VALUE else "draw")
            else:
                self._queue_sound("win" if vl > WIN_VALUE else "loss" if vl < -WIN_VALUE else "draw")
            return
        if r == "move_limit_draw":
            self._queue_sound("draw")

    def _ply_sound(self, *, for_ai: bool) -> None:
        self._terminal_sound(for_ai=for_ai)
        if self.game.terminal()[0]:
            return
        pos = self.game.pos
        if bool(pos.in_check()):
            self._queue_sound("check2" if for_ai else "check")
        elif bool(pos.captured_last()):
            self._queue_sound("capture2" if for_ai else "capture")
        else:
            self._queue_sound("move2" if for_ai else "move")

    def _apply_move(self, mv: str, *, for_ai: bool = False) -> None:
        if mv not in self._legal_strings():
            return
        self.game.make_move_iccs(mv)
        x1, y1, x2, y2 = parse_move_squares(mv)
        self.last_move = (x1, y1, x2, y2)
        self._ply_sound(for_ai=for_ai)
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

    def _view_flip_y(self) -> bool:
        """Flip board so the human-controlled side sits at the bottom (like XQWL bFlipped)."""
        human_red = self.strategy_red == STRATEGY_HUMAN
        human_black = self.strategy_black == STRATEGY_HUMAN
        if human_black and not human_red:
            return True
        return False

    def _both_ai(self) -> bool:
        return self.strategy_red == STRATEGY_XQWL and self.strategy_black == STRATEGY_XQWL

    def snapshot(self) -> dict:
        with self._lock:
            arr = self._raw_board()
            rows: list[list[dict[str, str | None]]] = []
            for iy in range(10):
                row: list[dict[str, str | None]] = []
                for ix in range(9):
                    ch = arr[iy, ix]
                    if not ch:
                        row.append({"ch": None, "side": None, "sprite": None})
                    else:
                        s = str(ch)
                        row.append(
                            {
                                "ch": s,
                                "side": _piece_side(s),
                                "sprite": PIECE_SPRITE.get(s),
                            }
                        )
                rows.append(row)
            side = self.game.get_side()
            cur_strat = self.strategy_red if side == "red" else self.strategy_black
            human_turn = cur_strat == STRATEGY_HUMAN
            input_locked = self._ai_busy or self._ai_pending or not human_turn
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
                "ai_pending": self._ai_pending,
                "input_locked": input_locked,
                "current_strategy": cur_strat,
                "ai_thinking": self._ai_thinking,
                "think_log": list(self._think_log[-60:]),
                "board_w": BOARD_WIDTH,
                "board_h": BOARD_HEIGHT,
                "square": SQUARE_SIZE,
                "edge": BOARD_EDGE,
                "view_flip_y": self._view_flip_y(),
            }

    def pop_client_messages(self) -> dict:
        with self._lock:
            t, self._toasts = self._toasts, []
            s, self._sounds = self._sounds, []
            return {"toasts": t, "sounds": s, "ai_errors": []}

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
            self._cancel_ai_timer()
            self.game.reset()
            self.sel_from = None
            self.last_move = None
            self._ai_thinking = ""
            self._think_log.clear()
        self.maybe_ai()
        return None

    def click_cell(self, ix: int, iy: int) -> dict | None:
        """Handle a click; ``iy`` is board_view row (0=black top, 9=red bottom)."""
        with self._lock:
            if self._ai_busy or self._ai_pending:
                return {"error": "小巫师思考中"}
            side = self.game.get_side()
            strat = self.strategy_red if side == "red" else self.strategy_black
            if strat != STRATEGY_HUMAN:
                return {"error": "当前非人类行棋"}
            if not (0 <= ix <= 8 and 0 <= iy <= 9):
                return {"error": "坐标越界"}
            if self.game.terminal()[0]:
                return {"error": "对局已结束"}
            arr = self._raw_board()
            vy = iy
            ch = arr[vy, ix]
            pc_side = _piece_side(str(ch) if ch else None)

            if pc_side == side:
                self.sel_from = (ix, vy)
                self._queue_sound("click")
                return None

            if self.sel_from is None:
                return None

            fx, fy = self.sel_from
            y1e = _board_view_y_to_iccs_y(fy)
            y2e = _board_view_y_to_iccs_y(iy)
            mv = f"{fx}{y1e}-{ix}{y2e}"
            pos = self.game.pos
            game_over = False

            if mv in self._legal_strings():
                self._apply_move(mv, for_ai=False)
                self.sel_from = None
                game_over = self.game.terminal()[0]
            elif bool(pos.pseudo_legal_iccs(mv)) and not bool(pos.try_make_move_iccs(mv)):
                self._queue_sound("illegal")
                return None
            else:
                if pc_side == side:
                    self.sel_from = (ix, vy)
                    self._queue_sound("click")
                return None

        if not game_over:
            self._schedule_ai_after_human_move()
        return {}

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
            self._ai_pending = False
            g_copy = self.game.copy()
            think_ms = self._think_ms
            engine = self._engine
            use_book = self._use_book and self._book_available
            both_ai = self._both_ai()

        def worker() -> None:
            mv = ""
            detail: dict = {}
            try:
                with self._lock:
                    mode = "开局库+搜索" if use_book else "纯搜索"
                    self._ai_thinking = f"象棋小巫师思考中（{mode}，约 {think_ms} ms）…"
                t0 = time.perf_counter()
                pos = g_copy.raw_position()
                detail = dict(engine.search_best_detail(pos, think_ms, use_book))
                mv = str(detail.get("iccs", "") or "")
                elapsed = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                with self._lock:
                    self._ai_busy = False
                    self._ai_pending = False
                    self._ai_thinking = ""
                    self._toasts.append({"kind": "info", "title": "小巫师错误", "body": str(e)})
                return
            if both_ai:
                pad = AI_VS_AI_MIN_STEP_SEC - elapsed / 1000.0
                if pad > 0:
                    time.sleep(pad)
            with self._lock:
                self._ai_busy = False
                self._ai_thinking = ""
                if mv:
                    arr = g_copy.board_view()
                    x1, y1, _, _ = parse_move_squares(mv)
                    piece_ch = str(arr[y1, x1])
                    entry = think_log_entry(
                        depth=int(detail.get("depth", 0)),
                        elapsed_ms=elapsed,
                        piece_ch=piece_ch,
                        iccs=mv,
                        from_book=bool(detail.get("from_book", False)),
                    )
                    self._think_log.append(entry)
                    if len(self._think_log) > 200:
                        self._think_log[:] = self._think_log[-120:]
                if mv not in self._legal_strings():
                    self._toasts.append({"kind": "info", "title": "小巫师", "body": f"非法着法 {mv}"})
                    return
                self._apply_move(mv, for_ai=True)
            self.maybe_ai()

        threading.Thread(target=worker, daemon=True).start()


def _html_page() -> str:
    bw, bh, sq, edge = BOARD_WIDTH, BOARD_HEIGHT, SQUARE_SIZE, BOARD_EDGE
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>象棋小巫师</title>
  <style>
    :root {{
      --bw:{bw}px; --bh:{bh}px; --sq:{sq}px; --edge:{edge}px;
      --bg:#d4c4a8; --panel:#ece3d2; --line:#2b2118; --text:#2b2118;
    }}
    *{{box-sizing:border-box}}
    body{{margin:0;font-family:"Microsoft YaHei","SimSun",serif;color:var(--text);
      background:var(--bg);min-height:100vh}}
    .win{{max-width:1280px;margin:0 auto;padding:12px 16px 20px}}
    .titlebar{{font-size:15px;font-weight:600;margin-bottom:10px}}
    .layout{{display:grid;grid-template-columns:minmax(300px,340px) minmax(0,1fr) 260px;gap:16px;align-items:start}}
    @media(max-width:1180px){{.layout{{grid-template-columns:1fr}} .think-panel{{order:3}}}}
    .board-wrap{{display:flex;justify-content:center}}
    .board-shell{{background:#8b7355;padding:6px;border:1px solid #5c4a32;box-shadow:0 2px 8px rgba(0,0,0,.25)}}
    #board{{position:relative;width:min(var(--bw),calc(100vw - 36px));aspect-ratio:{bw} / {bh};
      cursor:pointer;touch-action:manipulation;user-select:none}}
    #board .board-bg{{position:absolute;inset:0;width:100%;height:100%;display:block;pointer-events:none}}
    #layer{{position:absolute;inset:0;pointer-events:none}}
    .spr{{position:absolute;width:calc(var(--sq) / var(--bw) * 100%);height:calc(var(--sq) / var(--bh) * 100%);
      background-size:contain;background-repeat:no-repeat;background-position:center}}
    .sidepanel{{background:var(--panel);border:1px solid #b9a88d;padding:14px 16px}}
    h1{{font-size:16px;margin:0 0 8px}}
    .subtitle{{font-size:12px;color:#5c4a32;margin-bottom:12px}}
    label{{display:block;margin-top:10px;font-size:13px}}
    select,button{{width:100%;margin-top:6px;padding:8px 10px;font-size:14px}}
    button{{margin-top:14px;cursor:pointer}}
    .check-row{{display:flex;align-items:center;gap:8px;margin-top:12px;font-size:13px}}
    .check-row input{{width:16px;height:16px}}
    .think-panel{{background:var(--panel);border:1px solid #b9a88d;padding:14px 16px;align-self:stretch}}
    .think-panel h2{{font-size:15px;margin:0 0 8px}}
    .think-panel .hint{{font-size:12px;color:#5c4a32;margin-bottom:10px}}
    #think-log-body{{font-family:Consolas,"Microsoft YaHei",monospace;font-size:11px;line-height:1.5;
      max-height:min(560px,calc(100vh - 120px));overflow:auto;overflow-x:auto;background:rgba(255,255,255,.45);
      padding:8px;border:1px solid #c9b89a;min-width:0}}
    .think-line{{margin:0 0 6px;white-space:nowrap}}
    #status{{margin-top:12px;font-size:13px;min-height:3.5em;white-space:pre-wrap}}
    #ai-thinking{{font-size:12px;color:#7a4b00;min-height:1.6em;margin-top:8px}}
    .input-locked #board{{opacity:.94;pointer-events:none}}
    .dialog-overlay{{position:fixed;inset:0;background:rgba(20,14,8,.45);display:flex;align-items:center;
      justify-content:center;z-index:1000;padding:16px}}
    .dialog-overlay.hidden{{display:none}}
    .dialog-box{{background:var(--panel);border:1px solid #8b7355;box-shadow:0 4px 20px rgba(0,0,0,.35);
      min-width:min(360px,92vw);max-width:440px;padding:18px 20px 16px}}
    .dialog-title{{font-size:15px;font-weight:600;margin:0 0 10px}}
    .dialog-body{{font-size:13px;line-height:1.55;color:#3d2f1f;margin:0 0 16px;white-space:pre-wrap}}
    .dialog-actions{{display:flex;justify-content:flex-end;gap:8px}}
    .dialog-actions button{{width:auto;min-width:72px;margin:0;padding:7px 16px}}
    .toast-stack{{position:fixed;top:14px;right:14px;z-index:1001;display:flex;flex-direction:column;gap:8px;
      max-width:min(360px,calc(100vw - 28px));pointer-events:none}}
    .toast-item{{background:var(--panel);border:1px solid #b9a88d;box-shadow:0 2px 10px rgba(0,0,0,.2);
      padding:10px 14px;font-size:13px;line-height:1.45;opacity:1;transition:opacity .25s}}
    .toast-item.fade{{opacity:0}}
    .toast-item strong{{display:block;font-size:14px;margin-bottom:4px}}
  </style>
</head>
<body>
  <div class="win">
    <div class="titlebar">象棋小巫师</div>
    <div class="layout">
      <div class="think-panel">
        <h2>思考日志</h2>
        <div class="hint">小巫师每步决策：深度、耗时、棋子与坐标</div>
        <div id="think-log-body"></div>
      </div>
      <div class="board-wrap">
        <div class="board-shell">
          <div id="board">
            <img class="board-bg" src="/static/xqwl/board.png" alt="棋盘" width="{bw}" height="{bh}"/>
            <div id="layer"></div>
          </div>
        </div>
      </div>
      <div class="sidepanel">
        <h1>对弈设置</h1>
        <div class="subtitle">XQWL06 界面 · 默认红方人类、黑方小巫师</div>
        <label>红方策略</label><select id="sel-red"></select>
        <label>黑方策略</label><select id="sel-black"></select>
        <div id="book-row" class="check-row"><label><input type="checkbox" id="chk-book" disabled/> 使用开局库 BOOK.DAT</label></div>
        <div class="check-row"><label><input type="checkbox" id="chk-sound" checked/> 音效</label></div>
        <button type="button" id="btn-new">新局</button>
        <button type="button" id="btn-flip">翻转棋盘</button>
        <div id="status"></div>
        <div id="ai-thinking"></div>
      </div>
    </div>
  </div>
  <div id="dialog-overlay" class="dialog-overlay hidden" role="dialog" aria-modal="true">
    <div class="dialog-box">
      <div class="dialog-title" id="dialog-title"></div>
      <div class="dialog-body" id="dialog-body"></div>
      <div class="dialog-actions"><button type="button" id="dialog-ok">确定</button></div>
    </div>
  </div>
  <div id="toast-stack" class="toast-stack"></div>
<script>
(function(){{
  const BW={bw}, BH={bh}, SQ={sq}, EDGE={edge};
  const boardEl=document.getElementById("board"), layer=document.getElementById("layer");
  const statusEl=document.getElementById("status"), shell=document.querySelector(".win");
  const selRed=document.getElementById("sel-red"), selBlack=document.getElementById("sel-black");
  const btnNew=document.getElementById("btn-new"), btnFlip=document.getElementById("btn-flip");
  const chkBook=document.getElementById("chk-book"), bookRow=document.getElementById("book-row");
  const chkSound=document.getElementById("chk-sound");
  const aiThinkingEl=document.getElementById("ai-thinking"), thinkLogBody=document.getElementById("think-log-body");
  const dialogOverlay=document.getElementById("dialog-overlay"), dialogTitle=document.getElementById("dialog-title");
  const dialogBody=document.getElementById("dialog-body"), dialogOk=document.getElementById("dialog-ok");
  const toastStack=document.getElementById("toast-stack");
  let viewFlipY=false, userFlipped=false, pollTimer=null, pollInFlight=false, lastSnap=null;
  let lastBoardKey=null, lastThinkLogKey=null;
  const sounds={{}};
  ["click","illegal","move","move2","capture","capture2","check","check2","win","draw","loss"].forEach(function(n){{
    sounds[n]=new Audio("/static/xqwl/"+n+".wav");
  }});
  function playSounds(list){{
    if(!chkSound||!chkSound.checked||!list||!list.length)return;
    var i=0;
    function next(){{ if(i>=list.length)return; var a=sounds[list[i++]]; if(!a)return next();
      a.currentTime=0; a.play().catch(function(){{}}); a.onended=next; }}
    next();
  }}
  function showAlert(t,b){{
    if(!dialogOverlay)return;
    dialogTitle.textContent=t||"提示";
    dialogBody.textContent=b||"";
    dialogOverlay.classList.remove("hidden");
    dialogOk.focus();
  }}
  function hideAlert(){{if(dialogOverlay)dialogOverlay.classList.add("hidden");}}
  if(dialogOk)dialogOk.addEventListener("click",hideAlert);
  if(dialogOverlay)dialogOverlay.addEventListener("click",function(ev){{
    if(ev.target===dialogOverlay)hideAlert();
  }});
  document.addEventListener("keydown",function(ev){{
    if(ev.key==="Escape"&&!dialogOverlay.classList.contains("hidden"))hideAlert();
  }});
  function showToast(t,b,ms){{
    if(!toastStack)return;
    var el=document.createElement("div");
    el.className="toast-item";
    el.innerHTML="<strong></strong><span></span>";
    el.querySelector("strong").textContent=t||"提示";
    el.querySelector("span").textContent=b||"";
    toastStack.appendChild(el);
    setTimeout(function(){{el.classList.add("fade");}},ms||3200);
    setTimeout(function(){{if(el.parentNode)el.parentNode.removeChild(el);}},(ms||3200)+300);
  }}
  function fillStrategiesOnce(strategies){{
    if(selRed.options.length>0)return;
    strategies.forEach(function(t){{var o=document.createElement("option");o.value=o.textContent=t;selRed.appendChild(o);}});
    strategies.forEach(function(t){{var o=document.createElement("option");o.value=o.textContent=t;selBlack.appendChild(o);}});
  }}
  function srvY(iyVis){{return viewFlipY?(9-iyVis):iyVis;}}
  function visY(iySrv){{return viewFlipY?(9-iySrv):iySrv;}}
  function pctLeft(ix){{return ((EDGE+ix*SQ)/BW*100)+"%";}}
  function pctTop(iyVis){{return ((EDGE+iyVis*SQ)/BH*100)+"%";}}
  function addSprite(cls, ix, iySrv, url){{
    var d=document.createElement("div");
    d.className="spr "+cls;
    d.style.left=pctLeft(ix);
    d.style.top=pctTop(visY(iySrv));
    d.style.backgroundImage="url("+url+")";
    layer.appendChild(d);
  }}
  function boardRenderKey(snap){{
    var sel=snap.sel_from?snap.sel_from.join(","):"";
    var lm=snap.last_move?snap.last_move.join(","):"";
    return (snap.visual_sig||"")+"|"+sel+"|"+lm+"|"+(viewFlipY?"1":"0");
  }}
  function renderBoard(snap){{
    var key=boardRenderKey(snap);
    if(key===lastBoardKey)return;
    lastBoardKey=key;
    layer.replaceChildren();
    for(var iy=0;iy<10;iy++)for(var ix=0;ix<9;ix++){{
      var sq=snap.board[iy][ix];
      if(sq.sprite)addSprite("piece",ix,iy,"/static/xqwl/"+sq.sprite+".png");
    }}
    if(snap.sel_from)addSprite("hl",snap.sel_from[0],snap.sel_from[1],"/static/xqwl/selected.png");
    if(snap.last_move){{
      addSprite("hl",snap.last_move[0],snap.last_move[1],"/static/xqwl/selected.png");
      addSprite("hl",snap.last_move[2],snap.last_move[3],"/static/xqwl/selected.png");
    }}
  }}
  function renderThinkLog(entries){{
    if(!thinkLogBody)return;
    entries=entries||[];
    var key=entries.length+"|"+(entries.length?entries[entries.length-1].text:"");
    if(key===lastThinkLogKey)return;
    lastThinkLogKey=key;
    thinkLogBody.replaceChildren();
    entries.forEach(function(e){{
      var line=document.createElement("div");
      line.className="think-line";
      line.textContent=e.text||"";
      thinkLogBody.appendChild(line);
    }});
    thinkLogBody.scrollTop=thinkLogBody.scrollHeight;
  }}
  function isInputLocked(snap){{
    if(!snap)return true;
    if(typeof snap.input_locked==="boolean")return snap.input_locked;
    return !!(snap.ai_busy||snap.ai_pending);
  }}
  function applySnap(snap){{
    fillStrategiesOnce(snap.strategies||[]);
    selRed.value=snap.strategy_red; selBlack.value=snap.strategy_black;
    if(!userFlipped) viewFlipY=!!snap.view_flip_y;
    updateFlipButton();
    if(chkBook&&bookRow){{
      var avail=!!snap.book_available;
      chkBook.disabled=!avail;
      if(avail)chkBook.checked=!!snap.use_book;
    }}
    statusEl.textContent=snap.status_text||"";
    if(aiThinkingEl){{
      if(snap.ai_thinking)aiThinkingEl.textContent=snap.ai_thinking;
      else if(snap.ai_pending)aiThinkingEl.textContent="小巫师即将应招…";
      else aiThinkingEl.textContent="";
    }}
    renderThinkLog(snap.think_log||[]);
    renderBoard(snap);
    shell.classList.toggle("input-locked",isInputLocked(snap));
    lastSnap=snap;
  }}
  function handleMessages(msg){{
    playSounds(msg.sounds||[]);
    (msg.toasts||[]).forEach(function(t){{
      if(t.kind==="info"){{
        if(t.title==="终局")showAlert(t.title||"提示",t.body||"");
        else showToast(t.title||"提示",t.body||"",3500);
      }}
    }});
  }}
  function armPoll(ms){{if(pollTimer)clearTimeout(pollTimer);pollTimer=setTimeout(onePoll,ms);}}
  function onePoll(){{
    if(pollInFlight){{armPoll(300);return;}}
    pollTimer=null;
    pollInFlight=true;
    Promise.all([fetch("/api/state",{{cache:"no-store"}}).then(function(r){{return r.json();}}),
      fetch("/api/messages",{{cache:"no-store"}}).then(function(r){{return r.json();}})])
      .then(function(pair){{applySnap(pair[0]);handleMessages(pair[1]);armPoll(isInputLocked(pair[0])?120:300);}})
      .catch(function(){{armPoll(700);}})
      .then(function(){{pollInFlight=false;}});
  }}
  function clickFromEvent(ev){{
    var rect=boardEl.getBoundingClientRect();
    var px=(ev.clientX-rect.left)/rect.width*BW;
    var py=(ev.clientY-rect.top)/rect.height*BH;
    var ix=Math.floor((px-EDGE)/SQ);
    var iyVis=Math.floor((py-EDGE)/SQ);
    if(ix<0||ix>8||iyVis<0||iyVis>9)return null;
    return {{ix:ix, iy:srvY(iyVis)}};  // board_view row for server
  }}
  function applyClickResponse(j){{
    if(j.error){{showToast("走子",j.error,2800);return;}}
    if(j.state){{applySnap(j.state);handleMessages(j.messages||{{}});}}
    armPoll(j.state&&isInputLocked(j.state)?120:280);
  }}
  boardEl.addEventListener("click",function(ev){{
    if(shell.classList.contains("input-locked"))return;
    var c=clickFromEvent(ev);
    if(!c)return;
    fetch("/api/click",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(c)}})
      .then(function(r){{return r.json();}})
      .then(applyClickResponse);
  }});
  boardEl.addEventListener("touchstart",function(ev){{
    if(shell.classList.contains("input-locked"))return;
    if(!ev.changedTouches||!ev.changedTouches.length)return;
    ev.preventDefault();
    var t=ev.changedTouches[0];
    var c=clickFromEvent(t);
    if(!c)return;
    fetch("/api/click",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(c)}})
      .then(function(r){{return r.json();}})
      .then(applyClickResponse);
  }},{{passive:false}});
  selRed.addEventListener("change",function(){{
    userFlipped=false;
    fetch("/api/strategies",{{method:"POST",headers:{{"Content-Type":"application/json"}},
      body:JSON.stringify({{red:selRed.value,black:selBlack.value}})}}).then(function(r){{return r.json();}})
      .then(function(j){{if(j.error)showAlert("策略",j.error);else armPoll(20);}});
  }});
  selBlack.addEventListener("change",function(){{
    userFlipped=false;
    fetch("/api/strategies",{{method:"POST",headers:{{"Content-Type":"application/json"}},
      body:JSON.stringify({{red:selRed.value,black:selBlack.value}})}}).then(function(r){{return r.json();}})
      .then(function(j){{if(j.error)showAlert("策略",j.error);else armPoll(20);}});
  }});
  btnNew.addEventListener("click",function(){{
    userFlipped=false;
    lastBoardKey=null;
    lastThinkLogKey=null;
    fetch("/api/new_game",{{method:"POST"}}).then(function(r){{return r.json();}}).then(function(){{armPoll(20);}});
  }});
  function updateFlipButton(){{
    btnFlip.textContent=viewFlipY?"翻转棋盘（还原红下）":"翻转棋盘（己方在下）";
  }}
  btnFlip.addEventListener("click",function(){{
    userFlipped=true;
    viewFlipY=!viewFlipY;
    lastBoardKey=null;
    updateFlipButton();
    if(lastSnap)renderBoard(lastSnap);
  }});
  if(chkBook){{
    chkBook.addEventListener("change",function(){{
      fetch("/api/book",{{method:"POST",headers:{{"Content-Type":"application/json"}},
        body:JSON.stringify({{enabled:chkBook.checked}})}}).then(function(r){{return r.json();}})
        .then(function(j){{if(j.error)showAlert("开局库",j.error);else armPoll(20);}});
    }});
  }}
  onePoll();
}})();
</script>
</body>
</html>"""


def _default_book_path() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "deployment" / "db" / "BOOK.DAT",
        Path.cwd() / "deployment" / "db" / "BOOK.DAT",
        root / "data" / "BOOK.DAT",
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

    if not assets_available():
        raise SystemExit(
            "缺少 XQWL 界面资源。请运行: python scripts/fetch_xqwl_assets.py"
        )

    p = argparse.ArgumentParser(description="MyCChessSF 网页对弈（象棋小巫师 XQWL06）")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=5151)
    p.add_argument("--think-ms", type=int, default=1000, help="小巫师每步思考时间（毫秒）")
    p.add_argument("--book", type=Path, default=None, help="开局库 BOOK.DAT（默认 deployment/db/BOOK.DAT）")
    p.add_argument("--no-book-default", action="store_true", help="启动时默认关闭开局库")
    args = p.parse_args()

    engine = Engine()
    book = args.book if args.book is not None else _default_book_path()
    book_available = False
    if book is not None and book.is_file():
        engine.load_book(str(book))
        book_available = engine.book_size() > 0
        print(f"[play] 已加载开局库: {book} ({engine.book_size()} 项)", flush=True)
    else:
        print("[play] 未找到 BOOK.DAT", flush=True)

    session = XqwlWebSession(
        engine,
        think_ms=int(args.think_ms),
        book_available=book_available,
    )
    session.set_use_book(book_available and not args.no_book_default)

    app = Sanic("mycchess_sf_play_web")
    app.config.RESPONSE_TIMEOUT = max(60, int(args.think_ms) // 500 + 30)
    app.static("/static", str(STATIC_DIR.parent), name="static")

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
        if err and err.get("error"):
            return json(err)
        return json({"state": session.snapshot(), "messages": session.pop_client_messages()})

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
    async def _kick_ai(_app):
        await asyncio.sleep(0.2)
        session.maybe_ai()

    os.environ.setdefault("SANIC_IGNORE_PRODUCTION_WARNING", "1")
    print(f"[play] http://{args.host}:{args.port}/  (象棋小巫师 XQWL06)", flush=True)
    app.run(host=str(args.host), port=int(args.port), single_process=True, access_log=False, motd=False)


if __name__ == "__main__":
    main()
