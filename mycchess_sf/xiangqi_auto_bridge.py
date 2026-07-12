"""Chess98-compatible HTTP bridge (default :9494) for 相弈象棋 Selenium auto tests."""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mycchess_sf.bridge_codec import bridge4_to_iccs, iccs_to_computer_token
from mycchess_sf.xqwl_state import XqwlGameState


def _deploy_root() -> Path | None:
    lib = Path(__file__).resolve().parent.parent
    if (lib.parent / "db").is_dir() and (lib / "mycchess_sf").is_dir():
        return lib.parent
    return None


def _default_nnue_path() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    deploy = _deploy_root()
    candidates: list[Path] = []
    if deploy is not None:
        candidates.append(deploy / "model" / "quantized.xqnnue.bin")
    candidates.extend(
        [
            root / "deployment" / "model" / "quantized.xqnnue.bin",
            Path.cwd() / "deployment" / "model" / "quantized.xqnnue.bin",
            root / "data" / "nnue_model" / "quantized.xqnnue.bin",
            Path.cwd() / "data" / "nnue_model" / "quantized.xqnnue.bin",
            root / "nnue_qINT8" / "output" / "quantized.xqnnue.bin",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def _default_book_path() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    deploy = _deploy_root()
    candidates: list[Path] = []
    if deploy is not None:
        candidates.append(deploy / "db" / "BOOK.DAT")
    candidates.extend(
        [
            root / "deployment" / "db" / "BOOK.DAT",
            Path.cwd() / "deployment" / "db" / "BOOK.DAT",
            root / "data" / "BOOK.DAT",
            Path.cwd() / "data" / "BOOK.DAT",
        ]
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


class BridgeState:
    def __init__(
        self,
        *,
        think_ms: int,
        use_book: bool,
        use_nnue: bool,
        nnue_path: Path | None,
        book_path: Path | None,
        engine_red: bool,
    ) -> None:
        from xqwlight_core import Engine

        self._lock = threading.Lock()
        self._think_ms = max(50, int(think_ms))
        self._use_book = bool(use_book)
        self._engine_red = bool(engine_red)
        self._engine = Engine()
        self._game = XqwlGameState()
        self.computer_move = "null"
        self.board_code = "null"
        self._last_opponent_token = ""
        self._pending_opponent: str | None = None
        self._stop = threading.Event()

        if book_path is not None and book_path.is_file():
            self._engine.load_book(str(book_path))
        if use_nnue and nnue_path is not None and nnue_path.is_file():
            if not self._engine.load_nnue(str(nnue_path)):
                print(f"[bridge] NNUE load failed: {nnue_path}", flush=True)
        elif use_nnue:
            print("[bridge] NNUE enabled but no weights found", flush=True)

    def stop(self) -> None:
        self._stop.set()

    def current_fen(self) -> str:
        return self._game.fen()

    def _engine_turn(self) -> bool:
        side = self._game.get_side()
        return (side == "red") == self._engine_red

    def reset_session(self) -> None:
        with self._lock:
            self._game.reset()
            self.computer_move = "null"
            self._last_opponent_token = ""
            self._pending_opponent = None
        print("[bridge] reset -> new game", flush=True)

    def sync_fen(self, fen: str) -> None:
        """以网页 FEN 为唯一局面来源；轮到引擎时搜索并更新 computer_move。"""
        fen = fen.strip()
        with self._lock:
            if not self._game.set_fen(fen):
                raise ValueError(f"Invalid FEN: {fen!r}")
            self.computer_move = "null"
            self._last_opponent_token = ""
            self._pending_opponent = None
            engine_turn = self._engine_turn()
        if engine_turn and self._game.legal_moves_iccs_str():
            self._run_engine_search(apply_internal=False)

    def _run_engine_search(self, *, apply_internal: bool = True) -> None:
        with self._lock:
            if not self._engine_turn() or self.computer_move != "null":
                return
            if not self._game.legal_moves_iccs_str():
                return
            game_copy = self._game.copy()
            start_fen = self._game.fen()

        detail = self._engine.search_best_detail(
            game_copy.raw_position(), self._think_ms, self._use_book
        )
        iccs = str(detail.get("iccs", "") or "")
        if not iccs:
            return

        with self._lock:
            if not self._engine_turn() or self.computer_move != "null":
                return
            if apply_internal and self._game.fen() != start_fen:
                return
            if apply_internal:
                if not self._game.make_move_iccs(iccs):
                    print(f"[bridge] engine move failed to apply: {iccs!r}", flush=True)
                    return
            self.computer_move = iccs_to_computer_token(iccs)
            depth = int(detail.get("depth", 0))
            score = int(detail.get("score", 0))
            book = bool(detail.get("from_book", False))
            tag = "book" if book else f"d{depth}"
            print(f"[bridge] engine -> {iccs} ({tag}, vl={score})", flush=True)

    def _search_engine_move(self) -> None:
        self._run_engine_search(apply_internal=True)

    def _apply_opponent_token(self, token: str) -> None:
        """Legacy Chess98 着法 token；新流程请用 sync_fen。"""
        iccs = bridge4_to_iccs(token)
        legal = set(self._game.legal_moves_iccs_str())
        if iccs not in legal:
            side = self._game.get_side()
            ply = self._game.ply_count()
            sample = ", ".join(sorted(legal)[:12])
            raise ValueError(
                f"Illegal opponent move {token!r} -> {iccs!r} "
                f"(side={side}, ply={ply}; legal sample: {sample})"
            )
        self._game.make_move_iccs(iccs)
        self.computer_move = "null"
        self._last_opponent_token = token
        print(f"[bridge] opponent -> {iccs}", flush=True)

    def submit_opponent_move(self, token: str) -> str | None:
        token = token.strip()
        if not token or token in {"wait", "undo", "____"}:
            return None
        with self._lock:
            if token == self._last_opponent_token:
                return None
            try:
                self._apply_opponent_token(token)
                need_search = self._engine_turn() and self.computer_move == "null"
            except ValueError as exc:
                return str(exc)
        if need_search and self._game.legal_moves_iccs_str():
            self._run_engine_search()
        return None

    def request_undo(self) -> None:
        with self._lock:
            if self._game.ply_count() >= 2:
                self._game.undo_moves(2)
                self.computer_move = "null"
                self._last_opponent_token = ""
                print("[bridge] undo 2 plies", flush=True)

    def loop(self) -> None:
        """Legacy：仅处理 reset 后首轮搜索；对手着法由 sync_fen 驱动。"""
        while not self._stop.is_set():
            need_search = False
            with self._lock:
                if (
                    self._engine_turn()
                    and self.computer_move == "null"
                    and self._game.legal_moves_iccs_str()
                ):
                    need_search = True
            if need_search:
                try:
                    self._run_engine_search()
                except Exception as exc:
                    print(f"[bridge] error: {exc}", flush=True)
            time.sleep(0.1)


class _BridgeHandler(BaseHTTPRequestHandler):
    state: BridgeState

    def log_message(self, fmt: str, *args) -> None:  # noqa: ARG002
        return

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, PUT, DELETE, PATCH, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _text(self, code: int, body: str) -> None:
        try:
            data = body.encode("utf-8")
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # Selenium 轮询 /computer 时常提前关闭连接，可忽略
            return

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path == "/boardcode":
            self._text(200, self.state.board_code + "\n")
            return
        if path == "/computer":
            self._text(200, self.state.computer_move + "\n")
            return
        if path == "/undo":
            self.state.request_undo()
            self._text(200, "successful\n")
            return
        if path == "/reset":
            self.state.reset_session()
            self._text(200, "successful\n")
            return
        if path == "/sync" and "fen" in qs and qs["fen"]:
            from urllib.parse import unquote_plus

            fen = unquote_plus(qs["fen"][0])
            try:
                self.state.sync_fen(fen)
            except ValueError as exc:
                self._text(409, f"error: {exc}\n")
                return
            self._text(200, self.state.computer_move + "\n")
            return
        if path == "/fen":
            self._text(200, self.state.current_fen() + "\n")
            return
        if "move" in path or "move" in parsed.query or "playermove" in qs:
            token = ""
            if "playermove" in qs and qs["playermove"]:
                token = qs["playermove"][0]
            elif parsed.query:
                token = parsed.query.split("=", 1)[-1]
            err = self.state.submit_opponent_move(token)
            if err:
                self._text(409, f"error: {err}\n")
                return
            self._text(200, self.state.computer_move + "\n")
            return
        self._text(404, "not found\n")

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parsed.query
        if "boardcode" in qs:
            self.state.board_code = qs.split("=", 1)[-1]
            self._text(200, "successful\n")
            return
        if "move" in qs:
            self.state.computer_move = qs.split("=", 1)[-1]
            self._text(200, "successful\n")
            return
        self._text(404, "not found\n")


def main() -> None:
    p = argparse.ArgumentParser(description="象眸 SF 相弈自动对弈桥（Chess98 兼容 HTTP :9494）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9494)
    p.add_argument("--think-ms", type=int, default=1000, help="引擎每步思考毫秒")
    p.add_argument("--book", type=Path, default=None, help="开局库 BOOK.DAT")
    p.add_argument("--no-book", action="store_true", help="禁用开局库")
    p.add_argument("--nnue", type=Path, default=None, help="NNUE 权重路径")
    p.add_argument("--no-nnue-default", action="store_true", help="默认不用 NNUE")
    p.add_argument(
        "--engine-color",
        choices=("red", "black"),
        default="red",
        help="引擎在相弈网页上执棋颜色（默认红方，与 Chess98 UI 模式一致）",
    )
    args = p.parse_args()

    book = None if args.no_book else (args.book or _default_book_path())
    nnue = None if args.no_nnue_default else (args.nnue or _default_nnue_path())
    state = BridgeState(
        think_ms=int(args.think_ms),
        use_book=book is not None and book.is_file(),
        use_nnue=nnue is not None and nnue.is_file(),
        nnue_path=nnue,
        book_path=book,
        engine_red=args.engine_color == "red",
    )
    worker = threading.Thread(target=state.loop, daemon=True)
    worker.start()

    handler_cls = type("Handler", (_BridgeHandler,), {"state": state})
    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler_cls)
    print(
        f"[bridge] http://{args.host}:{args.port}/  "
        f"(engine={args.engine_color}, think={args.think_ms}ms)",
        flush=True,
    )
    print(
        "[bridge] 等待 Selenium 连接（本进程不会打开 play.xiangqi.com）",
        flush=True,
    )
    print(
        "[bridge] Windows: cd deployment\\tools\\auto && npm install && npm start",
        flush=True,
    )
    print(
        "[bridge] 诊断: bash scripts/diagnose_xiangqi_auto.sh",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] stopped", flush=True)
    finally:
        state.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
