#!/usr/bin/env python3
"""Regenerate cpp/inc/xqwl_extract.inc with English comments from XQWL06 (GB18030)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XQWL06 = ROOT.parent / "backups" / "xqwlight_source" / "Win32" / "XQWL06.CPP"
OUT = ROOT / "cpp" / "inc" / "xqwl_extract.inc"
CURRENT = ROOT / "cpp" / "inc" / "xqwl_extract.inc"

# Simplified Chinese comment -> English (XQWL06 position core)
TR = {
    "判断棋子是否在棋盘中的数组": "Square-on-board lookup table (256-byte board index)",
    "判断棋子是否在九宫中的数组": "Palace (fort) membership lookup table",
    "判断步长是否符合特定走法的数组，1=帅(将)，2=仕(士)，3=相(象)": "Legal step span table: 1=king, 2=advisor, 3=bishop",
    "根据步长判断马是否蹩腿的数组": "Knight leg-block offset table",
    "帅(将)的步长": "King step deltas",
    "仕(士)的步长": "Advisor step deltas",
    "马的步长，以帅(将)的步长作为马腿": "Knight leaps (king delta used as leg check)",
    "马被将军的步长，以仕(士)的步长作为马腿": "Knight check deltas (advisor delta as leg)",
    "棋盘初始设置": "Standard starting layout",
    "子力位置价值表": "Piece-square evaluation tables",
    "帅(将)": "King",
    "仕(士)": "Advisor",
    "相(象)": "Bishop",
    "马": "Knight",
    "车": "Rook",
    "炮": "Cannon",
    "兵(卒)": "Pawn",
    "判断棋子是否在棋盘中": "Whether a square index is on the board",
    "判断棋子是否在九宫中": "Whether a square is inside a palace",
    "获得格子的横坐标": "Rank (y) of a square index",
    "获得格子的纵坐标": "File (x) of a square index",
    "根据纵坐标和横坐标获得格子": "Pack file/rank into square index",
    "翻转格子": "Flip square vertically (mirror rank)",
    "纵坐标水平镜像": "Mirror file coordinate",
    "横坐标垂直镜像": "Mirror rank coordinate",
    "格子水平镜像": "Mirror square horizontally (file flip)",
    "走法是否符合帅(将)的步长": "King step span check",
    "走法是否符合仕(士)的步长": "Advisor step span check",
    "走法是否符合相(象)的步长": "Bishop step span check",
    "相(象)眼的位置": "Bishop blocking eye square",
    "马腿的位置": "Knight leg square",
    "是否未过河": "Square on home side of river",
    "是否已过河": "Square on away side of river",
    "是否在河的同一边": "Both squares on same river half",
    "是否在同一行": "Same rank",
    "是否在同一列": "Same file",
    "获得红黑标记(红子是8，黑子是16)": "Side tag (red=8, black=16)",
    "获得对方红黑标记": "Opponent side tag",
    "获得走法的起点": "Move source square",
    "获得走法的终点": "Move destination square",
    "根据起点和终点获得走法": "Pack move from src/dst",
    "走法水平镜像": "Mirror move horizontally",
    "RC4密码流生成器": "RC4 keystream generator (Zobrist init)",
    "用空密钥初始化密码流生成器": "Initialize RC4 with empty key",
    "生成密码流的下一个字节": "Next RC4 byte",
    "生成密码流的下四个字节": "Next four RC4 bytes as DWORD",
    "Zobrist结构": "Zobrist hash triple",
    "用零填充Zobrist": "Zero Zobrist keys",
    "用密码流填充Zobrist": "Fill Zobrist from RC4",
    "执行XOR操作": "XOR Zobrist keys",
    "Zobrist表": "Global Zobrist tables",
    "初始化Zobrist表": "Initialize Zobrist tables",
    "历史走法信息(占4字节)": "Move history entry (4 bytes)",
    "局面结构": "Position state",
    "轮到谁走，0=红方，1=黑方": "Side to move: 0=red, 1=black",
    "棋盘上的棋子": "Board piece bytes",
    "红、黑双方的子力价值": "Material scores for red/black",
    "距离根节点的步数，历史走法数": "Search ply and move-list length",
    "历史走法信息列表": "Move history stack",
    "清空棋盘": "Clear board and scores",
    "清空(初始化)历史走法信息": "Reset irreversible move list",
    "初始化棋盘": "Set standard start position",
    "交换走子方": "Switch side to move",
    "在棋盘上放一枚棋子": "Place piece and update hash/score",
    "红方加分，黑方(注意\"cucvlPiecePos\"取值要颠倒)减分": "Red adds PST; black uses flipped square index",
    "从棋盘上拿走一枚棋子": "Remove piece and update hash/score",
    "红方减分，黑方(注意\"cucvlPiecePos\"取值要颠倒)加分": "Red subtracts PST; black uses flipped index",
    "局面评价函数": "Static evaluation from side to move",
    "是否被将军": "Was side to move in check after last move",
    "上一步是否吃子": "Was last move a capture",
    "搬一步棋的棋子": "Apply move on board (no side change)",
    "撤消搬一步棋的棋子": "Undo board move",
    "走一步棋": "Make legal move (reject if leaves king in check)",
    "撤消走一步棋": "Undo one ply",
    "走一步空步": "Null move for pruning",
    "撤消走一步空步": "Undo null move",
    "生成所有走法，如果\"bCapture\"为\"TRUE\"则只生成吃子走法": "Generate pseudo-legal moves; captures only if bCapture",
    "判断走法是否合理": "Pseudo-legal move geometry (no check test)",
    "判断是否被将军": "Is side to move in check now",
    "判断是否被杀": "Is checkmate",
    "和棋分值": "Draw score by ply parity",
    "检测重复局面": "Repetition status for Chinese rules",
    "重复局面分值": "Score for repetition adjudication",
    "判断是否允许空步裁剪": "Null-move pruning material threshold",
    "对局面镜像": "Mirror position horizontally (opening book)",
    '"GenerateMoves"参数': "GenerateMoves flag",
    "生成所有走法，需要经过以下几个步骤：": "Move generation steps:",
    "1. 找到一枚本方棋子，并根据棋子判断：": "1. Iterate own pieces",
    "2. 根据棋子确定走法": "2. Emit moves by piece type",
    "判断走法是否合法": "Pseudo-legal move test",
    "判断走法是否合法，需要经过以下的判断过程：": "LegalMove checks:",
    "1. 判断起始点是否有本方棋子": "1. Source must be own piece",
    "2. 判断目标点是否有本方棋子": "2. Destination must not be own piece",
    "3. 根据棋子的类型检查走法是否合理": "3. Piece-specific geometry",
    "找到棋盘上的帅(将)，再做以下判断：": "Locate own king, then:",
    "1. 判断是否被对方的兵(卒)将军": "1. Pawn checks",
    "2. 判断是否被对方的马将军(以仕(士)的步长当作马腿)": "2. Knight checks (advisor delta as leg)",
    "3. 判断是否被对方的车或炮将军(包括将帅对脸)": "3. Rook/cannon/king-face checks",
    "检测重复局面": "Detect repetition along move list",
    "对局面镜像": "Build horizontally mirrored copy",
}

EXTRA = {
    "历史着法栈；原 256 不足以覆盖长棋谱（IMSA 等），越界写 mvsList 会触发 stack smashing / 堆损坏":
        "Move stack; 256 was too small for long games — use MAX_MOVES from prefix.h",
}


def translate_comment(text: str) -> str:
    t = text.strip()
    if t in EXTRA:
        return EXTRA[t]
    if t in TR:
        return TR[t]
    for zh, en in TR.items():
        if zh in t:
            t = t.replace(zh, en)
    # inline trailing comments on struct fields
    m = re.match(r"^(.+?)\s*//\s*(.+)$", t)
    if m:
        body, cmt = m.group(1), m.group(2).strip()
        en = TR.get(cmt, translate_comment(cmt) if cmt not in TR else TR[cmt])
        if cmt in TR:
            return f"{body}  // {TR[cmt]}"
    return t


def translate_line(line: str) -> str:
    if "//" not in line:
        return line
    if line.strip().startswith("//"):
        cmt = line.split("//", 1)[1].strip()
        en = TR.get(cmt, None)
        if en:
            indent = line[: line.index("//")]
            return f"{indent}// {en}"
        return line
    # trailing comment
    idx = line.index("//")
    code, cmt = line[:idx], line[idx + 2 :].strip()
    en = TR.get(cmt)
    if en:
        return f"{code}// {en}"
    return line


def main() -> None:
    text = XQWL06.read_bytes().decode("gb18030")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "ccInBoard[256]" in l) - 1
    end = next(i for i, l in enumerate(lines) if l.strip().startswith("static PositionStruct pos"))
    chunk = lines[start:end]

    out_lines: list[str] = [
        "// XQWL06 position core (portable extract; English comments)",
        "// Source: backups/xqwlight_source/Win32/XQWL06.CPP",
        "",
    ]
    for line in chunk:
        out_lines.append(translate_line(line))

    # Apply MyCChessSF patches from current file (push_mv, MakeMove guard)
    body = "\n".join(out_lines)
    body = body.replace(
        "BOOL PositionStruct::MakeMove(int mv) {\n  int pcCaptured;",
        "BOOL PositionStruct::MakeMove(int mv) {\n  if (nMoveNum >= MAX_MOVES) {\n    return FALSE;\n  }\n  int pcCaptured;",
    )
    if "const auto push_mv" not in body:
        body = body.replace(
            "  nGenMoves = 0;\n  pcSelfSide = SIDE_TAG(sdPlayer);",
            "  nGenMoves = 0;\n  const auto push_mv = [&](int mv) -> bool {\n"
            "    if (nGenMoves >= MAX_GEN_MOVES) {\n      return false;\n    }\n"
            "    mvs[nGenMoves++] = mv;\n    return true;\n  };\n"
            "  pcSelfSide = SIDE_TAG(sdPlayer);",
        )
        body = re.sub(
            r"mvs\[nGenMoves\+\+\] = MOVE\(sqSrc, sqDst\);\s*\n(\s*)}",
            r"if (!push_mv(MOVE(sqSrc, sqDst))) {\n\1  return nGenMoves;\n\1}",
            body,
        )

    OUT.write_text(body.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
