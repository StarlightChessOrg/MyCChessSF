import re
import sys

s = open(sys.argv[1], encoding="utf-8", errors="ignore").read()

# bot.move emit
for m in re.finditer(r"bot\.move", s):
    ctx = s[m.start() : m.start() + 500]
    if "emit" in ctx or "fen" in ctx:
        print("=== bot.move emit context ===")
        print(ctx[:500])
        print()

# localStorage botGameState usage
for m in re.finditer(r"botGameState", s):
    print("=== botGameState usage ===")
    print(s[max(0, m.start() - 150) : m.start() + 400])
    print()

# moves uci pattern
for m in re.finditer(r"uci", s):
    ctx = s[max(0, m.start() - 80) : m.start() + 120]
    if "move" in ctx.lower():
        print("=== uci move context ===")
        print(ctx)
        print()
        break

# guest_jwt
for pat in ["guest_jwt", "api.xiangqi.com", "users/guest"]:
    i = s.find(pat)
    if i >= 0:
        print(f"=== {pat} ===")
        print(s[i : i + 200])
        print()
