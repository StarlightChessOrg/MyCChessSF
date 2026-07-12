import re
import sys

path = sys.argv[1]
s = open(path, encoding="utf-8", errors="ignore").read()
for pat in ["botGameState", "localStorage", "gamePlayData", "currentFen", "turnNow", "playerSide"]:
    print(pat, s.count(pat))
print("--- bot events ---")
print(sorted(set(re.findall(r"bot\.[a-z._]+", s))))
for key in ["bot.move", "botGameState", "gamePlayData"]:
    i = s.find(key)
    if i >= 0:
        print(f"\n--- context {key} ---")
        print(s[max(0, i - 120) : i + 280])
