#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nnue_training"))

from features.half_ka_v2_hm import fen_to_feature_indices_both_perspectives
from mycchess_sf.fen_parse import FULL_INIT_FEN

dump = ROOT / "third_party" / "halfka_dump"
proc = subprocess.run(
    [str(dump)],
    input=FULL_INIT_FEN + "\n",
    capture_output=True,
    text=True,
    check=True,
)
print(proc.stdout)

stm_py, other_py = fen_to_feature_indices_both_perspectives(FULL_INIT_FEN)
print("py stm", sorted(int(x) for x in stm_py))
print("py other", sorted(int(x) for x in other_py))
