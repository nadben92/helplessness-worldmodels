"""Deuteranopia check of the v2 palette (Machado 2009, CIELAB deltaE76, threshold 12), pairs that share a figure.

    .venv/Scripts/python.exe slides/style_v2/check_palette.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style.common import deuteranopia, delta_e, hex_rgb  # noqa: E402  (pure colour functions of v1)
from style_v2 import palette as P  # noqa: E402

THRESHOLD = 12.0
GROUPS = {
    "roles": {"exposed": P.EXPOSED, "reference": P.REFERENCE, "oracle": P.ORACLE, "anti-oracle": P.ANTI_ORACLE,
              "correction A": P.CORRECTION, "induction B": P.INDUCTION, "chance": P.CHANCE},
    "zones": {"CD": P.ZONE_CD, "UD": P.ZONE_UD, "CN": P.ZONE_CN, "UN": P.ZONE_UN, "corridor": P.ZONE_CORRIDOR,
              "goal": P.GOAL},
    "zone lines": dict(P.ZONE_LINE),
}


def main() -> int:
    close = 0
    for group, cols in GROUPS.items():
        print(f"== {group}")
        for (na, a), (nb, b) in itertools.combinations(cols.items(), 2):
            d = delta_e(deuteranopia(a), deuteranopia(b))
            flag = d < THRESHOLD
            close += flag
            print(f"  {na:>13} / {nb:<13} normal {delta_e(hex_rgb(a), hex_rgb(b)):5.1f}  deuteranopia {d:5.1f}"
                  f"{'  <-- close' if flag else ''}")
    print(f"{close} close pair(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
