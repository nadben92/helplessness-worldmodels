"""Colour-vision check of the palette: CIELAB distance between every pair of colours that share a figure, in normal
vision and under simulated deuteranopia (Machado 2009). Pairs under the threshold are listed; the fix is a lightness
change, never a hue change (documented in MANIFEST.md).

    python slides/style/check_palette.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style import palette as P  # noqa: E402
from style.common import deuteranopia, delta_e, hex_rgb  # noqa: E402

THRESHOLD = 12.0  # deltaE76 under which two marks are hard to tell apart at slide size

GROUPS = {
    "roles": {"exposed": P.EXPOSED, "reference": P.REFERENCE, "oracle": P.ORACLE, "anti-oracle": P.ANTI_ORACLE,
              "correction A": P.CORRECTION, "induction B": P.INDUCTION, "theory": P.THEORY, "chance": P.CHANCE},
    "zones": {"CD": P.ZONE_CD, "UD": P.ZONE_UD, "CN": P.ZONE_CN, "UN": P.ZONE_UN, "corridor": P.ZONE_CORRIDOR,
              "goal": P.GOAL},
    "zone lines": dict(P.ZONE_LINE),
}


def main() -> int:
    bad = 0
    for group, cols in GROUPS.items():
        print(f"== {group}")
        for (na, a), (nb, b) in itertools.combinations(cols.items(), 2):
            d_n = delta_e(hex_rgb(a), hex_rgb(b))
            d_d = delta_e(deuteranopia(a), deuteranopia(b))
            flag = "  <-- close" if d_d < THRESHOLD else ""
            bad += bool(flag)
            print(f"  {na:>13} / {nb:<13} normal {d_n:5.1f}   deuteranopia {d_d:5.1f}{flag}")
        for n, c in cols.items():
            print(f"  {n:>13} vs background: deuteranopia {delta_e(deuteranopia(c), deuteranopia(P.BG)):5.1f}")
    print(f"{bad} close pair(s) under deuteranopia (threshold {THRESHOLD})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
