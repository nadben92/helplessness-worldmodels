"""Colours of the v2 slide figures (saturated): the single source of truth for figures_v2/. Same semantics everywhere."""

EXPOSED = "#E8412C"      # vermilion
REFERENCE = "#1F6FEB"    # bright blue
ORACLE = "#1DB954"       # bright green
ANTI_ORACLE = "#FF9F1C"  # bright orange
CORRECTION = "#6718AB"   # violet: latent correction (A); spec #8E2DE2 darkened (deuteranopia vs REFERENCE: 3.1 -> 19.7)
INDUCTION = "#FF2E88"    # hot pink: latent induction (B)
THEORY = "#000000"       # black, dotted
CHANCE = "#888888"       # grey, dashed
TEXT = "#000000"

THEORY_LS = ":"
CHANCE_LS = "--"
LW = 2.8
MS = 9

ZONE_CD = "#4DA3FF"
ZONE_UD = "#FF5A36"
ZONE_CN = "#00C2A8"
ZONE_UN = "#FFB347"
ZONE_CORRIDOR = "#D9D9D9"
GOAL = "#FFD60A"
ZONE_COLORS = {"CD": ZONE_CD, "UD": ZONE_UD, "CN": ZONE_CN, "UN": ZONE_UN, "corridor": ZONE_CORRIDOR}

# Zone lines on data plots: the zone colours themselves; the corridor (light grey fill) is drawn darker as a line.
ZONE_LINE = {"UD": ZONE_UD, "UN": ZONE_UN, "CD": ZONE_CD, "CN": ZONE_CN, "corridor": "#6E6E6E"}

SIZE_LABEL = 18
SIZE_TICK = 15
SIZE_BLOCK = 17  # block / zone / category names in schematics
