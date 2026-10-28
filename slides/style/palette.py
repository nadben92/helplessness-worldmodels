"""Colours and tokens of the progress-presentation figures: the single source of truth.

Every figure script takes its colours from here, never from a literal. Roles are semantic: the same role has the same
colour in every figure.
"""
from __future__ import annotations

# ------------------------------------------------------------------------------------------------ background and text
BG = "#F7F3EA"          # pearl white, figure and axes
TEXT = "#2B2A28"        # main text
TEXT_2 = "#6E675C"      # secondary text
GRID = "#E6DFD2"        # very light grid (horizontal lines only, when useful)
AXIS = "#9C9488"        # thin axes

# ---------------------------------------------------------------------------------------------------- semantic roles
EXPOSED = "#C0583A"     # terracotta: model trained with the uncontrollable zone
REFERENCE = "#3E6A8A"   # slate blue: paired reference
ORACLE = "#5E8C61"      # sage green
ANTI_ORACLE = "#C99A3B" # ochre
CORRECTION = "#6A4268"  # plum: latent correction (A); spec #7D4E7A darkened (deuteranopia vs REFERENCE)
INDUCTION = "#B9798A"   # dusty pink: latent induction (B)
THEORY = "#4A4640"      # dark grey, dotted
CHANCE = "#B8B0A2"      # light grey, dashed

THEORY_LS = (0, (1.2, 2.2))   # dotted
CHANCE_LS = (0, (5, 3))       # dashed

# ---------------------------------------------------------------------------------------------- environment zones
ZONE_CD = "#A9C3D6"     # controllable deterministic
ZONE_UD = "#DB8D72"     # uncontrollable deterministic (conveyor); spec #E3A792 darkened (deuteranopia vs UN)
ZONE_CN = "#A4C5B7"     # controllable noisy; spec #B7D1C6 darkened (deuteranopia vs corridor)
ZONE_UN = "#E5C1A4"     # uncontrollable noisy; spec #E8C9B0 darkened (deuteranopia vs corridor)
ZONE_CORRIDOR = "#E9E1D2"
GOAL = "#D9A441"

ZONE_COLORS = {"CD": ZONE_CD, "UD": ZONE_UD, "CN": ZONE_CN, "UN": ZONE_UN, "corridor": ZONE_CORRIDOR}

# Zone lines on data plots: the zone hue, darkened so it reads as a line on the background (same hue, lower lightness;
# see check_palette.py). Uncontrollable zones in the exposed family, controllable ones in the reference family.
ZONE_LINE = {"UD": EXPOSED, "UN": "#D08A5C", "CD": REFERENCE, "CN": "#5F9A87", "corridor": "#766F63"}

# ------------------------------------------------------------------------------------------------------- typography
FONT_FAMILY = ["Inter", "Source Sans 3", "DejaVu Sans"]
SIZE_TITLE = 21
SIZE_AXIS = 16
SIZE_TICK = 13.5
SIZE_ANNOT = 13
SIZE_SMALL = 11.5

ALPHA_CI = 0.35
