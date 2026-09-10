"""AppearanceSpec: a zone's visual identity, independent of its dynamics.

A Zone's appearance is just a category label (e.g. "A", "B", "reference")
-- a stable slot that the theme system (themes.py) turns into an actual
visual (colour/texture) given a theme seed. Deliberately holds no dynamics
information and no rendering logic itself: rendering must be a pure
function of (category, theme_seed), enforced by keeping this a plain data
holder rather than a class with its own render() that could be tempted to
peek at a zone's dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppearanceSpec:
    category: str
