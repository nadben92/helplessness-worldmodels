"""Zone layouts built from the shared YAML blocks (configs/zones/), for the zoned torus (TorusZonedEnv).

four_zones.py        E1/E2's 4-zone layout with the counterbalancing permutation and the derived reference
                     (moved from experiments/E2_online_4zones/env.py)
four_zones_pixel.py  the same layout rendered as pixels with a per-run theme, and E3's seed -> rotation -> block
                     class rule (moved from experiments/E3_worldmodel_4zones/env.py)

New layouts (E4's two zones, the test environments) are new modules here; these two never change.
"""
