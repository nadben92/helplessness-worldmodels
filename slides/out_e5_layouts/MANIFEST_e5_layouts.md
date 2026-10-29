# E5 shuffle layouts (p1-b-d1)

Files: layout_00..11.png (one per assignment, ASSIGNMENTS order of `pipeline/episode_sources.py`), E5_layouts_grid.png (3x4, row-major).
Script: `slides/out_e5_layouts/plot_e5_layouts.py`.

| idx | UD | CD | neutral | arrow (drow, dcol) | |
|---|---|---|---|---|---|
| 00 | nw | ne | se, sw | (1, 0) |  |
| 01 | nw | se | ne, sw | (1, 0) | **canonical** |
| 02 | nw | sw | ne, se | (1, 0) |  |
| 03 | ne | nw | se, sw | (1, 0) |  |
| 04 | ne | se | nw, sw | (1, 0) |  |
| 05 | ne | sw | nw, se | (1, 0) |  |
| 06 | se | nw | ne, sw | (-1, 0) |  |
| 07 | se | ne | nw, sw | (-1, 0) |  |
| 08 | se | sw | nw, ne | (-1, 0) |  |
| 09 | sw | nw | ne, se | (-1, 0) |  |
| 10 | sw | ne | nw, se | (-1, 0) |  |
| 11 | sw | se | nw, ne | (-1, 0) |  |

- Canonical (E4 and probes): index 1 (UD=nw, CD=se).
- RGB: T_inc / UD = (22, 0, 244), T_ctrl / CD = (227, 17, 15), neutral/corridor = (128, 128, 128).
- Goal: (row, col) = (0, 0), state 0, drawn as the render does (no agent marker).
- Arrows: white, one per UD cell, from the cell towards its successor given by `env.transition_tensor().argmax(-1)` (same for every action, asserted), shortest signed step on the torus, drawn at the cell. Drift convention: `away_from_goal`.
- Gray outlines (0.75, lw 0.5) mark the 4 quadrants (the render does not distinguish the neutral ones from the corridor).
- Render: `envs.observation.allocentric_pixel_backend.render_background` via `pipeline.episode_sources.assignment_world` (k=6, grid_lines=False).
