"""Independent random streams, every one derived by np.random.SeedSequence.spawn
from one integer seed (never two generators seeded with the same integer).

tabular.py     the tabular online protocol's streams (E2; moved from experiments/E2_online_4zones/rng.py)
worldmodel.py  the world-model runs' streams (E3: E2's streams first, then E3's own, plus the reference
               condition's root and the measurement slots; moved from experiments/E3_worldmodel_4zones/rng_streams.py)

The two modules keep their own names (both define STREAM_NAMES and spawn_streams) and only worldmodel.py
imports torch, so the tabular experiments never need it.
"""
