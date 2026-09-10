# World-model agent (E3+)

Reserved, empty until E3. Whatever gets built here must implement
`helplessness.agents.base.Agent` -- specifically `transition_distribution(s, a)`
-- so it can be scored by `helplessness.measures.controllability.controllability_from_tensor`
via `helplessness.agents.base.materialize_transition_tensor` exactly the way
the tabular agents already are (see `tests/agents/test_base.py`).

Not a Python package yet (no `__init__.py`): there is nothing to import.
Add one when the first concrete world-model agent is written.
