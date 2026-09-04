.PHONY: install test agents sweep swaps intervention rescue all-experiments

install:
	pip install -e .

test:
	pytest

# Regenerates results/agents/ (gitignored, see CONTRIBUTING.md) from scratch.
agents:
	python scripts/regenerate_agents.py

sweep: agents
	cd experiments/01_offline_sweep && python eval_v2.py --config configs/eval_v2_5000.yaml
	cd experiments/01_offline_sweep && python eval_v2.py --config configs/eval_v2_50000.yaml

swaps:
	cd experiments/02_swaps && python run.py --config config.yaml

intervention:
	cd experiments/03_intervention && python run.py --config config.yaml
	cd experiments/03_intervention && python rescue_diagnostics.py --config rescue_diagnostics_config.yaml

rescue:
	cd experiments/04_rescue_injection && python run.py --config config.yaml

all-experiments: sweep swaps intervention rescue
