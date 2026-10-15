# Convenience entry points; use the equivalent Python commands on Windows if make is unavailable.

.PHONY: audit train-unimodal train-fixed-fusion train-reliability evaluate figures test smoke

audit:            ## Build manifests from configured licensed data and audit a sample
	python scripts/run_data_audit.py --config configs/data_config.yaml

train-unimodal:   ## Train visual-only and audio-only baselines
	python -m src.train.train_unimodal --modality visual
	python -m src.train.train_unimodal --modality audio

train-fixed-fusion: ## Train equal-weight fusion before introducing the gate
	python -m src.train.train_fixed_fusion --mode equal_weight

train-reliability: ## Train the reliability-aware fusion model
	python -m src.train.train_reliability_fusion

evaluate:         ## Fit calibration/thresholds on validation and evaluate test once
	python scripts/run_full_evaluation.py --config configs/eval_config.yaml

figures:          ## Generate figures from saved evaluation outputs
	python scripts/export_poster_figures.py --eval-dir reports/evaluation --output-dir reports/figures

test:             ## Run the CPU unit suite
	python -m pytest tests/ -q

smoke:            ## Run CPU tests and fixture-only audit (no licensed data required)
	python -m pytest tests/ -q
	python scripts/run_data_audit.py --output-dir reports/fixture-audit
