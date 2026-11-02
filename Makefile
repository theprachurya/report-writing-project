# Convenience entry points; use the equivalent Python commands on Windows if make is unavailable.

PYTHON ?= $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)

.PHONY: audit train-unimodal train-fixed-fusion train-reliability evaluate figures test smoke download-mlaad evaluate-mlaad crossmanip-split web

audit:            ## Build manifests from configured licensed data and audit a sample
	$(PYTHON) scripts/run_data_audit.py --config configs/data_config.yaml

train-unimodal:   ## Train visual-only and audio-only baselines
	$(PYTHON) -m src.train.train_unimodal --modality visual
	$(PYTHON) -m src.train.train_unimodal --modality audio

train-fixed-fusion: ## Train equal-weight fusion before introducing the gate
	$(PYTHON) -m src.train.train_fixed_fusion --mode equal_weight

train-reliability: ## Train the reliability-aware fusion model
	$(PYTHON) -m src.train.train_reliability_fusion

evaluate:         ## Fit calibration/thresholds on validation and evaluate test once
	$(PYTHON) scripts/run_full_evaluation.py --config configs/eval_config.yaml

figures:          ## Generate figures from saved evaluation outputs
	$(PYTHON) scripts/export_poster_figures.py --eval-dir reports/evaluation --output-dir reports/figures

test:             ## Run the CPU unit suite
	$(PYTHON) -m pytest tests/ -q

smoke:            ## Run CPU tests and fixture-only audit (no licensed data required)
	$(PYTHON) -m pytest tests/ -q
	$(PYTHON) scripts/run_data_audit.py --fixture-only --output-dir reports/fixture-audit

download-mlaad:   ## Download the public German MLAAD-tiny audio evaluation slice
	$(PYTHON) scripts/download_mlaad_tiny.py

evaluate-mlaad:   ## Run the audio-only external check with an explicit trained checkpoint
	$(PYTHON) scripts/run_mlaad_audio_eval.py --checkpoint $(CHECKPOINT)

crossmanip-split: ## Build the configured held-out manipulation split before cross-family training
	$(PYTHON) scripts/build_cross_manip_split.py

web:              ## Start the local browser interface with an explicit full AV checkpoint
	$(PYTHON) scripts/serve_web.py --checkpoint $(CHECKPOINT)
