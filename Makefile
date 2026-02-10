# FinSentry - one-command setup and run
# Run from repo root: make setup && make run. Venv: backend/.venv (see README).

ROOT := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
PY ?= python3
BACKEND = backend
SCRIPTS = scripts
VENV = $(BACKEND)/.venv
PY_VENV = $(VENV)/bin/python
UVICORN_VENV = $(VENV)/bin/uvicorn

# Use venv binaries if venv exists, else system (ROOT-prefixed for use after cd)
PY_RUN := $(if $(wildcard $(ROOT)$(PY_VENV)),$(ROOT)$(PY_VENV),$(PY))
UVICORN_RUN := $(if $(wildcard $(ROOT)$(UVICORN_VENV)),$(ROOT)$(UVICORN_VENV),uvicorn)

.PHONY: setup run test-audio test-finance transcribe prepare-finance-dataset train-finance demo demo-medium demo-large bootstrap help

help:
	@echo "FinSentry backend targets:"
	@echo "  make setup      - Install deps (uv or pip) and run bootstrap"
	@echo "  make run        - Start API with uvicorn (backend must be set up)"
	@echo "  make test-audio - Run AI-1 audio pipeline demo"
	@echo "  make test-finance - Run AI-2 finance pipeline demo"
	@echo "  make transcribe - Batch-transcribe .wav in data/new_calls/wav (use ARGS='--overwrite' to overwrite)"
	@echo "  make prepare-finance-dataset - Build finance_train.jsonl from transcripts + labels (see README)"
	@echo "  make train-finance - LoRA train AI-2 (default: extra-dataset only; use ARGS for --data, --epochs)"
	@echo "  make demo        - Run hackathon demo script (use AUDIO=/path/file)"
	@echo "  make demo-medium - Same as demo (Whisper medium)"
	@echo "  make demo-large  - Demo with Whisper large-v2 (slower, higher quality)"
	@echo "  make bootstrap  - Check Python, ffmpeg, env vars only"
	@echo "  Venv: $(VENV) (used when present)"

setup:
	@if command -v uv >/dev/null 2>&1; then \
		cd $(ROOT)$(BACKEND) && uv sync && echo "Deps installed (uv)."; \
	else \
		([ -d $(ROOT)$(VENV) ] || $(PY) -m venv $(ROOT)$(VENV)) && \
		$(ROOT)$(PY_VENV) -m pip install -q -r $(ROOT)requirements.txt && echo "Deps installed (pip into $(VENV))."; \
	fi
	@$(PY_RUN) $(ROOT)$(SCRIPTS)/bootstrap.py

bootstrap:
	@$(PY_RUN) $(ROOT)$(SCRIPTS)/bootstrap.py

run:
	cd $(ROOT)$(BACKEND) && HF_HOME=$(ROOT)$(BACKEND)/.cache/huggingface HF_HUB_DISABLE_XET=1 $(UVICORN_RUN) main:app --reload --host 0.0.0.0 --port 8000 --reload-exclude "*.cache/*" --reload-exclude "*/.cache/*" --reload-exclude "*/__pycache__/*"

test-audio:
	cd $(ROOT)$(BACKEND) && HF_HOME=$(ROOT)$(BACKEND)/.cache/huggingface HF_HUB_DISABLE_XET=1 $(PY_RUN) $(ROOT)$(SCRIPTS)/test_audio_pipeline.py

test-finance:
	$(PY_RUN) $(ROOT)$(SCRIPTS)/test_finance_pipeline.py

transcribe:
	$(PY_RUN) $(ROOT)$(SCRIPTS)/transcribe_new_calls.py $(ARGS)

prepare-finance-dataset:
	$(PY_RUN) $(ROOT)$(SCRIPTS)/prepare_finance_training_data.py

train-finance:
	@mkdir -p $(ROOT)$(BACKEND)/.cache/huggingface
	cd $(ROOT)$(BACKEND) && HF_HOME=$(ROOT)$(BACKEND)/.cache/huggingface HF_HUB_DISABLE_XET=1 $(PY_RUN) -m ai.finance.training.train \
		--extra-dataset data/new_calls/finance_train.jsonl \
		--output models/finance_v2 \
		$(ARGS)

demo:
	@if [ -z "$(AUDIO)" ]; then \
		echo "Usage: make demo AUDIO=/path/to/file.(wav|mp3|m4a)"; \
		exit 1; \
	fi
	$(PY_RUN) $(ROOT)$(SCRIPTS)/judge_demo.py --model-size=medium "$(AUDIO)"

demo-medium: demo

demo-large:
	@if [ -z "$(AUDIO)" ]; then \
		echo "Usage: make demo-large AUDIO=/path/to/file.(wav|mp3|m4a)"; \
		exit 1; \
	fi
	$(PY_RUN) $(ROOT)$(SCRIPTS)/judge_demo.py --model-size=large-v2 "$(AUDIO)"
