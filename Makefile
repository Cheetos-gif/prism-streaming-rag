# PRISM - Streaming Live RAG
#
# Cross-platform Makefile for GNU Make 3.81+ on Linux, macOS and Windows.
#
#   Linux / macOS : make setup && make run
#   Windows       : winget install GnuWin32.Make   (or: choco install make)
#                   make setup ; make run
#
# Recipes stick to python -c for file operations and avoid shell builtins
# (rm, mkdir, touch, cp) plus percent signs, which cmd.exe expands. The same
# lines therefore run unchanged under sh, cmd.exe, PowerShell and MSYS bash.
#
# Overridable variables:
#   PYTHON=...         interpreter used to build the venv
#                      make setup PYTHON=python3.12   (Linux)
#                      make setup PYTHON=py -3.12     (Windows)
#   VENV=...           virtualenv directory, default .venv
#   SCENARIO=...       demo scenario, default field_service
#   PYTEST_ARGS=...    extra pytest flags
#   DEMO_ARGS=...      extra replay_demo.py flags, e.g. DEMO_ARGS=--no-llm
#   MOCK_EVENT_LOG=... output path for the mock telemetry log

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
# VENV_BIN and VENV_PY are recursively expanded on purpose: they are read
# before VENV is set below, so the value must be resolved at use time.
ifeq ($(OS),Windows_NT)
  PLATFORM    := windows
  PYTHON      ?= python
  ECHO_BLANK  := echo.
  VENV_BIN     = $(VENV)/Scripts
  VENV_PY      = $(VENV_BIN)/python.exe
else
  PLATFORM    := posix
  PYTHON      ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
  ECHO_BLANK  := echo
  VENV_BIN     = $(VENV)/bin
  VENV_PY      = $(VENV_BIN)/python
endif

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VENV           ?= .venv
SCENARIO       ?= field_service
PYTEST_ARGS    ?= -v
DEMO_ARGS      ?=
MOCK_EVENT_LOG ?= logs/mock_run.jsonl
PIPFLAGS       ?= --disable-pip-version-check

# Dashboard stylesheet build (only `make css` needs Node)
NODE          ?= node
NPM           ?= npm
DASHBOARD_DIR ?= telemetry/dashboard

VENV_STAMP := $(VENV)/pyvenv.cfg
DEPS_STAMP := $(VENV)/.prism-deps-installed
DEV_STAMP  := $(VENV)/.prism-dev-installed

# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
# Printed through $(info) rather than echo: make runs echo lines that contain
# no shell metacharacters without a shell and collapses their whitespace
# there, which would destroy the column alignment below.
define HELP_TEXT
PRISM - Streaming Live RAG
platform: $(PLATFORM)   python: $(PYTHON)   venv: $(VENV)
variables: PYTHON VENV SCENARIO PYTEST_ARGS DEMO_ARGS MOCK_EVENT_LOG

  setup         Create venv, install dependencies, seed .env
  venv          Create the virtualenv only
  install       Install pinned dependencies into the virtualenv
  dev           Install development tools, currently ruff
  doctor        Report interpreter, venv and dependency status
  run           Start the FastAPI engine on HOST:PORT from .env
  test          Run the pytest suite
  gates         Run the automated G2-G6 gate evaluator
  gates-offline Same gates scored against logs/mock_run.jsonl
  lint          ruff check
  format        ruff check --fix, then ruff format
  format-check  ruff format --check, fails when files need reformatting
  check         lint, format-check, test, gates
  demo          Replay one scenario, default field_service
  demo-all      Replay all three demo scenarios
  mock-events   Regenerate the mock telemetry log
  css           Recompile the dashboard stylesheet (needs Node)
  css-check     Recompile and fail when the committed stylesheet is stale
  env           Copy .env.template to .env when missing
  docker-build  Build the container image
  docker-up     docker compose up --build
  docker-down   docker compose down
  docker-logs   Follow container logs
  clean         Remove caches
  clean-logs    Remove generated session and run logs
  distclean     clean, then delete the virtualenv
endef

.PHONY: help
help:
	@$(info $(HELP_TEXT))
	@$(ECHO_BLANK)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
.PHONY: check-python
check-python:
	@$(PYTHON) -c "import sys; sys.exit('ERROR: PRISM requires Python 3.10 or newer, found {}.{}. Pass PYTHON=... to pick another interpreter.'.format(*sys.version_info[:2])) if sys.version_info < (3, 10) else None"
	@$(PYTHON) -c "import sys; print('[warn] Python {}.{} detected. requirements.txt pins numpy==1.26.4, which ships wheels for 3.10-3.12 only. Pass PYTHON=... if the install fails.'.format(*sys.version_info[:2])) if sys.version_info >= (3, 13) else None"

$(VENV_STAMP): | check-python
	@echo Creating virtualenv in $(VENV) with $(PYTHON)
	$(PYTHON) -m venv $(VENV)

.PHONY: venv
venv: $(VENV_STAMP)

$(DEPS_STAMP): requirements.txt | $(VENV_STAMP)
	$(VENV_PY) -m pip install $(PIPFLAGS) --upgrade pip
	$(VENV_PY) -m pip install $(PIPFLAGS) -r requirements.txt
	@$(VENV_PY) -c "import pathlib; pathlib.Path('$(DEPS_STAMP)').touch()"

.PHONY: install
install: check-python $(DEPS_STAMP)

$(DEV_STAMP): requirements-dev.txt | $(DEPS_STAMP)
	$(VENV_PY) -m pip install $(PIPFLAGS) -r requirements-dev.txt
	@$(VENV_PY) -c "import pathlib; pathlib.Path('$(DEV_STAMP)').touch()"

.PHONY: dev
dev: check-python $(DEV_STAMP)

.PHONY: env
env:
	@$(PYTHON) -c "import pathlib, shutil; d = pathlib.Path('.env'); t = pathlib.Path('.env.template'); print('.env already exists, left untouched') if d.exists() else (shutil.copyfile(t, d), print('created .env from .env.template'))"

.PHONY: setup
setup: env venv install dev doctor

# ---------------------------------------------------------------------------
# Lint and format (configuration lives in pyproject.toml)
# ---------------------------------------------------------------------------
.PHONY: lint format format-check
lint: check-python $(DEV_STAMP)
	$(VENV_PY) -m ruff check .

format: check-python $(DEV_STAMP)
	$(VENV_PY) -m ruff check --fix .
	$(VENV_PY) -m ruff format .

format-check: check-python $(DEV_STAMP)
	$(VENV_PY) -m ruff format --check .

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
.PHONY: doctor
doctor: check-python
	@$(PYTHON) -c "import sys; print('system python : ' + sys.version.split()[0] + ' at ' + sys.executable)"
	@$(PYTHON) -c "import pathlib; p = pathlib.Path('$(VENV_PY)'); print('virtualenv    : ' + ('present at ' + str(p) if p.exists() else 'missing, run make setup'))"
	@$(PYTHON) -c "import pathlib; print('.env          : ' + ('present' if pathlib.Path('.env').exists() else 'missing, run make env'))"
	@$(PYTHON) -c "import pathlib, subprocess, sys; p = pathlib.Path('$(VENV_PY)'); print('dependencies  : virtualenv missing, run make setup') if not p.exists() else sys.exit(subprocess.call([str(p), '-c', 'import fastapi, uvicorn, numpy, rank_bm25, sentence_transformers, pytest']))"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
.PHONY: run serve
run: check-python $(DEPS_STAMP)
	$(VENV_PY) -m controller.main

serve: run

# ---------------------------------------------------------------------------
# Tests and evaluation
# ---------------------------------------------------------------------------
.PHONY: test
test: check-python $(DEPS_STAMP)
	$(VENV_PY) -m pytest tests/ $(PYTEST_ARGS)

.PHONY: gates gates-offline
gates: check-python $(DEPS_STAMP)
	$(VENV_PY) -m evaluation.gates

gates-offline: check-python $(DEPS_STAMP)
	$(VENV_PY) -m evaluation.gates --mode offline

.PHONY: check
check: lint format-check test gates

# ---------------------------------------------------------------------------
# Demos and utilities
# ---------------------------------------------------------------------------
.PHONY: demo
demo: check-python $(DEPS_STAMP)
	$(VENV_PY) scripts/replay_demo.py --scenario $(SCENARIO) $(DEMO_ARGS)

.PHONY: demo-all
demo-all: SCENARIO := all
demo-all: demo

.PHONY: mock-events
mock-events: check-python
	@$(PYTHON) scripts/generate_mock_events.py $(MOCK_EVENT_LOG)

# ---------------------------------------------------------------------------
# Dashboard stylesheet
# ---------------------------------------------------------------------------
# styles.css is compiled from tailwind.config.js + tailwind.input.css and is
# committed, so neither `make run` nor the Docker image needs Node. Rerun this
# target after touching index.html or the config, and commit the result; CI
# fails when the committed file is stale. Needs Node only for this target.
.PHONY: css css-check
css: check-node
	$(NPM) --prefix $(DASHBOARD_DIR) install --no-audit --no-fund
	$(NPM) --prefix $(DASHBOARD_DIR) run build

css-check: check-node
	$(NPM) --prefix $(DASHBOARD_DIR) install --no-audit --no-fund
	$(NPM) --prefix $(DASHBOARD_DIR) run build
	git diff --exit-code -- $(DASHBOARD_DIR)/styles.css

.PHONY: check-node
check-node:
	@$(NODE) -e "console.log('node ' + process.version)"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
.PHONY: docker-build docker-up docker-down docker-logs
docker-build:
	docker compose build

docker-up: env
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
.PHONY: clean clean-logs distclean
clean:
	@$(PYTHON) -c "import pathlib, shutil; caches = [p for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; [shutil.rmtree(p, ignore_errors=True) for p in caches]; shutil.rmtree('.pytest_cache', ignore_errors=True); print('removed __pycache__ and .pytest_cache')"

clean-logs:
	@$(PYTHON) -c "import pathlib; stale = [p for p in pathlib.Path('logs').glob('session_*.jsonl')] + [p for p in pathlib.Path('logs').glob('run_*.jsonl')]; [p.unlink() for p in stale]; print('removed {} generated log file(s)'.format(len(stale)))"

distclean: clean
	@$(PYTHON) -c "import shutil; shutil.rmtree('$(VENV)', ignore_errors=True); print('removed $(VENV)')"
	@echo .env kept on purpose
