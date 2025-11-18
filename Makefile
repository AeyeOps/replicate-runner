
.PHONY: clean sync build install standalone standalone-env help all

.DEFAULT_GOAL := help

PYTHON ?= python3
STANDALONE_VENV := .venv-standalone
STANDALONE_BIN := $(STANDALONE_VENV)/bin
STANDALONE_PIP := $(STANDALONE_BIN)/pip
STANDALONE_PYINSTALLER := $(STANDALONE_BIN)/pyinstaller
STANDALONE_ARTIFACT := dist/replicate-runner
STANDALONE_DEPLOY := /opt/bin/replicate-runner

help:
	@echo "Available targets:"
	@echo "  clean       - Remove build artifacts, caches, and logs"
	@echo "  sync        - Sync dependencies using uv"
	@echo "  build       - Build the package using uv"
	@echo "  install     - Install in development mode using uv"
	@echo "  standalone  - Create standalone executable with PyInstaller"
	@echo "  all         - Sync and build"
	@echo "  help        - Show this menu (default)"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf replicate_runner.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f replicate-runner.exe
	@echo "Clean complete."

sync:
	@echo "Syncing dependencies with uv..."
	uv sync
	@echo "Sync complete."

build:
	@echo "Building package with uv..."
	uv build
	@echo "Build complete."

install: sync
	@echo "Installing package in development mode..."
	uv pip install -e .
	@echo "Installation complete."

standalone-env:
	@if [ ! -d "$(STANDALONE_VENV)" ]; then \
		echo "Creating isolated virtualenv at $(STANDALONE_VENV)..."; \
		$(PYTHON) -m venv $(STANDALONE_VENV); \
	fi
	@echo "Syncing standalone build dependencies..."
	@$(STANDALONE_PIP) install --upgrade pip wheel setuptools >/dev/null
	@$(STANDALONE_PIP) install --upgrade pyinstaller >/dev/null
	@$(STANDALONE_PIP) install --upgrade -e . >/dev/null

standalone: clean standalone-env
	@echo "Building standalone executable with PyInstaller..."
	@$(STANDALONE_PYINSTALLER) --clean replicate-runner.spec
	@if [ -f "$(STANDALONE_ARTIFACT)" ]; then \
		echo "Deploying standalone binary to $(STANDALONE_DEPLOY)..."; \
		cp -f $(STANDALONE_ARTIFACT) $(STANDALONE_DEPLOY); \
		echo "Standalone executable copied to $(STANDALONE_DEPLOY)"; \
	else \
		echo "[WARN] Standalone binary $(STANDALONE_ARTIFACT) not found"; \
	fi
