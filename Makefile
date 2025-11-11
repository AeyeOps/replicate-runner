.PHONY: clean sync build install standalone help all

# Default target
all: sync build

help:
	@echo "Available targets:"
	@echo "  clean       - Remove build artifacts, caches, and logs"
	@echo "  sync        - Sync dependencies using uv"
	@echo "  build       - Build the package using uv"
	@echo "  install     - Install in development mode using uv"
	@echo "  standalone  - Create standalone executable with PyInstaller"
	@echo "  all         - Sync and build (default)"

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

standalone: clean
	@echo "Creating standalone executable with PyInstaller..."
	uv run pyinstaller replicate-runner.spec
	@echo "Standalone executable created: dist/replicate-runner.exe"
