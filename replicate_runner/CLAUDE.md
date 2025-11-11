# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

<overview>
Replicate Runner is a CLI tool for running Replicate AI models and publishing private LoRAs to HuggingFace Hub. The tool provides a typer-based CLI interface with rich console output, configuration management, and async logging.
</overview>

## Package Management: uv

<uv-workflow>
This project uses **uv** for fast, modern Python package management with Python 3.12+.

### Key Commands
- `uv venv` - Create virtual environment (.venv)
- `uv pip install -e .` - Install project in editable mode
- `uv pip install -e ".[dev]"` - Install with dev dependencies
- `uv run replicate-runner` - Run CLI without activating venv
- `uv pip compile pyproject.toml` - Generate lockfile

### Development Setup
```bash
# Create venv and install
uv venv
uv pip install -e .

# Or run directly
uv run replicate-runner --help
```

### Code Quality Tools (dev dependencies)
- **Black**: Code formatting (`black replicate_runner/`)
- **Ruff**: Fast linting (`ruff check replicate_runner/`)
- **Mypy**: Type checking (`mypy replicate_runner/`)
- **Pytest**: Testing (`pytest tests/`)

Run before committing:
```bash
uv run black replicate_runner/
uv run ruff check --fix replicate_runner/
uv run mypy replicate_runner/
```
</uv-workflow>

## Architecture

<architecture>
### CLI Structure
The application uses a hierarchical typer CLI structure with two main command groups:

**main.py**: Entry point with `run()` function, initializes logger, adds subcommands
- **commands/replicate_cmds.py**: Replicate cloud API integration (`replicate` namespace)
- **commands/hf_cmds.py**: HuggingFace Hub integration (`hf` namespace)

Command execution examples:
```bash
replicate-runner replicate run-model owner/model version --param key:value
replicate-runner hf publish-hf-lora --name my-lora --source-dir ./output
replicate-runner hf list-models --limit 20
```

### Configuration System
**ConfigLoader** (config_loader.py:6-51) implements a multi-source configuration strategy:
1. **Environment Variables**: Loads from `.env` files (searches CWD, parent, grandparent)
2. **YAML Configs**: Merges all `*.yaml` files from `config/` directory
3. **Precedence**: Environment variables override YAML configs via `get()` method

Required environment variables:
- `REPLICATE_API_TOKEN` - For Replicate cloud API
- `HF_TOKEN` - For HuggingFace Hub uploads

Key locations searched for `.env`:
- Current working directory
- One level up
- Two levels up

### Logging System
**Asynchronous rotating file logger** (logger_config.py:20-69):
- Log file: `replicate_runner.log` (10MB max size, 5 backups)
- Queue-based async logging via `QueueHandler` + `QueueListener`
- Automatic compression: Rotated logs compressed to `.gz` format
- Custom `doRollover` override compresses backup files after rotation

### Replicate Integration
**run_replicate_model()** (commands/replicate_cmds.py:7-24):
- Authenticates using `REPLICATE_API_TOKEN` from config
- Accepts model name, version ID, and input parameters
- Returns prediction results from Replicate API

### HuggingFace Integration
**publish_lora()** (commands/hf_cmds.py:12-129):
- Uploads LoRAs to HuggingFace Hub as private repos (default)
- Automatically finds .safetensors file (excludes index files)
- Discovers samples directory and intelligently selects latest samples per prompt
- Sample pattern: `timestamp__step_index.ext` (e.g., `1737531308253__000000000_0.jpg`)
- Groups by prompt index, selects highest step/timestamp for each prompt
- Creates repo and uploads: 1 model file + N sample images

**list_models()** (commands/hf_cmds.py:132-172):
- Lists your HuggingFace models with privacy status and download counts
- Auto-detects username from HF token if not specified
</architecture>

## Development Workflow

<development>
### Environment Setup
1. Create `.env` file in project root with tokens:
```bash
REPLICATE_API_TOKEN=your-replicate-token
HF_TOKEN=your-huggingface-token
```

2. Install project:
```bash
uv venv
uv pip install -e .
```

3. (Optional) Configure model-specific settings in `config/model_configs.yaml`

### Running the CLI
```bash
# Using uv run (recommended - no venv activation needed)
uv run replicate-runner --help
uv run replicate-runner hf publish-hf-lora --name my-lora --source-dir ./output

# Or activate venv manually
source .venv/Scripts/activate  # Windows Git Bash
replicate-runner --help
```

### Publishing a LoRA to HuggingFace
```bash
# Typical ai-toolkit output directory structure:
# output/my_lora_v1.0/
# ├── my_lora_v1.0.safetensors
# ├── my_lora_v1.0_000001000.safetensors  (checkpoints - ignored)
# └── samples/
#     ├── 1737531308253__000000000_0.jpg
#     ├── 1737531349391__000000000_1.jpg
#     └── 1737531390527__000000250_0.jpg  (later step - selected)

# Publish with intelligent sample selection
uv run replicate-runner hf publish-hf-lora \
  --name my-lora-v1 \
  --source-dir //wsl.localhost/ubuntu-22.04/opt/ai-toolkit/output/my_lora_v1.0 \
  --private true

# Creates: https://huggingface.co/yourusername/my-lora-v1
# Uploads: my_lora_v1.0.safetensors + samples/1737531390527__000000250_0.jpg (latest for prompt 0)
```

### Code Quality Workflow
```bash
# Format code
uv run black replicate_runner/

# Lint and auto-fix
uv run ruff check --fix replicate_runner/

# Type check
uv run mypy replicate_runner/

# Run tests (when added)
uv run pytest tests/
```
</development>

## Important Patterns

<patterns>
### Adding New Commands
1. Create command function in appropriate `commands/*.py` module
2. Use `@app.command()` or `@app.command(name="custom-name")` decorator
3. Access config via `ConfigLoader().get("KEY", default)`
4. Use `console.print()` with rich markup for user feedback
5. Add command group to main.py via `app.add_typer(module.app, name="namespace")`

Example:
```python
# commands/new_cmds.py
import typer
from rich.console import Console

console = Console()
app = typer.Typer()

@app.command()
def my_command(
    param: str = typer.Option(..., "--param", help="Description"),
):
    """Command description."""
    from replicate_runner.config_loader import ConfigLoader

    config = ConfigLoader()
    token = config.get("API_TOKEN", "")
    console.print(f"[green]Success![/green]")
```

```python
# main.py
from replicate_runner.commands import replicate_cmds, hf_cmds, new_cmds

app.add_typer(new_cmds.app, name="new")
```

### Configuration Access
```python
from replicate_runner.config_loader import ConfigLoader

config_loader = ConfigLoader()
value = config_loader.get("CONFIG_KEY", "default_value")
```

### Logging Usage
Logger is initialized globally in `main_callback()`. Access via:
```python
import logging
logger = logging.getLogger("replicate_runner")
logger.info("Message")
```

### HuggingFace Sample Discovery Logic
The `publish-hf-lora` command intelligently handles ai-toolkit output:
- Finds .safetensors file without "index" in filename (main model, not checkpoint)
- Scans samples/ directory for pattern: `timestamp__step_index.ext`
- Groups images by prompt index (last number)
- Selects latest step for each unique prompt
- Uploads to `samples/` subdirectory in HuggingFace repo
</patterns>

## File Structure

<structure>
```
replicate_runner/
├── main.py                 # CLI entry point, command registration
├── config_loader.py        # Multi-source configuration loader
├── logger_config.py        # Async rotating file logger setup
├── commands/
│   ├── __init__.py
│   ├── replicate_cmds.py   # Replicate API command implementations
│   └── hf_cmds.py          # HuggingFace Hub publishing commands
└── config/
    └── model_configs.yaml  # Model-specific YAML configurations (optional)

# Root directory
├── pyproject.toml          # Project metadata, dependencies, uv config
├── .env                    # Tokens and secrets (gitignored)
├── .gitignore              # Excludes .venv/, *.log, *.pyc, etc.
└── .venv/                  # Virtual environment (created by uv venv)
```
</structure>

## Dependencies

<dependencies>
**Core dependencies:**
- `typer` - CLI framework with type hints
- `rich` - Rich terminal output and formatting
- `python-dotenv` - Environment variable loading
- `pyyaml` - YAML configuration parsing
- `replicate` - Replicate cloud API client
- `huggingface-hub` - HuggingFace Hub uploads and repo management

**Dev dependencies:**
- `black` - Code formatter
- `ruff` - Fast Python linter
- `mypy` - Static type checker
- `pytest` - Testing framework
</dependencies>
