# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Replicate Runner is a CLI tool for running Replicate AI models and publishing LoRAs to HuggingFace Hub. It provides intelligent automation for FLUX+LoRA workflows including smart parameter type inference, automated sample curation, and seamless HuggingFace publishing.

## Package Management

This project uses **uv** for Python package management (Python 3.11+).

### Essential Commands
```bash
# Development setup
uv pip install -e .          # Install in editable mode
uv run replicate-runner      # Run without activating venv

# Build commands (see Makefile)
make clean                   # Remove build artifacts
make sync                    # Sync dependencies with uv
make build                   # Build package
make install                 # Install in dev mode
make standalone              # Create PyInstaller executable

# Code quality
uv run black replicate_runner/
uv run ruff check --fix replicate_runner/
uv run mypy replicate_runner/
```

## Architecture

### CLI Structure
Hierarchical Typer-based CLI with two main command namespaces:

```
replicate-runner
├── replicate          # Replicate API commands (replicate_cmds.py)
│   ├── run-model      # Execute Replicate models with type-inferred params
│   └── init-run-file  # Generate batch run templates from LoRA catalogs
└── hf                 # HuggingFace Hub commands (hf_cmds.py)
    ├── publish-hf-lora    # Intelligent LoRA publishing with sample curation
    ├── list-models        # List HF models with privacy status
    └── loras              # LoRA catalog management (sub-app)
        ├── list           # List available LoRA collections
        ├── show           # Display collection details
        └── view-images    # Open reference images in viewer
```

### Configuration System (config_loader.py:6-64)
Multi-source configuration with precedence hierarchy:

1. **Environment Variables**: `.env` files (searches CWD, parent, grandparent)
2. **YAML Configs**: Merges all `config/*.yaml` files
3. **Precedence**: Environment overrides YAML via `get()` method

Required environment variables:
- `REPLICATE_API_TOKEN` - Replicate API access
- `HF_TOKEN` or `HF_API_TOKEN` - HuggingFace Hub authentication
- `LORA_IMAGE_VIEWER` - (Optional) Path to image viewer for `hf loras view-images`

**Auto-injection**: When input params reference `huggingface.co` in `extra_lora`, `hf_lora`, or `lora_weights`, HF token is automatically injected as `hf_api_token`.

### LoRA Catalog System (lora_catalog.py)
Centralized LoRA configuration in `replicate_runner/config/loras.yaml`:

**Structure**:
```yaml
loras:                          # Individual LoRA definitions
  andie_v1_1:
    name: "Andie v1.1"
    hf_repo: "steveant/andie-lora-v1.1"
    lora_weights: "huggingface.co/steveant/andie-lora-v1.1"
    trigger: "andie"
    default_prompt: "{trigger}, woman, portrait..."
    base_images: ["path/to/reference.jpg"]

lora_collections:               # Logical groupings
  studio_muses:
    description: "Primary portrait LoRAs"
    default_prompt: "{trigger}, woman, portrait..."
    loras: [andie_v1_1, demi_v1_3]
```

**Path Resolution**: `base_images` paths are resolved relative to project root or expanded from home directory.

### Parameter Type Inference (replicate_cmds.py:41-93)
Automatic type conversion from CLI strings using `ast.literal_eval()`:

```bash
--param steps:20           # int(20)
--param scale:0.8          # float(0.8)
--param enable:true        # bool(True)
--param prompt:"hello"     # str("hello")
--param sizes:[512,768]    # list([512, 768])
```

Boolean handling: `true/false` (case-insensitive) → Python `True/False`

### Model Reference Normalization (replicate_cmds.py:96-119)
Flexible version pinning with conflict detection:

```bash
# Three equivalent ways to specify version:
replicate-runner replicate run-model owner/model <version>
replicate-runner replicate run-model owner/model:<version>
replicate-runner replicate run-model owner/model/version  # URL format
```

**Conflict check**: Errors if version specified both as argument and embedded in model string.

### HuggingFace LoRA Publishing (hf_cmds.py:284-426)
Intelligent publishing workflow for ai-toolkit LoRA outputs:

**Sample Selection Logic** (hf_cmds.py:340-366):
1. Find main `.safetensors` file (excludes files with "index" in name)
2. Scan `samples/` for pattern: `{timestamp}__{step}_{index}.{ext}`
3. Group images by prompt index (last number)
4. Select highest step/timestamp per prompt group
5. Upload curated samples to `samples/` subdirectory

**Trigger Word Extraction** (hf_cmds.py:155-188):
- Reads `config.yaml` from source directory
- Extracts from `config.process[0].trigger_word`
- Falls back to "TOK" if not found

**Model Card Generation** (hf_cmds.py:190-282):
- YAML frontmatter with widget-based gallery
- Auto-generates diffusers usage examples
- Includes licensing and training metadata

### Output Persistence (replicate_cmds.py:133-164)
Automatic download of `FileOutput` results:

```
output/
└── {model}_{timestamp}/
    ├── output_0.webp
    ├── output_1.webp
    └── ...
```

Model name sanitization: `/` → `_`, `:` → `__`

### Batch Run Templates (replicate_cmds.py:269-313)
Generate YAML templates for systematic LoRA testing:

```bash
replicate-runner replicate init-run-file runs/test.yaml \
  --collection studio_muses \
  --lora andie_v1_1 \
  --overwrite
```

**Generated structure**:
- `model`, `version`: Replicate model reference
- `run_settings`: Loop count and images per loop
- `defaults.params`: Base parameters for all runs
- `loras`: Array of LoRA metadata
- `predictions`: Array of per-LoRA run configurations

**Default parameters** (replicate_cmds.py:21-38):
- Model: `lucataco/flux-dev-lora`
- Steps: 35, guidance: 2.3, lora_scale: 0.9
- Format: webp quality 100
- Aspect ratio: 3:4

## Key Implementation Patterns

### Adding New Commands
```python
# commands/new_cmds.py
import typer
from rich.console import Console
from replicate_runner.config_loader import ConfigLoader

console = Console()
app = typer.Typer()

@app.command()
def my_command(
    param: str = typer.Option(..., "--param", help="Description"),
):
    """Command docstring."""
    config = ConfigLoader()
    token = config.get("API_TOKEN", "")
    console.print("[green]Success![/green]")
```

Register in `main.py`:
```python
from replicate_runner.commands import new_cmds
app.add_typer(new_cmds.app, name="namespace")
```

### Configuration Access
```python
from replicate_runner.config_loader import ConfigLoader

config = ConfigLoader()
value = config.get("KEY", "default")  # Env overrides YAML
```

### Rich Console Output
Use semantic color coding:
- `[blue]` - Informational messages
- `[green]` - Success messages
- `[yellow]` - Warnings
- `[red]` - Errors

### LoRA Catalog Operations
```python
from replicate_runner.lora_catalog import (
    load_lora_catalog,
    resolve_collection,
    gather_base_images,
)

catalog = load_lora_catalog()
collection_meta, entries = resolve_collection(catalog, "studio_muses")
images = gather_base_images(entries)
```

## File Structure
```
replicate_runner/
├── main.py                    # CLI entry point, command registration
├── config_loader.py           # Multi-source config (env + YAML)
├── logger_config.py           # Async rotating file logger
├── lora_catalog.py            # LoRA catalog loading and resolution
├── commands/
│   ├── replicate_cmds.py      # Replicate model execution, batch templates
│   └── hf_cmds.py             # HuggingFace publishing, LoRA catalog CLI
└── config/
    ├── model_configs.yaml     # Optional model-specific configs
    └── loras.yaml             # LoRA catalog and collections

# Project root
├── t.py                       # Example batch script (iterative generation)
├── scripts/
│   └── run_flux_example.sh    # Quick-start FLUX+LoRA runner
├── Makefile                   # Build automation
└── replicate-runner.spec      # PyInstaller configuration
```

## Common Workflows

### Testing a LoRA Collection
```bash
# 1. View configured LoRAs
replicate-runner hf loras list
replicate-runner hf loras show studio_muses

# 2. Generate batch run template
replicate-runner replicate init-run-file test-run.yaml \
  --collection studio_muses

# 3. Quick test single LoRA
./scripts/run_flux_example.sh  # Uses env vars for customization
```

### Publishing ai-toolkit Output
```bash
# Typical ai-toolkit directory:
# output/my_lora/
# ├── my_lora.safetensors
# ├── samples/
# │   ├── 1737531308253__000000000_0.jpg  (step 0, prompt 0)
# │   └── 1737531390527__000000250_0.jpg  (step 250, prompt 0) <- SELECTED
# └── config.yaml

replicate-runner hf publish-hf-lora \
  --name my-lora-v1 \
  --source-dir /path/to/ai-toolkit/output/my_lora \
  --private true
```

Trigger word auto-extracted from `config.yaml` or defaults to "TOK".

### Running Replicate Models
```bash
# Latest version with auto type inference
replicate-runner replicate run-model lucataco/flux-dev-lora \
  --param prompt:"a photo of andie, woman, portrait" \
  --param hf_lora:"steveant/andie-lora-v1.1" \
  --param lora_scale:0.8 \
  --param num_outputs:4 \
  --param num_inference_steps:28

# Pin specific version (three equivalent methods)
replicate-runner replicate run-model lucataco/flux-dev-lora <version-id>
replicate-runner replicate run-model lucataco/flux-dev-lora:<version-id>
replicate-runner replicate run-model lucataco/flux-dev-lora/version/<version-id>
```

Outputs automatically saved to `output/{model}_{timestamp}/`

## Design Patterns

### Fail-Fast Philosophy
Per user instructions: no backward compatibility, no defensive try-catch for legacy features. Clear errors preferred over silent fallbacks.

### Type Safety
- Full type hints using Python 3.11+ syntax
- Safe parsing via `ast.literal_eval()` (no `eval()`)
- Explicit type conversions with clear error messages

### Sample Selection Algorithm
For ai-toolkit samples matching `{timestamp}__{step}_{index}.{ext}`:
1. Regex extract: `(\d+)__(\d+)_(\d+)\.`
2. Group by index (prompt number)
3. Sort by `(step, timestamp)` tuple
4. Select `max()` from each group

This ensures latest training iteration per prompt without manual curation.

## Dependencies

**Core**:
- `typer` - CLI framework with type hints
- `rich` - Terminal output and formatting
- `python-dotenv` - Environment variable loading
- `pyyaml` - YAML config parsing
- `replicate` - Replicate API client
- `huggingface-hub` - HF Hub uploads

**Dev** (pyproject.toml tool.uv.dev-dependencies):
- `black`, `ruff`, `mypy` - Code quality
- `pytest` - Testing framework
