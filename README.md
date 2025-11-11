# Replicate Runner

**Streamline your AI model workflows with intelligent automation for Replicate and HuggingFace.**

A production-ready CLI tool that eliminates friction in FLUX+LoRA development pipelines through smart type inference, automated sample curation, and seamless HuggingFace publishing.

## Value Proposition

### Smart Parameter Management
Skip JSON formatting entirely. Natural CLI syntax with automatic type inference:
```bash
--param steps:20 --param scale:0.8 --param enable:true
```
Automatically converts to `int(20)`, `float(0.8)`, `bool(True)`.

### Intelligent LoRA Publishing
Publish ai-toolkit LoRAs to HuggingFace with zero manual curation:
- Analyzes 40+ training samples per prompt
- Selects best final sample based on timestamp and training step
- Auto-generates model cards with metadata and galleries
- Extracts trigger words from config automatically

### Optimized FLUX+LoRA Workflow
Complete the cycle faster:
```
train LoRA -> publish to HF -> test with FLUX
```
Streamlined upload step with automatic metadata extraction and gallery generation.

## Quick Start

### Installation
```bash
# Clone and install
git clone https://github.com/AeyeOps/replicate-runner
cd replicate-runner
uv pip install -e .

# Or run directly without installing
uv run replicate-runner --help
```

### Configuration
Create `.env` in project root:
```bash
REPLICATE_API_TOKEN=r8_your_token_here
HF_TOKEN=hf_your_token_here
```

## Usage Examples

### Run FLUX with LoRA
```bash
replicate-runner replicate run-model \
  lucataco/flux-dev-lora \
  <version-id> \
  --param prompt:"a photo of TOK, person, portrait" \
  --param hf_lora:"youruser/your-lora" \
  --param lora_scale:0.8 \
  --param num_outputs:1 \
  --param num_inference_steps:28
```

### Publish LoRA to HuggingFace
```bash
replicate-runner hf publish-hf-lora \
  --name my-lora-v1 \
  --source-dir /path/to/ai-toolkit/output/my_lora_v1.0 \
  --private true
```

**What it does:**
1. Finds main `.safetensors` file (excludes checkpoints)
2. Scans `samples/` directory for training outputs
3. Groups samples by prompt index
4. Selects highest training step per prompt
5. Extracts trigger word from `config.yaml`
6. Generates HuggingFace model card with gallery
7. Uploads model + curated samples + README

**Sample Selection Logic:**
```
Training outputs:
  1737531308253__000000000_0.jpg  (step 0, prompt 0)
  1737531349391__000000000_1.jpg  (step 0, prompt 1)
  1737531390527__000000250_0.jpg  (step 250, prompt 0) <- SELECTED
  1737531431663__000000250_1.jpg  (step 250, prompt 1) <- SELECTED

Result: 1 model file + 2 best sample images
```

### Type Inference
Automatic conversion from CLI strings to Python types:
```bash
--param steps:20           # -> int(20)
--param scale:0.8          # -> float(0.8)
--param enable:true        # -> bool(True)
--param prompt:"hello"     # -> str("hello")
--param sizes:[512,768]    # -> list([512, 768])
```

## Use Cases

**Production Workflows:**
- **LoRA Training Iteration**: Batch upload 5+ LoRA variations with consistent metadata
- **CI/CD Integration**: Automated LoRA publishing pipelines
- **Parameter Exploration**: Rapid testing of LoRA scales and inference steps
- **Team Standardization**: Consistent publishing format and sample selection

**Time Savings:**
- LoRA publishing: ~5 minutes per upload (sample selection, model card generation)
- Model testing: ~30 seconds per run (no JSON formatting required)

**When to Use Web UI Instead:**
- One-off image generation
- Visual exploration of new models
- Side-by-side output comparison

## Installation Options

### Quick Start (uv - recommended)
```bash
uv pip install -e .
# Or: uv run replicate-runner --help
```

### Standard Python
```bash
pip install -e .
```

### Standalone Executable
```bash
make standalone
# Creates: dist/replicate-runner.exe (~179MB)
```

## Technical Features

### Security
- Safe type parsing with `ast.literal_eval()` (no code execution)
- Environment-based secret management
- Input validation on all parameters

### Performance
- Async rotating log handler (10MB max, 5 backups, gzip compression)
- Single-pass parameter parsing (O(n))
- Minimal API calls

### Code Quality
- Full type hints
- Comprehensive docstrings
- Production-grade error handling
- Rich console output for superior UX

## Requirements

- Python 3.11+
- Replicate API token
- HuggingFace token (for publishing)

## Dependencies

Core:
- `typer>=0.9.0` - CLI framework
- `rich>=13.0.0` - Terminal output
- `python-dotenv>=1.0.0` - Environment variables
- `pyyaml>=6.0.0` - Config parsing
- `replicate>=0.25.0` - Replicate API
- `huggingface-hub>=0.20.0` - HuggingFace API

## Development

### Build Commands
```bash
make clean      # Remove build artifacts
make sync       # Sync dependencies
make build      # Build package
make install    # Install in dev mode
make standalone # Create executable
make all        # clean + sync + build
```

### Project Structure
```
replicate_runner/
├── main.py              # CLI entry point, Typer app
├── config_loader.py     # Multi-source config (env + YAML)
├── logger_config.py     # Async rotating file logger
├── commands/
│   ├── replicate_cmds.py  # Replicate model execution
│   └── hf_cmds.py         # HuggingFace publishing
└── config/
    └── model_configs.yaml # Optional model configs
```

## Documentation

- `CLAUDE.md` - Development guide for AI assistants
- `FLUX_LORA_EXAMPLE.md` - Complete FLUX+LoRA workflow guide
- `replicate_runner/CLAUDE.md` - Package architecture details

## Contributing

Contributions welcome. Priority areas:
- Unit test coverage
- Model-specific parameter helpers
- Batch processing commands
- Progress tracking for long-running models

Standard process:
1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

## FAQ

**Q: Why not use Replicate's Python SDK directly?**
A: The SDK is more flexible for custom scripts. This tool adds automation and convenience for repetitive workflows.

**Q: When should I use this vs the web UI?**
A: Web UI for exploration and visual work. This tool for repetitive workflows, automation, and pipelines.

**Q: Is this production-ready?**
A: For personal/small team use, yes. For enterprise scale, add comprehensive test suite, CI/CD pipeline, version pinning, and error telemetry.

## Built With

- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting

## Acknowledgments

Special thanks to [ai-toolkit](https://github.com/ostris/ai-toolkit) by [@ostris](https://github.com/ostris) - the LoRA training framework that this tool's publishing workflow is designed around. The intelligent sample selection and automated publishing features were built specifically to streamline ai-toolkit's output format.

## License

MIT
