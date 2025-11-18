import ast
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import replicate
from ratelimit import limits, sleep_and_retry
from replicate.helpers import FileOutput
import typer
import yaml
from rich.console import Console

from replicate_runner.lora_catalog import LoraEntry, load_lora_catalog, resolve_collection
from replicate_runner.persona import PERSONA_ACTION_MARKER, PersonaActionResolver

console = Console()

DEFAULT_MODEL_REF = "lucataco/flux-dev-lora"
DEFAULT_PROMPT_TEMPLATE = (
    "{trigger}, a woman, portrait, dramatic studio lighting, photorealistic, 85mm lens"
)
DEFAULT_RUN_SETTINGS = {
    "loops": 10,
    "images_per_loop": 4,
}
DEFAULT_MODEL_PARAMS = {
    "go_fast": False,
    "guidance": 4.0,
    "lora_scale": 1.2,
    "prompt_strength": 0.5,
    "num_outputs": 4,
    "num_inference_steps": 40,
    "aspect_ratio": "9:16",
    "output_format": "webp",
    "output_quality": 90,
    "seed": 0,
    "disable_safety_checker": True,
}

DEFAULT_RATE_CALLS = 4
DEFAULT_RATE_PERIOD = 5

persona_resolver = PersonaActionResolver()


def _load_rate_limits() -> Tuple[int, int]:
    def _env_int(key: str, default: int) -> int:
        try:
            return max(1, int(os.environ.get(key, default)))
        except (TypeError, ValueError):
            return default

    return (
        _env_int("REPLICATE_RATE_CALLS", DEFAULT_RATE_CALLS),
        _env_int("REPLICATE_RATE_PERIOD", DEFAULT_RATE_PERIOD),
    )


RATE_LIMIT_CALLS, RATE_LIMIT_PERIOD = _load_rate_limits()


def parse_param_value(value: str) -> Any:
    """
    Parse a parameter value with automatic type inference.

    Tries to evaluate as Python literal (int, float, bool, list, dict, etc.).
    Falls back to string if evaluation fails.

    Examples:
        "42" -> 42 (int)
        "3.14" -> 3.14 (float)
        "true" -> True (bool)
        "True" -> True (bool)
        "hello" -> "hello" (str)
        "[1,2,3]" -> [1, 2, 3] (list)
    """
    # Handle lowercase boolean strings (common in JSON/YAML)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If literal_eval fails, treat as string
        return value


def parse_params(param_list: List[str]) -> dict:
    """
    Parse list of "key:value" strings into a typed dictionary.

    Args:
        param_list: List of strings in format "key:value"

    Returns:
        Dictionary with parsed keys and type-inferred values

    Raises:
        ValueError: If any param doesn't contain ':'
    """
    params = {}
    for param in param_list:
        if ":" not in param:
            raise ValueError(
                f"Invalid param format: '{param}'. Expected 'key:value'"
            )

        # Split only on first colon (allows colons in values)
        key, value = param.split(":", 1)
        params[key.strip()] = parse_param_value(value.strip())

    return params


def split_model_reference(model: str) -> Tuple[str, Optional[str]]:
    """Normalize model references to ``owner/model`` plus optional version."""

    cleaned = model.strip()

    # Format: owner/model:version
    if ":" in cleaned:
        owner_model, version = cleaned.split(":", 1)
        return owner_model, version or None

    # Format: owner/model/version (common mistake when copying URLs)
    parts = cleaned.split("/")
    if len(parts) > 2:
        owner_model = "/".join(parts[:2])
        version = "/".join(parts[2:]) or None
        return owner_model, version

    return cleaned, None


def format_model_reference(model_name: str, version: Optional[str]) -> str:
    """Return ``owner/model`` or ``owner/model:version`` for Replicate."""

    return f"{model_name}:{version}" if version else model_name


def needs_hf_token(params: dict) -> bool:
    """Return True when inputs reference Hugging Face assets that may need auth."""

    hf_keys = ("extra_lora", "hf_lora", "lora_weights")
    for key in hf_keys:
        value = params.get(key)
        if isinstance(value, str) and "huggingface.co" in value:
            return True
    return False


def collect_file_outputs(result: Any) -> List[FileOutput]:
    """Return all FileOutput objects contained in the model result."""

    if isinstance(result, FileOutput):
        return [result]
    if isinstance(result, list):
        return [item for item in result if isinstance(item, FileOutput)]
    return []


def persist_file_outputs(outputs: List[FileOutput], model_ref: str) -> List[Path]:
    """Download FileOutput objects to disk and return saved paths."""

    if not outputs:
        return []

    safe_model = model_ref.replace("/", "_").replace(":", "__")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    dest_dir = Path("output") / f"{safe_model}_{timestamp}_{unique_suffix}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for idx, file_output in enumerate(outputs):
        url = str(file_output)
        suffix = Path(urlparse(url).path).suffix or ".bin"
        filepath = dest_dir / f"output_{idx}{suffix}"
        with open(filepath, "wb") as fh:
            fh.write(file_output.read())
        saved_paths.append(filepath)

    return saved_paths


@sleep_and_retry
@limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
def _rate_limited_run(client: replicate.Client, model_ref: str, input_params: dict):
    return client.run(model_ref, input=input_params)


def run_replicate_model(
    token: str,
    model_ref: str,
    input_params: dict
):
    """Run a Replicate model reference using the official client."""

    client = replicate.Client(api_token=token)
    console.print(f"Running [bold]{model_ref}[/bold]")
    console.print(f"Parameters: {input_params}")

    return _rate_limited_run(client, model_ref, input_params)


app = typer.Typer(invoke_without_command=True)


@app.callback()
def replicate_root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


def _pick_loras_for_template(
    catalog, collection: Optional[str], explicit_keys: List[str]
) -> Tuple[List[LoraEntry], Optional[str]]:
    selected: List[LoraEntry] = []
    seen: Set[str] = set()
    collection_prompt: Optional[str] = None

    if collection:
        try:
            collection_meta, entries = resolve_collection(catalog, collection)
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
        collection_prompt = collection_meta.default_prompt
        for entry in entries:
            if entry.key not in seen:
                selected.append(entry)
                seen.add(entry.key)

    for key in explicit_keys:
        entry = catalog.loras.get(key)
        if not entry:
            console.print(f"[yellow]Unknown LoRA '{key}' skipped.[/yellow]")
            continue
        if entry.key not in seen:
            selected.append(entry)
            seen.add(entry.key)

    if not selected:
        selected = list(catalog.loras.values())
        if not selected:
            console.print(
                "[red]No LoRAs configured. Populate config/loras.yaml before generating templates.[/red]"
            )
            raise typer.Exit(1)

    return selected, collection_prompt


def _build_run_template(
    loras: List[LoraEntry],
    collection: Optional[str],
    collection_prompt: Optional[str],
) -> Dict[str, Any]:
    prompt_template = collection_prompt or DEFAULT_PROMPT_TEMPLATE
    lora_section: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []

    for entry in loras:
        lora_section.append(
            {
                "key": entry.key,
                "name": entry.name,
                "trigger": entry.trigger,
                "hf_repo": entry.repo_id,
                "lora_weights": entry.lora_weights,
                "default_prompt": entry.default_prompt or prompt_template,
            }
        )
        predictions.append(
            {
                "name": f"{entry.name} sweep",
                "lora": entry.key,
                "prompt": entry.default_prompt or prompt_template,
                "params": {
                    "hf_lora": entry.repo_id,
                    "lora_weights": entry.lora_weights,
                },
            }
        )

    return {
        "model": DEFAULT_MODEL_REF,
        "version": None,
        "collection": collection,
        "prompt_template": prompt_template,
        "run_settings": DEFAULT_RUN_SETTINGS,
        "defaults": {
            "params": DEFAULT_MODEL_PARAMS,
        },
        "loras": lora_section,
        "predictions": predictions,
    }


@app.command(name="init-run-file")
def init_run_file(
    output: Path = typer.Argument(
        Path("replicate-run.yaml"),
        help="Where to write the generated YAML template.",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Name of the LoRA collection to pre-populate with.",
    ),
    lora: Optional[List[str]] = typer.Option(
        None,
        "--lora",
        help="Specific LoRA keys to include (repeat flag for multiples).",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Allow overwriting an existing file.",
    ),
):
    """Generate a multi-run YAML template inspired by t.py's batch script."""

    catalog = load_lora_catalog()
    explicit_keys = lora or []
    selected_loras, collection_prompt = _pick_loras_for_template(
        catalog, collection, explicit_keys
    )
    template = _build_run_template(selected_loras, collection, collection_prompt)

    if output.exists() and not overwrite:
        console.print(
            f"[red]File {output} already exists. Pass --overwrite to replace it.[/red]"
        )
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        yaml.safe_dump(template, fh, sort_keys=False, allow_unicode=True)

    console.print(
        f"[green]Generated replicate run template with {len(selected_loras)} LoRAs at {output}[/green]"
    )


@app.command(name="run-model")
def run_model(
    model_name: str = typer.Argument(
        ...,
        help="Replicate model name (e.g. 'owner/model')"
    ),
    version: Optional[str] = typer.Argument(
        None,
        help="Optional Replicate model version ID (defaults to the model's latest version)"
    ),
    param: List[str] = typer.Option(
        [],
        "--param",
        "-p",
        help="Model parameters as key:value pairs. Supports auto type inference. "
             "Examples: --param prompt:'hello' --param steps:20 --param scale:0.8 --param flag:true"
    )
):
    """
    Run a Replicate model with multi-parameter support and automatic type inference.

    Type Inference Examples:

        --param steps:20           -> int(20)

        --param scale:0.8          -> float(0.8)

        --param enable:true        -> bool(True)

        --param prompt:"hello"     -> str("hello")

    \b
    Usage Examples:
        # Single parameter (latest version)
        replicate-runner replicate run-model stability-ai/sdxl --param prompt:"a cat"

        # Pin a specific version by adding a second argument or owner/model:version
        replicate-runner replicate run-model lucataco/flux-dev-lora <version> \\
            --param prompt:"a photo of TOK, person, portrait" \\
            --param hf_lora:"steveant/steve-lora-v1.1" \\
            --param lora_scale:0.8 \\
            --param num_outputs:1 \\
            --param num_inference_steps:28
    """
    from replicate_runner.config_loader import ConfigLoader

    config_loader = ConfigLoader()
    replicate_token = config_loader.get("REPLICATE_API_TOKEN", "")

    if not replicate_token:
        console.print(
            "[red]No Replicate token provided. "
            "Set REPLICATE_API_TOKEN in .env or config.[/red]"
        )
        raise typer.Exit(1)

    normalized_model_name, embedded_version = split_model_reference(model_name)

    if version and embedded_version and version != embedded_version:
        console.print(
            "[red]Conflicting version information provided. "
            "Use either -- model:version or a separate VERSION argument, not both.[/red]"
        )
        raise typer.Exit(1)

    resolved_version = version or embedded_version
    model_ref = format_model_reference(normalized_model_name, resolved_version)

    if not resolved_version:
        console.print(
            "[yellow]No explicit version provided. Using the latest version "
            "configured on Replicate.[/yellow]"
        )

    # Parse parameters with type inference
    try:
        input_params = parse_params(param)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print(
            "[yellow]Tip: Use format --param key:value "
            "(e.g., --param prompt:'hello world')[/yellow]"
        )
        raise typer.Exit(1)

    if not input_params:
        console.print(
            "[yellow]Warning: No parameters provided. "
            "Model may require input parameters.[/yellow]"
        )

    # Show parsed parameters for transparency
    console.print("[blue]Parsed parameters:[/blue]")
    for key, value in input_params.items():
        type_name = type(value).__name__
        console.print(f"  {key}: {value!r} ({type_name})")

    hf_token = (
        config_loader.get("HF_API_TOKEN", "")
        or config_loader.get("HF_TOKEN", "")
    )
    if hf_token and "hf_api_token" not in input_params and needs_hf_token(input_params):
        input_params["hf_api_token"] = hf_token
        console.print(
            "[blue]Injected HuggingFace token for private LoRA download.[/blue]"
        )

    prompt_value = input_params.get("prompt")
    if isinstance(prompt_value, str):
        lower_prompt = prompt_value.lower()
        if any(token in lower_prompt for token in ("andie", "audra")) and PERSONA_ACTION_MARKER.lower() not in lower_prompt:
            action = persona_resolver.pick_any()
            if action:
                augmented = f"{prompt_value}, while she is {action} {PERSONA_ACTION_MARKER}"
                input_params["prompt"] = augmented
                console.print(
                    f"[blue]Added random persona action to prompt:[/blue] {augmented}"
                )

    try:
        output = run_replicate_model(
            token=replicate_token,
            model_ref=model_ref,
            input_params=input_params,
        )
        console.print(f"\n[green]✓ Model execution complete[/green]")
        console.print(f"[green]Output:[/green] {output}")

        saved_paths = persist_file_outputs(
            collect_file_outputs(output),
            model_ref=model_ref,
        )
        if saved_paths:
            console.print("[green]Saved files:[/green]")
            for path in saved_paths:
                console.print(f"  - {path}")
    except Exception as e:
        console.print(f"[red]Error running model: {e}[/red]")
        raise typer.Exit(1)
