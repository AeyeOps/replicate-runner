import ast
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

import replicate
from replicate.helpers import FileOutput
import typer
from rich.console import Console

console = Console()


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
    dest_dir = Path("output") / f"{safe_model}_{timestamp}"
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


def run_replicate_model(
    token: str,
    model_ref: str,
    input_params: dict
):
    """Run a Replicate model reference using the official client."""

    client = replicate.Client(api_token=token)
    console.print(f"Running [bold]{model_ref}[/bold]")
    console.print(f"Parameters: {input_params}")

    prediction = client.run(model_ref, input=input_params)
    return prediction


app = typer.Typer()


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
