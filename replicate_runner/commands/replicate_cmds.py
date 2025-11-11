import ast
from typing import Any, List
import replicate
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


def run_replicate_model(
    token: str,
    model_name: str,
    version: str,
    input_params: dict
):
    """
    Runs a replicate model using the given token, model name, version, and parameters.
    For private Hugging Face LoRA models, ensure your token has access.
    """
    client = replicate.Client(api_token=token)
    console.print(f"Running [bold]{model_name}:{version}[/bold]")
    console.print(f"Parameters: {input_params}")

    model = client.models.get(model_name)
    version_obj = model.versions.get(version)

    prediction = version_obj.predict(**input_params)
    return prediction


app = typer.Typer()


@app.command(name="run-model")
def run_model(
    model_name: str = typer.Argument(
        ...,
        help="Replicate model name (e.g. 'owner/model')"
    ),
    version: str = typer.Argument(
        ...,
        help="Replicate model version ID"
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
        # Single parameter
        replicate-runner replicate run-model stability-ai/sdxl v1 --param prompt:"a cat"

        # Multiple parameters with FLUX + LoRA
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

    try:
        output = run_replicate_model(
            token=replicate_token,
            model_name=model_name,
            version=version,
            input_params=input_params,
        )
        console.print(f"\n[green]✓ Model execution complete[/green]")
        console.print(f"[green]Output:[/green] {output}")
    except Exception as e:
        console.print(f"[red]Error running model: {e}[/red]")
        raise typer.Exit(1)
