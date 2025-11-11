import typer
from pathlib import Path
from rich.console import Console
from huggingface_hub import HfApi, create_repo
from collections import defaultdict
import re
import tempfile
import yaml

console = Console()
app = typer.Typer()


def _extract_trigger_from_config(source_dir: Path) -> str:
    """
    Extract trigger word from config.yaml in the source directory.

    Args:
        source_dir: Path to the LoRA directory

    Returns:
        Trigger word from config, or "TOK" as default
    """
    config_path = source_dir / "config.yaml"
    if not config_path.exists():
        console.print("[yellow]No config.yaml found, using default trigger 'TOK'[/yellow]")
        return "TOK"

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Navigate to config.process[0].trigger_word
        if 'config' in config and 'process' in config['config']:
            process_list = config['config']['process']
            if process_list and isinstance(process_list, list) and len(process_list) > 0:
                trigger = process_list[0].get('trigger_word')
                if trigger:
                    console.print(f"[blue]Found trigger word in config.yaml: {trigger}[/blue]")
                    return trigger

        console.print("[yellow]trigger_word not found in config.yaml, using default 'TOK'[/yellow]")
        return "TOK"
    except Exception as e:
        console.print(f"[yellow]Could not read config.yaml: {e}. Using default trigger 'TOK'[/yellow]")
        return "TOK"


def _generate_model_card(
    name: str,
    sample_files: list[Path],
    repo_id: str,
    trigger: str = None,
    description: str = ""
) -> str:
    """
    Generate README.md content with YAML frontmatter and widget-based gallery.

    Args:
        name: Model repository name
        sample_files: List of sample image files to include in gallery
        repo_id: Full repository ID (username/repo-name)
        trigger: Trigger word/phrase for the model
        description: Optional model description

    Returns:
        Formatted README.md content as string
    """
    # Default trigger if not provided
    if not trigger:
        trigger = "TOK"

    # Generate widget entries for each sample
    widget_entries = []
    for i, img in enumerate(sample_files):
        # Create generic prompt template
        widget_entries.append(
            f"-   text: '{trigger}, a person, portrait'\n"
            f"    output:\n"
            f"        url: samples/{img.name}"
        )

    widget_yaml = "\n".join(widget_entries)

    # Create model card with YAML frontmatter
    model_card = f"""---
tags:
- text-to-image
- flux
- lora
- diffusers
- template:sd-lora
- ai-toolkit
widget:
{widget_yaml}
base_model: black-forest-labs/FLUX.1-dev
instance_prompt: {trigger}
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---
# {name}

Model trained with [AI Toolkit by Ostris](https://github.com/ostris/ai-toolkit)

<Gallery />

## Trigger words

You should use `{trigger}` to trigger the image generation.

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, etc.

Weights for this model are available in Safetensors format.

[Download](/{repo_id}/tree/main) them in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-dev', torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='lora.safetensors')
image = pipeline('{trigger}, a person, portrait').images[0]
image.save("my_image.png")
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## Training Details

- Trained with AI Toolkit
- Base model: FLUX.1-dev
- Training type: LoRA
- Total samples: {len(sample_files)}
"""

    return model_card


@app.command(name="publish-hf-lora")
def publish_lora(
    source_dir: str = typer.Option(..., "--source-dir", help="Path to LoRA weights directory or file"),
    name: str = typer.Option(..., "--name", help="Repository name (e.g., 'my-lora-model')"),
    username: str = typer.Option(None, "--username", help="HuggingFace username (defaults to your account)"),
    trigger: str = typer.Option(None, "--trigger", help="Trigger word for the LoRA (default: TOK)"),
    private: bool = typer.Option(True, "--private", help="Make repo private (default: True)"),
    exist_ok: bool = typer.Option(True, "--exist-ok/--no-exist-ok", help="Don't error if repo exists"),
):
    """
    Publish a LoRA model to HuggingFace Hub as a private repository.

    Examples:
        replicate-runner hf publish-hf-lora --name my-lora --source-dir ./weights --trigger "person" --private
        replicate-runner hf publish-hf-lora --name my-lora --source-dir ./lora.safetensors --username myuser --no-private
    """
    from replicate_runner.config_loader import ConfigLoader

    config_loader = ConfigLoader()
    hf_token = config_loader.get("HF_TOKEN", "")
    if not hf_token:
        console.print("[red]No HuggingFace token provided. Set HF_TOKEN in .env or config.[/red]")
        raise typer.Exit(1)

    # Get username from API if not provided
    api = HfApi(token=hf_token)
    if not username:
        try:
            user_info = api.whoami(token=hf_token)
            username = user_info["name"]
            console.print(f"[blue]Using username: {username}[/blue]")
        except Exception as e:
            console.print(f"[red]Could not determine username: {e}[/red]")
            raise typer.Exit(1)

    repo_id = f"{username}/{name}"

    # Validate path exists
    path = Path(source_dir)
    if not path.exists():
        console.print(f"[red]Path does not exist: {source_dir}[/red]")
        raise typer.Exit(1)

    # Extract trigger word from config.yaml if not provided
    if not trigger:
        trigger = _extract_trigger_from_config(path)

    try:
        # Find .safetensors file (without "index" in name)
        safetensors_files = [f for f in path.glob("*.safetensors") if "index" not in f.name.lower()]
        if not safetensors_files:
            console.print(f"[red]No .safetensors file found in {source_dir}[/red]")
            raise typer.Exit(1)
        safetensors_file = safetensors_files[0]
        console.print(f"[blue]Found LoRA weights: {safetensors_file.name}[/blue]")

        # Find samples directory and get last unique images
        samples_dir = path / "samples"
        sample_files = []
        if samples_dir.exists() and samples_dir.is_dir():
            # Group images by prompt index (the last number after underscore)
            image_groups = defaultdict(list)
            for img in samples_dir.glob("*.*"):
                if img.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue

                # Pattern: timestamp__step_index.ext
                # We want to group by index (last number) and take the latest step
                match = re.match(r"(\d+)__(\d+)_(\d+)\.", img.name)
                if match:
                    timestamp = int(match.group(1))
                    step = int(match.group(2))
                    index = int(match.group(3))
                    image_groups[index].append((step, timestamp, img))
                else:
                    # If pattern doesn't match, include it anyway
                    image_groups[img.stem].append((0, 0, img))

            # Get the highest step/timestamp image from each group
            for index, images in image_groups.items():
                latest = max(images, key=lambda x: (x[0], x[1]))  # Sort by step, then timestamp
                sample_files.append(latest[2])
                console.print(f"[blue]Including sample: {latest[2].name}[/blue]")
        else:
            console.print("[yellow]No samples directory found, skipping sample images[/yellow]")

        # Create repo if it doesn't exist
        console.print(f"[blue]Creating repository: {repo_id}[/blue]")
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=private,
            exist_ok=exist_ok,
            repo_type="model"
        )

        # Upload safetensors file
        console.print(f"[blue]Uploading {safetensors_file.name}...[/blue]")
        api.upload_file(
            path_or_fileobj=str(safetensors_file),
            path_in_repo=safetensors_file.name,
            repo_id=repo_id,
            repo_type="model",
        )

        # Upload sample images
        for sample in sample_files:
            console.print(f"[blue]Uploading sample {sample.name}...[/blue]")
            api.upload_file(
                path_or_fileobj=str(sample),
                path_in_repo=f"samples/{sample.name}",
                repo_id=repo_id,
                repo_type="model",
            )

        # Generate and upload model card (README.md)
        console.print("[blue]Generating model card (README.md)...[/blue]")
        readme_content = _generate_model_card(name, sample_files, repo_id, trigger)

        # Create temporary README.md file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
            tmp.write(readme_content)
            tmp_path = tmp.name

        try:
            console.print("[blue]Uploading model card...[/blue]")
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

        visibility = "private" if private else "public"
        total_files = 1 + len(sample_files) + 1  # model + samples + README
        console.print(f"[green]SUCCESS: Published {visibility} LoRA to: https://huggingface.co/{repo_id}[/green]")
        console.print(f"[green]Uploaded {total_files} files (1 model + {len(sample_files)} samples + README)[/green]")

    except Exception as e:
        console.print(f"[red]Error publishing LoRA: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_models(
    username: str = typer.Option(None, "--user", help="Filter by username (default: your account)"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of models to list"),
):
    """
    List your HuggingFace models.

    Examples:
        replicate-runner hf list-models
        replicate-runner hf list-models --user username --limit 20
    """
    from replicate_runner.config_loader import ConfigLoader

    config_loader = ConfigLoader()
    hf_token = config_loader.get("HF_TOKEN", "")
    if not hf_token:
        console.print("[red]No HuggingFace token provided. Set HF_TOKEN in .env or config.[/red]")
        raise typer.Exit(1)

    try:
        api = HfApi(token=hf_token)

        # Get user info if no username specified
        if not username:
            user_info = api.whoami(token=hf_token)
            username = user_info["name"]
            console.print(f"[blue]Listing models for: {username}[/blue]\n")

        models = api.list_models(author=username, limit=limit)

        for i, model in enumerate(models, 1):
            private_badge = "[red](private)[/red]" if model.private else "[green](public)[/green]"
            console.print(f"{i}. {model.id} {private_badge}")
            if model.downloads:
                console.print(f"   Downloads: {model.downloads}")
            console.print()

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        raise typer.Exit(1)
