from __future__ import annotations

from typing import Optional

import replicate
import typer
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from rich.console import Console

from replicate_runner.config_loader import ConfigLoader
from replicate_runner.lora_catalog import load_lora_catalog
from replicate_runner.profiles import ProfileManager, build_profile_updates, USER_SCOPE, PROFILE_SCOPES

console = Console()
app = typer.Typer(help="Guided exploration of models and LoRAs", invoke_without_command=True)
hf_api = HfApi()


@app.callback()
def explore_root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


def _scope_callback(value: str) -> str:
    scope = (value or USER_SCOPE).lower()
    if scope not in PROFILE_SCOPES:
        raise typer.BadParameter(f"Scope must be one of: {', '.join(PROFILE_SCOPES)}")
    return scope


def _replicate_client() -> replicate.Client:
    loader = ConfigLoader()
    token = loader.get("REPLICATE_API_TOKEN", "")
    if not token:
        console.print("[red]Set REPLICATE_API_TOKEN to explore models.[/red]")
        raise typer.Exit(1)
    return replicate.Client(api_token=token)


@app.command("models")
def explore_models(
    search: Optional[str] = typer.Option(None, "--search", help="Search term"),
    model: Optional[str] = typer.Option(None, "--model", help="Show details for a specific owner/name"),
    limit: int = typer.Option(10, "--limit", help="How many models to list"),
    save_profile: Optional[str] = typer.Option(None, "--save-profile", help="Write a profile using --model metadata"),
    scope: str = typer.Option(USER_SCOPE, "--scope", help="Scope for --save-profile", callback=_scope_callback),
):
    client = _replicate_client()

    if model:
        try:
            model_obj = client.models.get(model)
        except Exception as exc:
            console.print(f"[red]Unable to load model {model}: {exc}[/red]")
            raise typer.Exit(1)

        console.print(f"[bold blue]{model_obj.id}[/bold blue]")
        if model_obj.description:
            console.print(model_obj.description)
        console.print(f"Runs: {model_obj.run_count}")
        if model_obj.latest_version:
            console.print(f"Latest version: {model_obj.latest_version.id}")

        if save_profile:
            manager = ProfileManager()
            updates = build_profile_updates(
                description=model_obj.description,
                model=model_obj.id,
                version=model_obj.latest_version.id if model_obj.latest_version else None,
            )
            manager.save_profile(save_profile, updates, scope=scope)
            console.print(f"[green]Saved profile '{save_profile}' for {model_obj.id}[/green]")
        return

    if save_profile:
        console.print("[red]--save-profile requires --model.[/red]")
        raise typer.Exit(1)

    try:
        page = client.models.search(search) if search else client.models.list()
    except Exception as exc:
        console.print(f"[red]Failed to fetch models: {exc}[/red]")
        raise typer.Exit(1)

    console.print("[bold blue]Models[/bold blue]")
    count = 0
    for entry in page:
        console.print(f"- [green]{entry.id}[/green] :: {entry.description or ''} (runs: {entry.run_count})")
        count += 1
        if count >= limit:
            break


def _hf_info(repo_id: str) -> Optional[str]:
    try:
        info = hf_api.model_info(repo_id)
    except HfHubHTTPError:
        return None
    return f"likes: {info.likes}, downloads: {info.downloads}" if info else None


@app.command("loras")
def explore_loras(
    key: Optional[str] = typer.Option(None, "--key", help="Show a single LoRA entry"),
    search: Optional[str] = typer.Option(None, "--search", help="Filter by name"),
    save_profile: Optional[str] = typer.Option(None, "--save-profile", help="Create a profile seeded with this LoRA"),
    model: Optional[str] = typer.Option(None, "--model", help="Model to pair when saving a profile"),
    scope: str = typer.Option(USER_SCOPE, "--scope", help="Scope for saved profile", callback=_scope_callback),
):
    catalog = load_lora_catalog()
    entries = list(catalog.loras.values())

    if key:
        entry = catalog.loras.get(key)
        if not entry:
            console.print(f"[red]Unknown LoRA '{key}'.[/red]")
            raise typer.Exit(1)
        entries = [entry]

    if search:
        needle = search.lower()
        entries = [e for e in entries if needle in e.key.lower() or needle in (e.name or '').lower()]

    if not entries:
        console.print("[yellow]No LoRAs matched.[/yellow]")
        raise typer.Exit(0)

    for entry in entries:
        console.print(f"[bold cyan]{entry.name}[/bold cyan] (key: {entry.key})")
        console.print(f"Trigger: {entry.trigger}")
        console.print(f"Repo: {entry.repo_id}")
        if entry.description:
            console.print(entry.description)
        if entry.default_prompt:
            console.print(f"Default prompt: {entry.default_prompt}")
        stats = _hf_info(entry.repo_id)
        if stats:
            console.print(stats)
        console.print()

    if save_profile:
        if not key:
            console.print("[red]Specify --key when using --save-profile for loras.[/red]")
            raise typer.Exit(1)
        if not model:
            console.print("[red]Provide --model to pair with this LoRA when saving.[/red]")
            raise typer.Exit(1)
        entry = catalog.loras[key]
        manager = ProfileManager()
        updates = build_profile_updates(
            description=entry.description,
            model=model,
            lora=entry.lora_weights,
            trigger=entry.trigger,
            prompt_template=entry.default_prompt,
            persona_tokens=[entry.trigger],
        )
        manager.save_profile(save_profile, updates, scope=scope)
        console.print(f"[green]Saved profile '{save_profile}' using LoRA {entry.key}[/green]")
