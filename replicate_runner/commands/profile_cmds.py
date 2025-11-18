from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console

from replicate_runner.commands.replicate_cmds import (
    format_model_reference,
    parse_params,
    run_replicate_model,
    split_model_reference,
)
from replicate_runner.config_loader import ConfigLoader
from replicate_runner.persona import PersonaActionResolver
from replicate_runner.profile_runtime import merge_params, parse_profile_defaults
from replicate_runner.prompt_engine import (
    FALLBACK_PROMPT_TEMPLATE,
    PromptError,
    extract_tokens,
    normalize_prompt,
    render_prompt,
    resolve_persona_action_value,
)
from replicate_runner.profiles import (
    PROFILE_SCOPES,
    USER_SCOPE,
    ProfileManager,
    ResolvedProfile,
    build_profile_updates,
)


console = Console()
app = typer.Typer(help="Manage saved prompt profiles.", invoke_without_command=True)
persona_resolver = PersonaActionResolver()


@app.callback()
def profile_root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


class ProfileRunError(RuntimeError):
    pass


@dataclass
class ProfileExecutionPlan:
    resolved: ResolvedProfile
    model_name: str
    version: Optional[str]
    lora: Optional[str]
    prompt: str
    params: Dict[str, Any]
    base_model_only: bool
    persona_action_text: str
    template_used: str


def _scope_callback(value: str) -> str:
    scope = (value or USER_SCOPE).lower()
    if scope not in PROFILE_SCOPES:
        raise typer.BadParameter(f"Scope must be one of: {', '.join(PROFILE_SCOPES)}")
    return scope


def _profile_manager() -> ProfileManager:
    return ProfileManager()


def _resolve_persona_enabled(default_value: bool, override: Optional[bool]) -> bool:
    if override is None:
        return default_value
    return override


def prepare_profile_run(
    resolved: ResolvedProfile,
    *,
    model_override: Optional[str] = None,
    version_override: Optional[str] = None,
    lora_override: Optional[str] = None,
    subject: Optional[str] = None,
    mood: Optional[str] = None,
    action: Optional[str] = None,
    camera: Optional[str] = None,
    lighting: Optional[str] = None,
    prompt_override: Optional[str] = None,
    persona_flag: Optional[bool] = None,
    base_model_only: bool = False,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> ProfileExecutionPlan:
    data = resolved.data
    defaults = parse_profile_defaults(data.get("defaults"))
    template = data.get("prompt_template") or FALLBACK_PROMPT_TEMPLATE

    profile_model = data.get("model")
    if not profile_model and not model_override:
        raise ProfileRunError(
            f"Profile '{resolved.name}' is missing a model. Use `profile save --model` or pass --model at runtime."
        )

    model_input = model_override or profile_model or ""
    normalized_model, embedded_version = split_model_reference(model_input)

    profile_version = data.get("version") or embedded_version
    override_version = version_override
    if override_version and embedded_version and override_version != embedded_version:
        raise ProfileRunError(
            "Conflicting version details detected. Specify the version either as part of --model or with --version, not both."
        )
    resolved_version = override_version or profile_version or embedded_version

    profile_lora = data.get("lora")
    if base_model_only and lora_override:
        raise ProfileRunError("Cannot combine --lora with --base-model-only.")

    resolved_lora = None if base_model_only else (lora_override or profile_lora)
    if not resolved_lora and not base_model_only:
        raise ProfileRunError(
            "No LoRA configured. Provide --lora or confirm the base model with --base-model-only."
        )

    resolved_subject = subject or defaults.subject
    trigger = data.get("trigger")
    persona_tokens = defaults.persona_tokens or ([trigger] if trigger else [])
    persona_enabled = _resolve_persona_enabled(defaults.persona_enabled, persona_flag)

    if prompt_override:
        prompt_text = normalize_prompt(prompt_override)
        persona_action_text = ""
    else:
        token_set = extract_tokens(template)
        persona_action_text = resolve_persona_action_value(
            tokens=token_set,
            action_override=action,
            persona_enabled=persona_enabled,
            persona_tokens=persona_tokens,
            resolver=persona_resolver,
        )

        try:
            prompt_text = render_prompt(
                template,
                trigger=trigger,
                subject=resolved_subject,
                mood=mood,
                action_text=action,
                persona_action_text=persona_action_text,
                camera=camera,
                lighting=lighting,
            )
        except PromptError as exc:
            raise ProfileRunError(str(exc)) from exc

    params = merge_params(
        defaults.params,
        param_overrides or {},
        prompt_text,
        resolved_lora,
        base_model_only,
    )

    model_ref = format_model_reference(normalized_model, resolved_version)

    return ProfileExecutionPlan(
        resolved=resolved,
        model_name=model_ref,
        version=resolved_version,
        lora=resolved_lora,
        prompt=prompt_text,
        params=params,
        base_model_only=base_model_only,
        persona_action_text=persona_action_text,
        template_used=template,
    )


@app.command("list")
def list_profiles():
    """List all resolved profiles."""

    manager = _profile_manager()
    names = manager.available_profiles()
    if not names:
        console.print("[yellow]No profiles found. Use `profile save` or the prompt wizard to create one.[/yellow]")
        raise typer.Exit(0)

    console.print("[bold blue]Available profiles:[/bold blue]")
    for name in names:
        try:
            resolved = manager.resolve_profile(name)
        except KeyError:
            continue
        description = resolved.data.get("description") or ""
        model = resolved.data.get("model") or "<missing>"
        console.print(f"- [green]{name}[/green] – {model} {description}")


@app.command("show")
def show_profile(name: str = typer.Argument(..., help="Profile name")):
    """Display merged profile details."""

    manager = _profile_manager()
    try:
        resolved = manager.resolve_profile(name)
    except KeyError:
        console.print(f"[red]Profile '{name}' not found.[/red]")
        raise typer.Exit(1)

    data = resolved.data
    console.print(f"[bold blue]{name}[/bold blue]")
    if data.get("description"):
        console.print(data["description"])
    console.print(f"Model: {data.get('model') or '<missing>'}")
    if data.get("version"):
        console.print(f"Version: {data['version']}")
    console.print(f"LoRA: {data.get('lora') or '<none>'}")
    console.print(f"Trigger: {data.get('trigger') or '<none>'}")
    console.print(f"Prompt template: {data.get('prompt_template') or FALLBACK_PROMPT_TEMPLATE}")

    defaults = data.get("defaults") or {}
    if defaults:
        console.print("Defaults:")
        for key, value in defaults.items():
            console.print(f"  - {key}: {value}")

    console.print("Sources:")
    for source in resolved.sources:
        console.print(f"  - {source.scope}: {source.path}")


@app.command("save")
def save_profile(
    name: str = typer.Argument(..., help="Profile name to create/update"),
    description: Optional[str] = typer.Option(None, "--description", help="Short summary"),
    model: Optional[str] = typer.Option(None, "--model", help="Replicate model (owner/name or owner/name:version)"),
    version: Optional[str] = typer.Option(None, "--version", help="Explicit model version"),
    lora: Optional[str] = typer.Option(None, "--lora", help="Default LoRA reference"),
    trigger: Optional[str] = typer.Option(None, "--trigger", help="Trigger token used in templates"),
    prompt_template: Optional[str] = typer.Option(None, "--prompt-template", help="Custom template string"),
    subject: Optional[str] = typer.Option(None, "--subject", help="Default subject text"),
    persona_token: Optional[List[str]] = typer.Option(None, "--persona-token", help="Persona token(s) for action filtering"),
    persona_enabled: Optional[bool] = typer.Option(None, "--persona-enabled/--no-persona-enabled", help="Toggle persona injection by default"),
    param: List[str] = typer.Option([], "--param", help="Default parameter overrides (key:value)"),
    scope: str = typer.Option(USER_SCOPE, "--scope", help="Which layer to write", callback=_scope_callback),
    unset: List[str] = typer.Option([], "--unset", help="Dot-path to remove at this scope"),
):
    """Create or update a profile at the selected scope."""

    manager = _profile_manager()
    try:
        default_params = parse_params(param)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    persona_tokens = persona_token if persona_token is not None else None
    updates = build_profile_updates(
        description=description,
        model=model,
        version=version,
        lora=lora,
        trigger=trigger,
        prompt_template=prompt_template,
        subject=subject,
        persona_tokens=persona_tokens,
        persona_enabled=persona_enabled,
        defaults_params=default_params,
    )

    if not updates and not unset:
        console.print("[yellow]Nothing to update. Provide fields to save or --unset paths.[/yellow]")
        raise typer.Exit(0)

    try:
        location = manager.save_profile(name, updates, scope=scope, unset_paths=unset)
    except PermissionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[green]Saved profile '{name}' to {scope} scope at {location}[/green]"
    )


@app.command("delete")
def delete_profile(
    name: str = typer.Argument(..., help="Profile name to delete"),
    scope: str = typer.Option(USER_SCOPE, "--scope", help="Scope to modify", callback=_scope_callback),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
):
    """Remove a profile definition from a single scope."""

    if not yes:
        confirmed = typer.confirm(f"Delete profile '{name}' from {scope}?", default=False)
        if not confirmed:
            console.print("[yellow]Deletion cancelled.[/yellow]")
            raise typer.Exit(0)

    manager = _profile_manager()
    removed = manager.delete_profile(name, scope=scope)
    if not removed:
        console.print(f"[yellow]Profile '{name}' not present in {scope} scope.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]Removed profile '{name}' from {scope} scope.[/green]")


@app.command("run")
def run_profile(
    name: str = typer.Argument(..., help="Profile name"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model reference"),
    version: Optional[str] = typer.Option(None, "--version", help="Override model version"),
    lora: Optional[str] = typer.Option(None, "--lora", help="Override LoRA reference"),
    subject: Optional[str] = typer.Option(None, "--subject", help="Override subject token"),
    mood: Optional[str] = typer.Option(None, "--mood", help="Mood descriptor"),
    action: Optional[str] = typer.Option(None, "--action", help="Explicit action to insert"),
    camera: Optional[str] = typer.Option(None, "--camera", help="Camera notes"),
    lighting: Optional[str] = typer.Option(None, "--lighting", help="Lighting description"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Bypass template and use a literal prompt"),
    persona_action: Optional[bool] = typer.Option(
        None,
        "--persona-action/--no-persona-action",
        help="Temporarily enable/disable persona injection",
    ),
    base_model_only: bool = typer.Option(False, "--base-model-only", help="Acknowledge running without LoRA"),
    param: List[str] = typer.Option([], "--param", help="Model input overrides (key:value)"),
):
    """Resolve a profile and execute it on Replicate."""

    manager = _profile_manager()
    try:
        resolved = manager.resolve_profile(name)
    except KeyError:
        console.print(f"[red]Profile '{name}' not found.[/red]")
        raise typer.Exit(1)

    try:
        param_overrides = parse_params(param)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    try:
        plan = prepare_profile_run(
            resolved,
            model_override=model,
            version_override=version,
            lora_override=lora,
            subject=subject,
            mood=mood,
            action=action,
            camera=camera,
            lighting=lighting,
            prompt_override=prompt,
            persona_flag=persona_action,
            base_model_only=base_model_only,
            param_overrides=param_overrides,
        )
    except ProfileRunError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    config_loader = ConfigLoader()
    token = config_loader.get("REPLICATE_API_TOKEN", "")
    if not token:
        console.print("[red]Set REPLICATE_API_TOKEN to run profiles.[/red]")
        raise typer.Exit(1)

    if plan.resolved.sources:
        source_bits = [f"{src.scope} ({src.path})" for src in plan.resolved.sources]
        console.print(f"[blue]Sources:[/blue] {' -> '.join(source_bits)}")

    console.print(f"[blue]Model:[/blue] {plan.model_name}")
    console.print(f"[blue]LoRA:[/blue] {plan.lora or '<base model>'}")
    console.print(f"[blue]Prompt:[/blue] {plan.prompt}")

    try:
        run_replicate_model(token=token, model_ref=plan.model_name, input_params=plan.params)
    except Exception as exc:
        console.print(f"[red]Failed to run profile: {exc}[/red]")
        raise typer.Exit(1)

    console.print("[green]Prediction complete.[/green]")
