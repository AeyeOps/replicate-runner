from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console

from replicate_runner.commands.profile_cmds import (
    ProfileRunError,
    prepare_profile_run,
)
from replicate_runner.commands.replicate_cmds import parse_params, split_model_reference, run_replicate_model
from replicate_runner.config_loader import ConfigLoader
from replicate_runner.prompt_engine import (
    FALLBACK_PROMPT_TEMPLATE,
    escape_prompt_literal,
    extract_tokens,
)
from replicate_runner.profiles import ProfileManager, ProfileSource, ResolvedProfile

console = Console()
app = typer.Typer(help="Guided prompt wizard", invoke_without_command=True)

DEFAULT_BASE_MODEL = "black-forest-labs/flux-dev-lora"
BASE_MODEL_GUARDS = {DEFAULT_BASE_MODEL}


@app.callback()
def prompt_root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


def _normalize_lora_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    raw = value.strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered.startswith("http://") or lowered.startswith("https://"):
        return raw
    if "huggingface.co/" in lowered:
        return raw
    if "://" in raw:
        return raw
    if raw.endswith(".safetensors") or raw.endswith(".bin"):
        return raw
    if "/" in raw:
        return f"huggingface.co/{raw}"
    return raw


def _maybe_swap_model_and_lora(values: Dict[str, Optional[str]]):
    model = values.get("model")
    lora = values.get("lora")
    if lora and lora in BASE_MODEL_GUARDS and model and model not in BASE_MODEL_GUARDS:
        console.print(
            f"[yellow]LoRA input '{lora}' looks like a base model. Swapping so the Replicate model is '{lora}' and the LoRA uses '{model}'.[/yellow]"
        )
        values["model"], values["lora"] = lora, model
    values["lora"] = _normalize_lora_value(values.get("lora"))


def _require_value(label: str, current: Optional[str], no_interactive: bool) -> str:
    if current:
        return current
    if no_interactive:
        raise typer.BadParameter(f"Missing required value for {label} under --no-interactive mode")
    return typer.prompt(label).strip()


def _build_profile_stub(
    model: str,
    version: Optional[str],
    lora: Optional[str],
    template: str,
) -> ResolvedProfile:
    data = {
        "model": model,
        "version": version,
        "lora": lora,
        "prompt_template": template,
        "defaults": {},
    }
    return ResolvedProfile(name="<wizard>", data=data, sources=[ProfileSource(scope="wizard", path=Path("<wizard>"))])


def _prompt_model_value(existing: Optional[str], resolved: Optional[ResolvedProfile], no_interactive: bool) -> str:
    if existing:
        return existing
    if resolved and resolved.data.get("model"):
        return resolved.data.get("model")
    if no_interactive:
        return DEFAULT_BASE_MODEL
    return typer.prompt(
        "Replicate model (owner/name)",
        default=DEFAULT_BASE_MODEL,
    ).strip()


def _determine_prompt_fields(
    template: str,
    resolved: Optional[ResolvedProfile],
    *,
    model: Optional[str],
    lora: Optional[str],
    subject: Optional[str],
    mood: Optional[str],
    action: Optional[str],
    camera: Optional[str],
    lighting: Optional[str],
    base_model_only: bool,
    no_interactive: bool,
) -> Dict[str, Optional[str]]:
    tokens = extract_tokens(template)
    data = resolved.data if resolved else {}

    model_value = _prompt_model_value(model, resolved, no_interactive)

    lora_value = lora or data.get("lora")
    if not base_model_only and not lora_value:
        lora_value = _require_value("LoRA (hf repo or weights)", None, no_interactive)

    subject_value = subject or (data.get("defaults") or {}).get("subject")
    trigger = data.get("trigger")

    if "subject" in tokens and not subject_value:
        subject_value = _require_value("Prompt subject", subject_value, no_interactive)

    if "subject_or_trigger" in tokens and not (trigger or subject_value):
        subject_value = _require_value("Prompt subject", subject_value, no_interactive)

    mood_value = mood
    if "mood" in tokens and mood_value is None and not no_interactive:
        mood_value = typer.prompt("Mood", default="").strip() or None

    action_value = action
    if ("action" in tokens or "persona_action" in tokens) and action_value is None and not no_interactive:
        action_value = typer.prompt("Action", default="").strip() or None

    camera_value = camera
    if "camera" in tokens and camera_value is None and not no_interactive:
        camera_value = typer.prompt("Camera", default="").strip() or None

    lighting_value = lighting
    if "lighting" in tokens and lighting_value is None and not no_interactive:
        lighting_value = typer.prompt("Lighting", default="").strip() or None

    values = {
        "model": model_value,
        "lora": lora_value,
        "subject": subject_value,
        "mood": mood_value,
        "action": action_value,
        "camera": camera_value,
        "lighting": lighting_value,
    }

    _maybe_swap_model_and_lora(values)
    return values


@app.command("wizard")
def prompt_wizard(
    profile: Optional[str] = typer.Option(None, "--profile", help="Seed the wizard from an existing profile"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model reference"),
    version: Optional[str] = typer.Option(None, "--version", help="Override model version"),
    lora: Optional[str] = typer.Option(None, "--lora", help="Override LoRA reference"),
    subject: Optional[str] = typer.Option(None, "--subject", help="Override subject"),
    mood: Optional[str] = typer.Option(None, "--mood", help="Mood descriptor"),
    action: Optional[str] = typer.Option(None, "--action", help="Explicit action text"),
    camera: Optional[str] = typer.Option(None, "--camera", help="Camera details"),
    lighting: Optional[str] = typer.Option(None, "--lighting", help="Lighting details"),
    persona_action: Optional[bool] = typer.Option(None, "--persona-action/--no-persona-action", help="Toggle persona injection"),
    base_model_only: bool = typer.Option(False, "--base-model-only", help="Skip LoRA and run the base model"),
    prompt: Optional[str] = typer.Option(None, "--prompt", help="Literal prompt override"),
    param: List[str] = typer.Option([], "--param", help="Model parameter overrides"),
    no_interactive: bool = typer.Option(False, "--no-interactive", help="Error instead of prompting"),
    run: bool = typer.Option(False, "--run", help="Execute immediately"),
):
    manager = ProfileManager()
    resolved: Optional[ResolvedProfile] = None

    if profile:
        try:
            resolved = manager.resolve_profile(profile)
        except KeyError:
            console.print(f"[red]Profile '{profile}' not found.[/red]")
            raise typer.Exit(1)

    template = (resolved.data.get("prompt_template") if resolved else None) or FALLBACK_PROMPT_TEMPLATE

    try:
        values = _determine_prompt_fields(
            template,
            resolved,
            model=model,
            lora=lora,
            subject=subject,
            mood=mood,
            action=action,
            camera=camera,
            lighting=lighting,
            base_model_only=base_model_only,
            no_interactive=no_interactive,
        )
    except typer.BadParameter as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    try:
        param_overrides = parse_params(param)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    execution_profile = resolved
    if execution_profile is None:
        execution_profile = _build_profile_stub(values["model"], version, values["lora"], template)

    profile_has_model = resolved is not None and bool(resolved.data.get("model"))
    profile_has_lora = resolved is not None and bool(resolved.data.get("lora"))

    model_override = model if model is not None else (values["model"] if not profile_has_model else None)
    lora_override = lora if lora is not None else (values["lora"] if not profile_has_lora else None)

    try:
        plan = prepare_profile_run(
            execution_profile,
            model_override=model_override,
            version_override=version,
            lora_override=lora_override,
            subject=values["subject"],
            mood=values["mood"],
            action=values["action"],
            camera=values["camera"],
            lighting=values["lighting"],
            prompt_override=prompt,
            persona_flag=persona_action,
            base_model_only=base_model_only,
            param_overrides=param_overrides,
        )
    except ProfileRunError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Prompt[/bold blue]: {plan.prompt}")

    escaped_prompt = escape_prompt_literal(plan.prompt)

    if profile:
        cmd: List[str] = ["replicate-runner", "profile", "run", profile]
        if base_model_only:
            cmd.append("--base-model-only")
        if model_override:
            cmd.extend(["--model", model_override])
        if version:
            cmd.extend(["--version", version])
        if lora_override:
            cmd.extend(["--lora", lora_override])
        if persona_action is False:
            cmd.append("--no-persona-action")
        elif persona_action is True:
            cmd.append("--persona-action")
        for raw_param in param:
            cmd.extend(["--param", raw_param])
        cmd.extend(["--prompt", f'"{escaped_prompt}"'])
    else:
        owner_model, embedded_version = split_model_reference(values["model"])
        cmd = ["replicate-runner", "replicate", "run-model", owner_model]
        version_arg = version or embedded_version
        if version_arg:
            cmd.append(version_arg)
        cmd.extend(["--param", f'prompt:"{escaped_prompt}"'])
        if values["lora"] and not base_model_only:
            cmd.extend(["--param", f'hf_lora:"{values["lora"]}"'])
        for raw_param in param:
            cmd.extend(["--param", raw_param])

    console.print(f"[bold blue]Command[/bold blue]: {' '.join(cmd)}")

    if not run:
        return

    config_loader = ConfigLoader()
    token = config_loader.get("REPLICATE_API_TOKEN", "")
    if not token:
        console.print("[red]Set REPLICATE_API_TOKEN to run immediately.[/red]")
        raise typer.Exit(1)

    try:
        run_replicate_model(token=token, model_ref=plan.model_name, input_params=plan.params)
    except Exception as exc:
        console.print(f"[red]Wizard run failed: {exc}[/red]")
        raise typer.Exit(1)

    console.print("[green]Wizard run complete.[/green]")
