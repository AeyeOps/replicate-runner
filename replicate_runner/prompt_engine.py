from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set

from replicate_runner.persona import PERSONA_ACTION_MARKER, PersonaActionResolver

FALLBACK_PROMPT_TEMPLATE = (
    "{subject_or_trigger}, {mood}, while she is {persona_action} with {camera} lighting"
)

TOKEN_PATTERN = re.compile(r"{([^{}]+)}")


class PromptError(RuntimeError):
    pass


def extract_tokens(template: str) -> Set[str]:
    return set(TOKEN_PATTERN.findall(template or ""))


def render_template(template: str, values: Dict[str, Optional[str]]) -> str:
    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token in values:
            value = values.get(token)
            return str(value) if value is not None else ""
        return match.group(0)

    return TOKEN_PATTERN.sub(_replace, template)


def normalize_prompt(text: str) -> str:
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return collapsed.strip()


def escape_prompt_literal(text: str) -> str:
    return text.replace("\"", r"\"")


def resolve_persona_action_value(
    *,
    tokens: Set[str],
    action_override: Optional[str],
    persona_enabled: bool,
    persona_tokens: Sequence[str],
    resolver: PersonaActionResolver,
) -> str:
    if "persona_action" not in tokens:
        return ""
    if action_override and (not persona_enabled or "action" not in tokens):
        return action_override
    if not persona_enabled:
        return ""
    choice = resolver.pick(persona_tokens)
    return choice or ""


def render_prompt(
    template: str,
    *,
    trigger: Optional[str],
    subject: Optional[str],
    mood: Optional[str],
    action_text: Optional[str],
    persona_action_text: Optional[str],
    camera: Optional[str],
    lighting: Optional[str],
) -> str:
    tokens = extract_tokens(template)
    values: Dict[str, Optional[str]] = {
        "trigger": trigger,
        "subject": subject,
        "mood": mood,
        "action": action_text,
        "persona_action": persona_action_text,
        "camera": camera,
        "lighting": lighting,
    }

    if "subject_or_trigger" in tokens:
        if trigger:
            values["subject_or_trigger"] = trigger
        elif subject:
            values["subject_or_trigger"] = subject
        else:
            raise PromptError("Prompt template requires a trigger or subject for {subject_or_trigger}.")

    if "subject" in tokens and not subject:
        raise PromptError("Prompt template requires a subject, but none was provided.")

    rendered = render_template(template, values)
    rendered = normalize_prompt(rendered)

    if persona_action_text and PERSONA_ACTION_MARKER not in rendered:
        rendered = f"{rendered} {PERSONA_ACTION_MARKER}".strip()

    return rendered
