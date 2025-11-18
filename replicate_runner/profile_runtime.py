from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProfileDefaults:
    params: Dict[str, Any]
    subject: Optional[str]
    persona_tokens: List[str]
    persona_enabled: bool


def parse_profile_defaults(raw_defaults: Optional[Dict[str, Any]]) -> ProfileDefaults:
    defaults = raw_defaults or {}
    params = defaults.get("params") if isinstance(defaults, dict) else {}
    parsed_params = deepcopy(params) if isinstance(params, dict) else {}

    subject = defaults.get("subject") if isinstance(defaults, dict) else None
    persona_tokens = defaults.get("persona_tokens") if isinstance(defaults, dict) else []
    if not isinstance(persona_tokens, list):
        persona_tokens = []
    persona_tokens = [str(token) for token in persona_tokens if token]

    persona_enabled_raw = defaults.get("persona_enabled") if isinstance(defaults, dict) else None
    persona_enabled = True if persona_enabled_raw is None else bool(persona_enabled_raw)

    return ProfileDefaults(
        params=parsed_params,
        subject=str(subject) if subject is not None else None,
        persona_tokens=persona_tokens,
        persona_enabled=persona_enabled,
    )


def merge_params(
    base_params: Dict[str, Any],
    overrides: Dict[str, Any],
    prompt: str,
    lora_reference: Optional[str],
    base_model_only: bool,
) -> Dict[str, Any]:
    merged = deepcopy(base_params)
    merged["prompt"] = prompt

    if base_model_only:
        merged.pop("hf_lora", None)
        merged.pop("lora_uri", None)
        merged.pop("extra_lora", None)
    elif lora_reference:
        merged["hf_lora"] = lora_reference

    merged.update(overrides)
    return merged
