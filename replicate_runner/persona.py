from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import yaml

PERSONA_ACTION_MARKER = "(persona_action)"


@dataclass
class PersonaAction:
    tokens: List[str]
    text: str


DEFAULT_ACTIONS = [
    PersonaAction(tokens=["andie", "audra", "ariel"], text="twirling a transparent umbrella in the rain"),
    PersonaAction(tokens=["andie", "audra"], text="balancing on subway stairs while tying her shoe"),
    PersonaAction(tokens=["andie", "audra", "stevie"], text="sketching neon fashion concepts on a tablet"),
    PersonaAction(tokens=["audra"], text="checking vintage film on a retro camera"),
    PersonaAction(tokens=["ariel", "audra"], text="sipping espresso outside a Parisian cafe"),
    PersonaAction(tokens=["andie", "stevie"], text="reading choreography notes beside a boombox"),
    PersonaAction(tokens=["audra", "stevie"], text="adjusting a silk scarf caught in the wind"),
    PersonaAction(tokens=["audra", "ariel"], text="stepping off a tram with a bouquet of flowers"),
    PersonaAction(tokens=["audra"], text="spinning through a splash of fountain mist"),
    PersonaAction(tokens=["andie", "audra", "stevie"], text="tossing her hair as she steps into a spotlight"),
]


class PersonaActionResolver:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / "config" / "persona_actions.yaml"
        self.actions = self._load_actions()

    def _load_actions(self) -> List[PersonaAction]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    payload = yaml.safe_load(fh) or {}
                values = payload.get("actions") if isinstance(payload, dict) else None
                if isinstance(values, list):
                    entries: List[PersonaAction] = []
                    for item in values:
                        if not isinstance(item, dict):
                            continue
                        text = item.get("text")
                        tokens = item.get("tokens") or []
                        if not text:
                            continue
                        if not isinstance(tokens, list):
                            tokens = []
                        entries.append(
                            PersonaAction(
                                tokens=[str(token) for token in tokens if token is not None],
                                text=str(text),
                            )
                        )
                    if entries:
                        return entries
            except Exception:
                pass
        return DEFAULT_ACTIONS.copy()

    def pick(self, persona_tokens: Optional[Sequence[str]] = None) -> Optional[str]:
        tokens_lower = {token.lower() for token in persona_tokens or [] if token}
        candidates = self.actions
        if tokens_lower:
            filtered = [
                entry
                for entry in self.actions
                if tokens_lower & {token.lower() for token in entry.tokens}
            ]
            if filtered:
                candidates = filtered
        if not candidates:
            return None
        return random.choice(candidates).text

    def pick_any(self) -> Optional[str]:
        return self.pick()
