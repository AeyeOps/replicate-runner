from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _user_config_dir() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".config"


PACKAGE_SCOPE = "package"
WORKSPACE_SCOPE = "workspace"
USER_SCOPE = "user"
PROFILE_SCOPES = [PACKAGE_SCOPE, WORKSPACE_SCOPE, USER_SCOPE]


@dataclass
class ProfileSource:
    scope: str
    path: Path


@dataclass
class ProfileLayer:
    scope: str
    path: Path
    data: Dict[str, Any]


@dataclass
class ResolvedProfile:
    name: str
    data: Dict[str, Any]
    sources: List[ProfileSource]


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key not in result:
            result[key] = deepcopy(value)
            continue

        existing = result[key]
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        elif isinstance(existing, list) and isinstance(value, list):
            result[key] = deepcopy(value)
        else:
            result[key] = deepcopy(value)
    return result


class ProfileManager:
    def __init__(self, cwd: Optional[Path] = None):
        self.cwd = cwd or Path.cwd()
        self.layers = self._load_layers()

    @property
    def package_path(self) -> Path:
        return Path(__file__).parent / "config" / "profiles.yaml"

    @property
    def workspace_path(self) -> Path:
        return self.cwd / "config" / "profiles.yaml"

    @property
    def user_path(self) -> Path:
        return _user_config_dir() / "replicate-runner" / "profiles.yaml"

    def _load_layers(self) -> Dict[str, ProfileLayer]:
        paths = {
            PACKAGE_SCOPE: self.package_path,
            WORKSPACE_SCOPE: self.workspace_path,
            USER_SCOPE: self.user_path,
        }
        layers: Dict[str, ProfileLayer] = {}
        for scope, path in paths.items():
            data = _read_yaml(path)
            layers[scope] = ProfileLayer(scope=scope, path=path, data=data)
        return layers

    def reload(self):
        self.layers = self._load_layers()

    def available_profiles(self) -> List[str]:
        names: set[str] = set()
        for layer in self.layers.values():
            profiles = layer.data.get("profiles") if isinstance(layer.data, dict) else {}
            if isinstance(profiles, dict):
                names.update(name for name, value in profiles.items() if isinstance(value, dict))
        return sorted(names)

    def resolve_profile(self, name: str) -> ResolvedProfile:
        merged: Dict[str, Any] = {}
        sources: List[ProfileSource] = []
        found = False
        for scope in PROFILE_SCOPES:
            layer = self.layers.get(scope)
            if not layer:
                continue
            profiles = layer.data.get("profiles") if isinstance(layer.data, dict) else {}
            if not isinstance(profiles, dict):
                continue
            payload = profiles.get(name)
            if not isinstance(payload, dict):
                continue
            merged = _deep_merge(merged, payload)
            sources.append(ProfileSource(scope=scope, path=layer.path))
            found = True
        if not found:
            raise KeyError(f"Profile '{name}' not found in any scope")
        return ResolvedProfile(name=name, data=merged, sources=sources)

    def _layer_for_scope(self, scope: str) -> ProfileLayer:
        if scope not in self.layers:
            valid = ", ".join(PROFILE_SCOPES)
            raise ValueError(f"Unknown scope '{scope}'. Valid scopes: {valid}")
        return self.layers[scope]

    def save_profile(
        self,
        name: str,
        updates: Dict[str, Any],
        scope: str = USER_SCOPE,
        unset_paths: Optional[List[str]] = None,
    ) -> Path:
        layer = self._layer_for_scope(scope)
        data = deepcopy(layer.data)
        profiles = data.setdefault("profiles", {})
        if not isinstance(profiles, dict):
            profiles = {}
            data["profiles"] = profiles

        existing = profiles.get(name)
        if isinstance(existing, dict):
            new_payload = _deep_merge(existing, updates)
        else:
            new_payload = deepcopy(updates)

        for path_expr in unset_paths or []:
            if not path_expr:
                continue
            self._apply_unset(new_payload, path_expr.split("."))

        profiles[name] = new_payload
        self._write_layer(layer.path, data)
        layer.data = data
        return layer.path

    def delete_profile(self, name: str, scope: str = USER_SCOPE) -> bool:
        layer = self._layer_for_scope(scope)
        data = deepcopy(layer.data)
        profiles = data.get("profiles") if isinstance(data, dict) else None
        if not isinstance(profiles, dict) or name not in profiles:
            return False
        profiles.pop(name, None)
        if not profiles:
            data.pop("profiles", None)
        self._write_layer(layer.path, data)
        layer.data = data
        return True

    def _write_layer(self, path: Path, payload: Dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
        except PermissionError as exc:
            raise PermissionError(
                f"Unable to write profiles to {path}. Check permissions or choose another scope."
            ) from exc

    def _apply_unset(self, data: Dict[str, Any], path_segments: List[str]):
        if not path_segments:
            return
        key = path_segments[0]
        if len(path_segments) == 1:
            data.pop(key, None)
            return
        child = data.get(key)
        if not isinstance(child, dict):
            return
        self._apply_unset(child, path_segments[1:])
        if isinstance(child, dict) and not child:
            data.pop(key, None)


def build_profile_updates(
    description: Optional[str] = None,
    model: Optional[str] = None,
    version: Optional[str] = None,
    lora: Optional[str] = None,
    trigger: Optional[str] = None,
    prompt_template: Optional[str] = None,
    subject: Optional[str] = None,
    persona_tokens: Optional[List[str]] = None,
    persona_enabled: Optional[bool] = None,
    defaults_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if description is not None:
        payload["description"] = description
    if model is not None:
        payload["model"] = model
    if version is not None:
        payload["version"] = version
    if lora is not None:
        payload["lora"] = lora
    if trigger is not None:
        payload["trigger"] = trigger
    if prompt_template is not None:
        payload["prompt_template"] = prompt_template

    defaults: Dict[str, Any] = {}
    if subject is not None:
        defaults["subject"] = subject
    if persona_tokens is not None:
        defaults["persona_tokens"] = persona_tokens
    if persona_enabled is not None:
        defaults["persona_enabled"] = persona_enabled
    if defaults_params:
        params = defaults.setdefault("params", {})
        params.update(defaults_params)

    if defaults:
        payload["defaults"] = defaults
    return payload
