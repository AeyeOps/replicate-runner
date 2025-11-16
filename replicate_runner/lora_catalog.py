from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from replicate_runner.config_loader import ConfigLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LoraEntry:
    """Normalized representation of a single LoRA definition."""

    key: str
    name: str
    repo_id: str
    lora_weights: str
    trigger: str
    description: str | None = None
    default_prompt: str | None = None
    base_images: List[Path] = field(default_factory=list)


@dataclass
class LoraCollection:
    """Metadata describing a logical collection of LoRAs."""

    key: str
    description: str | None = None
    default_prompt: str | None = None
    lora_keys: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class LoraCatalog:
    """Wrapper around the loaded LoRA catalog configuration."""

    loader: ConfigLoader
    loras: Dict[str, LoraEntry]
    collections: Dict[str, LoraCollection]


def _resolve_base_images(values: Iterable[str] | None) -> List[Path]:
    resolved: List[Path] = []
    if not values:
        return resolved

    for raw in values:
        if raw is None:
            continue
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        resolved.append(path)
    return resolved


def _normalize_lora_entry(key: str, payload: dict) -> LoraEntry | None:
    repo_id = payload.get("hf_repo") or payload.get("repo_id")
    if not repo_id:
        return None

    weights = (
        payload.get("lora_weights")
        or payload.get("weights_url")
        or payload.get("repo_url")
    )
    if not weights:
        weights = f"huggingface.co/{repo_id}"

    return LoraEntry(
        key=key,
        name=payload.get("name", key),
        repo_id=repo_id,
        lora_weights=weights,
        trigger=payload.get("trigger") or payload.get("token") or "TOK",
        description=payload.get("description"),
        default_prompt=payload.get("default_prompt"),
        base_images=_resolve_base_images(payload.get("base_images")),
    )


def _normalize_collection(key: str, payload: dict) -> LoraCollection:
    lora_keys = payload.get("loras") or []
    # Ensure entries are strings even if user provided dictionaries later.
    normalized_keys = []
    for item in lora_keys:
        if isinstance(item, dict):
            entry_key = item.get("key")
            if entry_key:
                normalized_keys.append(entry_key)
        else:
            normalized_keys.append(str(item))

    return LoraCollection(
        key=key,
        description=payload.get("description"),
        default_prompt=payload.get("default_prompt"),
        lora_keys=normalized_keys,
        tags=list(payload.get("tags", [])),
    )


def load_lora_catalog() -> LoraCatalog:
    """Load the LoRA catalog plus collection metadata from config files."""

    loader = ConfigLoader()
    config = loader.config or {}
    loras: Dict[str, LoraEntry] = {}

    for key, payload in (config.get("loras") or {}).items():
        if not isinstance(payload, dict):
            continue
        entry = _normalize_lora_entry(key, payload)
        if entry:
            loras[key] = entry

    collections: Dict[str, LoraCollection] = {}
    for key, payload in (config.get("lora_collections") or {}).items():
        if not isinstance(payload, dict):
            continue
        collections[key] = _normalize_collection(key, payload)

    return LoraCatalog(loader=loader, loras=loras, collections=collections)


def resolve_collection(
    catalog: LoraCatalog, collection_name: str
) -> Tuple[LoraCollection, List[LoraEntry]]:
    """Return collection metadata and ordered LoRA entries."""

    if collection_name not in catalog.collections:
        available = ", ".join(sorted(catalog.collections)) or "<none>"
        raise KeyError(
            f"Unknown LoRA collection '{collection_name}'. Available: {available}"
        )

    collection = catalog.collections[collection_name]
    entries: List[LoraEntry] = []
    for key in collection.lora_keys:
        entry = catalog.loras.get(key)
        if entry:
            entries.append(entry)
    return collection, entries


def gather_base_images(entries: Iterable[LoraEntry]) -> List[Path]:
    images: List[Path] = []
    for entry in entries:
        images.extend(entry.base_images)
    return images
