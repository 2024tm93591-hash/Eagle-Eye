"""Configuration loading helpers."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str | os.PathLike | None) -> Path | None:
    """Resolve a path relative to the project root (absolute paths are kept)."""
    if p is None or p == "":
        return None
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = resolve_path(path or "config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return deep_update(cfg, overrides or {})
