"""YAML configuration loading with `base:` inheritance and dotted overrides."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("base", None)
    if base:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        data = deep_merge(_load_yaml(os.path.normpath(base_path)), data)
    return data


def _coerce(value: str) -> Any:
    return yaml.safe_load(value)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``key.sub=value`` overrides (value parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.subkey=value")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    cfg = _load_yaml(path)
    cfg = apply_overrides(cfg, overrides)
    cfg["_config_path"] = os.path.abspath(path)
    mode = cfg.get("mode", "baseline")
    if mode not in ("baseline", "full"):
        raise ValueError(f"mode must be 'baseline' or 'full', got {mode!r}")
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def is_full(cfg: dict) -> bool:
    return cfg.get("mode", "baseline") == "full"


def semantic_enabled(cfg: dict) -> bool:
    flag = get(cfg, "blocking.methods.semantic.enabled", "auto")
    if flag == "auto":
        return is_full(cfg)
    return bool(flag)


def model_dir(cfg: dict) -> str:
    return os.path.join(get(cfg, "paths.models_dir", "models"), cfg.get("run_name", "default"))


def output_dir(cfg: dict) -> str:
    return os.path.join(get(cfg, "paths.outputs_dir", "outputs"), cfg.get("run_name", "default"))
