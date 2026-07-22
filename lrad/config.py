"""YAML config loading, dotted CLI overrides, and JSON-safe conversion.

Shared by every runner script — these used to live in
``scripts/run_celeba.py``, which forced the other scripts to import a
sibling script through a ``sys.path`` hack just to parse a config.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_override(cfg: dict, key_path: str, value: str) -> None:
    node = cfg
    keys = key_path.split(".")
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    # YAML-style lists, e.g. model.channels=[32,64,128,256,256]
    if value.startswith("[") and value.endswith("]"):
        try:
            node[keys[-1]] = yaml.safe_load(value)
            return
        except yaml.YAMLError:
            pass
    for cast in (int, float):
        try:
            node[keys[-1]] = cast(value)
            return
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        node[keys[-1]] = value.lower() == "true"
        return
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Return a deep copy of ``cfg`` with ``key.path=value`` items applied."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        k, v = item.split("=", 1)
        _apply_override(cfg, k.strip(), v.strip())
    return cfg


def to_jsonable(obj):
    """Recursively convert numpy scalars/containers for ``json.dump``.

    Raw arrays become ``None`` on purpose: the only arrays reaching a
    summary are ROC curves and score vectors, far too large for a JSON
    meant to be eyeballed.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)
