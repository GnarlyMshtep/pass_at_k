"""Config resolver: merge base JSON5 + override JSON5 → Hydra override list.

The base JSON5 contains all the Hydra overrides you always want (the "template").
The override JSON5 contains run-specific changes layered on top.

Keys prefixed with "+" in the JSON5 dicts become "+key=value" in Hydra
(required for fields that don't exist in verl's ppo_trainer.yaml defaults).

Example base_config.json5:
    {
        "data": {"train_batch_size": 32},
        "trainer": {
            "save_freq": 40,
            "+rollout_dump_freq": 1,   // new field → +trainer.rollout_dump_freq=1
        },
    }
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pyjson5


def resolve_config(
    base_config_path: str,
    overrides_path: Optional[str],
    extra_hydra_overrides: list[str],
) -> tuple[dict[str, Any], list[str]]:
    """Merge configs and return the merged dict plus the full Hydra override list.

    Args:
        base_config_path: Path to base JSON5 (all "always-on" overrides).
        overrides_path: Optional path to run-specific JSON5 overrides.
        extra_hydra_overrides: Raw Hydra override strings to append verbatim.

    Returns:
        (merged_config, hydra_overrides) where merged_config is the raw nested
        dict (useful for extracting values like trainer.project_name) and
        hydra_overrides is the list of strings to pass to verl.trainer.main_ppo.
    """
    base = _load_json5(path=base_config_path)

    if overrides_path is not None:
        overrides = _load_json5(path=overrides_path)
        merged = _deep_merge(base=base, overrides=overrides)
    else:
        merged = dict(base)

    merged = _expand_env_vars(d=merged)
    hydra_overrides = _flatten_to_hydra_overrides(d=merged)

    if extra_hydra_overrides:
        resolved_keys = {entry.split("=", 1)[0].lstrip("+") for entry in hydra_overrides}
        _validate_extra_overrides(
            extra_overrides=extra_hydra_overrides,
            resolved_keys=resolved_keys,
        )
        hydra_overrides.extend(extra_hydra_overrides)

    return merged, hydra_overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json5(path: str) -> dict[str, Any]:
    with open(path) as f:
        result = pyjson5.load(f)
    if not isinstance(result, dict):
        raise ValueError(f"Config file must be a JSON5 object (dict), got {type(result)}: {path}")
    return result


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into base. Override wins on conflicts.

    Both bare keys and +-prefixed keys are merged by their bare name so that
    an override can change an existing key from regular to new (or vice versa).
    """
    result: dict[str, Any] = {}

    # Index base by bare key (strip leading +)
    for raw_k, v in base.items():
        result[raw_k] = v  # keep raw key (with + if present)

    for raw_k, v in overrides.items():
        bare_k = raw_k.lstrip("+")
        # Find existing entry with same bare key (with or without +)
        existing_raw = _find_raw_key(d=result, bare_key=bare_k)

        if existing_raw is not None:
            existing_v = result[existing_raw]
            if (
                isinstance(existing_v, dict)
                and isinstance(v, dict)
            ):
                # Both are dicts — recurse, using override's raw key (preserves +)
                del result[existing_raw]
                result[raw_k] = _deep_merge(base=existing_v, overrides=v)
            else:
                del result[existing_raw]
                result[raw_k] = v
        else:
            result[raw_k] = v

    return result


def _find_raw_key(d: dict[str, Any], bare_key: str) -> Optional[str]:
    """Return the raw key in d whose bare (no-+) form matches bare_key, or None."""
    for raw_k in d:
        if raw_k.lstrip("+") == bare_key:
            return raw_k
    return None


def _expand_env_vars(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively expand $ENV_VAR in string values (not in keys)."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[k] = os.path.expandvars(v)
        elif isinstance(v, dict):
            result[k] = _expand_env_vars(d=v)
        elif isinstance(v, list):
            result[k] = [
                os.path.expandvars(x) if isinstance(x, str) else x
                for x in v
            ]
        else:
            result[k] = v
    return result


def _flatten_to_hydra_overrides(
    d: dict[str, Any],
    prefix: str = "",
    force_new: bool = False,
) -> list[str]:
    """Recursively flatten a nested dict into Hydra override strings.

    A key starting with "+" is treated as a new Hydra field (+key=value).
    If force_new is True, all descendants are treated as new fields.
    """
    overrides: list[str] = []
    for raw_key, value in d.items():
        is_new = force_new or raw_key.startswith("+")
        bare_key = raw_key.lstrip("+")
        full_key = f"{prefix}.{bare_key}" if prefix else bare_key
        hydra_key = f"+{full_key}" if is_new else full_key

        if isinstance(value, dict):
            overrides.extend(
                _flatten_to_hydra_overrides(
                    d=value,
                    prefix=full_key,
                    force_new=is_new,
                )
            )
        elif isinstance(value, list):
            items_str = ",".join(_format_scalar(v=v) for v in value)
            overrides.append(f"{hydra_key}=[{items_str}]")
        else:
            overrides.append(f"{hydra_key}={_format_scalar(v=value)}")

    return overrides


def _validate_extra_overrides(
    extra_overrides: list[str],
    resolved_keys: set[str],
) -> None:
    """Validate --extra-overrides against the resolved config keys.

    Rules:
        1. Each override must be key=value (contain '=').
        2. '+' prefix not allowed — new fields belong in JSON5 configs.
        3. Key must exist in the resolved config (from base + overrides JSON5).

    Raises ValueError listing all issues.
    """
    errors: list[str] = []
    for entry in extra_overrides:
        if "=" not in entry:
            errors.append(f"Not a key=value pair: {entry!r}")
            continue
        key = entry.split("=", 1)[0]
        if key.startswith("+"):
            errors.append(
                f"'+' prefix not allowed in --extra-overrides (new fields belong "
                f"in JSON5 configs): {entry!r}"
            )
            continue
        if key not in resolved_keys:
            errors.append(
                f"Key {key!r} not found in resolved config. "
                f"Typo? Available keys with same prefix: "
                f"{_suggest_keys(key=key, resolved_keys=resolved_keys)}"
            )
    if errors:
        error_list = "\n  ".join(errors)
        raise ValueError(
            f"--extra-overrides validation failed:\n  {error_list}"
        )


def _suggest_keys(key: str, resolved_keys: set[str], max_suggestions: int = 5) -> str:
    """Suggest similar keys from the resolved set for error messages."""
    prefix = key.split(".")[0]
    matches = sorted(k for k in resolved_keys if k.startswith(prefix))[:max_suggestions]
    if matches:
        return ", ".join(matches)
    return "(none)"


def _format_scalar(v: Any) -> str:
    """Format a scalar value for a Hydra override string."""
    if isinstance(v, bool):
        # Hydra accepts True/False (Python casing)
        return str(v)
    if isinstance(v, str):
        # Quote strings containing Hydra-special characters
        if any(c in v for c in (" ", ",", "[", "]", "{", "}", "=")):
            escaped = v.replace('"', '\\"')
            return f'"{escaped}"'
        return v
    if v is None:
        return "null"
    return str(v)
