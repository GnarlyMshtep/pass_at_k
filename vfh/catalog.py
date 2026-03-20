"""Catalog script for tracking notable VFH run checkpoints.

Usage:
    python -m vfh.catalog <run_dir_or_wandb_id>
    python -m vfh.catalog logs/VerlRun/03/02/hidden_wmonitor_backdoor_qwen3_8b_remove_21_39_186oohgn
    python -m vfh.catalog 186oohgn

The catalog file location is controlled by the RUN_CATALOG_PATH env var,
defaulting to ../catalog.json (sibling to the repo root).
"""

from __future__ import annotations

import json
import os
import re
import readline  # noqa: F401 — enables line editing (arrow keys, etc.) in input()
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow running as `python vfh/catalog.py` (not just `python -m vfh.catalog`)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dacite
import pyjson5
import tyro

from vfh.catalog_types import Catalog, CatalogEntry, CatalogTag


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

class _C:
    """ANSI color codes for terminal output."""
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    RESET = "\033[0m"


def _colored(text: str, *codes: str) -> str:
    return "".join(codes) + text + _C.RESET


class _UserCancelled(Exception):
    """Raised when user types 'esc' or Ctrl+C to cancel an interactive prompt."""
    pass


def _input_or_esc(prompt: str) -> str:
    """Like input(), but returns raises _UserCancelled if user types 'esc' or hits Ctrl+C."""
    try:
        value = input(prompt)
    except (KeyboardInterrupt, EOFError):
        print()  # newline after ^C
        raise _UserCancelled()
    if value.strip().lower() == "esc":
        raise _UserCancelled()
    return value


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CATALOG_PATH = _REPO_ROOT.parent / "catalog.json"
_LOGS_ROOT = _REPO_ROOT / "logs" / "VerlRun"


def _get_catalog_path() -> Path:
    return Path(os.environ.get("RUN_CATALOG_PATH", str(_DEFAULT_CATALOG_PATH)))


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------


@dataclass
class CatalogConfig:
    """Catalog a notable run checkpoint."""
    path: str  # run dir path OR wandb ID (8-char string)


# ---------------------------------------------------------------------------
# Catalog I/O
# ---------------------------------------------------------------------------


def load_catalog(catalog_path: Path) -> Catalog:
    """Load catalog from JSON file, or return empty catalog if not found."""
    if not catalog_path.exists():
        return Catalog()
    with open(catalog_path) as f:
        data = json.load(f)
    return dacite.from_dict(
        data_class=Catalog,
        data=data,
        config=dacite.Config(cast=[tuple]),
    )


def save_catalog(catalog: Catalog, catalog_path: Path) -> None:
    """Save catalog to JSON file."""
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(catalog_path, "w") as f:
        json.dump(
            {"entries": [e.__dict__ for e in catalog.entries],
             "tags": [t.__dict__ for t in catalog.tags]},
            f,
            indent=2,
            default=_serialize,
        )
    print(f"  Saved catalog to: {catalog_path}")


# ---------------------------------------------------------------------------
# Extractor base class
# ---------------------------------------------------------------------------


_DEFAULT_WANDB_ENTITY = "matan-shtepel-carnegie-mellon-university"


class RunExtractor(ABC):
    """Base class for extracting catalog metadata from a run directory.

    Subclasses must implement can_extract, extract, and wandb_url.
    """

    @abstractmethod
    def can_extract(self, path: Path) -> bool:
        """Return True if this extractor can handle the given run dir."""
        ...

    @abstractmethod
    def extract(self, path: Path) -> CatalogEntry:
        """Extract metadata from the run directory.

        Should fail loudly if required fields are missing.
        The 'description' and 'tags' fields should be left empty
        (filled in interactively later).
        Must populate wandb_url (construct from project/entity if not in metadata).
        """
        ...


# ---------------------------------------------------------------------------
# VFH Extractor
# ---------------------------------------------------------------------------


def _parse_hydra_overrides(overrides: list[str]) -> dict[str, str]:
    """Parse resolved_hydra_overrides list into a flat dict.

    Handles both 'key=value' and '+key=value' forms.
    """
    result: dict[str, str] = {}
    for override in overrides:
        # Strip leading + if present
        clean = override.lstrip("+")
        eq_idx = clean.find("=")
        if eq_idx == -1:
            continue
        key = clean[:eq_idx]
        value = clean[eq_idx + 1:]
        result[key] = value
    return result


def _extract_reward_config(overrides_dict: dict[str, str]) -> dict[str, Any]:
    """Extract all custom_reward_function.reward_kwargs.reward_config.* keys."""
    prefix = "custom_reward_function.reward_kwargs.reward_config."
    reward_config: dict[str, Any] = {}
    for key, value in overrides_dict.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):]
            reward_config[config_key] = value
    return reward_config


def _find_checkpoint_range(run_dir: Path) -> tuple[int, int] | None:
    """Find (min_step, max_step) of global_step_N dirs in checkpoints/.

    Includes cleaned checkpoints (they still have the directory).
    Returns None if no checkpoints found.
    """
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        return None

    steps: list[int] = []
    for d in checkpoints_dir.iterdir():
        if d.is_dir() and d.name.startswith("global_step_"):
            match = re.match(r"global_step_(\d+)", d.name)
            if match:
                steps.append(int(match.group(1)))

    # Also check for .dvc files (checkpoint may have been deleted after backup)
    for f in checkpoints_dir.iterdir():
        if f.name.endswith(".dvc") and f.name.startswith("global_step_"):
            match = re.match(r"global_step_(\d+)\.dvc", f.name)
            if match:
                step = int(match.group(1))
                if step not in steps:
                    steps.append(step)

    if not steps:
        return None

    return (min(steps), max(steps))


def _find_rollout_range(run_dir: Path) -> tuple[int, int] | None:
    """Find (min_epoch, max_epoch) from {N}.jsonl files in rollouts/train/.

    Also checks rollouts/train.dvc if the dir was backed up and deleted.
    Returns None if no rollouts found.
    """
    rollouts_train = run_dir / "rollouts" / "train"
    epochs: list[int] = []

    if rollouts_train.is_dir():
        for f in rollouts_train.iterdir():
            if f.suffix == ".jsonl" and f.stem.isdigit():
                epochs.append(int(f.stem))

    # Also check for .dvc file (rollouts may have been deleted after backup)
    train_dvc = run_dir / "rollouts" / "train.dvc"
    if not epochs and not train_dvc.exists():
        return None

    if not epochs:
        # .dvc exists but dir is empty/gone — we know rollouts existed but can't determine range
        return None

    return (min(epochs), max(epochs))


class VFHExtractor(RunExtractor):
    """Extract catalog metadata from a VFH (Verl For Humans) run directory."""

    def can_extract(self, path: Path) -> bool:
        return (path / "run_metadata.json5").exists()

    def extract(self, path: Path) -> CatalogEntry:
        metadata_file = path / "run_metadata.json5"
        if not metadata_file.exists():
            raise ValueError(f"No run_metadata.json5 in {path}")

        with open(metadata_file) as f:
            meta = pyjson5.load(f)

        run_id: str = meta["run_id"]
        overrides: list[str] = meta["resolved_hydra_overrides"]
        overrides_dict = _parse_hydra_overrides(overrides=overrides)

        # W&B URL: use from metadata, or construct from project + entity
        wandb_url: str = meta.get("wandb_url", "")
        if not wandb_url:
            project_name = overrides_dict.get("trainer.project_name", "")
            wandb_entity = os.environ.get("WANDB_ENTITY", _DEFAULT_WANDB_ENTITY)
            if project_name:
                wandb_url = f"https://wandb.ai/{wandb_entity}/{project_name}/runs/{run_id}"
            else:
                print(f"  {_colored('Warning:', _C.YELLOW)} Could not construct W&B URL — trainer.project_name not found in overrides")

        # Base model
        base_model = overrides_dict.get("actor_rollout_ref.model.path")
        if base_model is None:
            raise ValueError(f"Cannot find actor_rollout_ref.model.path in overrides for {run_id}")

        # Train dataset
        train_dataset = overrides_dict.get("data.train_files")
        if train_dataset is None:
            raise ValueError(f"Cannot find data.train_files in overrides for {run_id}")

        # Reward config (may be empty — that's OK)
        reward_config = _extract_reward_config(overrides_dict=overrides_dict)
        # Also include the reward function name and path for context
        reward_name = overrides_dict.get("custom_reward_function.name", "")
        reward_path = overrides_dict.get("custom_reward_function.path", "")
        if reward_name:
            reward_config["_name"] = reward_name
        if reward_path:
            reward_config["_path"] = reward_path

        # Checkpoint range
        checkpoint_range = _find_checkpoint_range(run_dir=path)

        # Rollout range
        rollout_range = _find_rollout_range(run_dir=path)

        # Lineage
        origin = meta.get("origin", {})
        parent_run_id = origin.get("parent_run_id")
        child_run_ids: list[str] = meta.get("child_run_ids", [])

        follows: list[str] = [parent_run_id] if parent_run_id else []
        preceded_by: list[str] = list(child_run_ids)

        return CatalogEntry(
            run_id=run_id,
            run_dir=str(path),
            description="",  # filled in interactively
            base_model=base_model,
            reward_config=reward_config,
            train_dataset=train_dataset,
            checkpoint_range=checkpoint_range,
            rollout_range=rollout_range,
            tags=[],  # filled in interactively
            wandb_url=wandb_url,
            cataloged_at=datetime.now(tz=timezone.utc).isoformat(),
            source_framework="vfh",
            follows=follows,
            preceded_by=preceded_by,
        )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_EXTRACTORS: list[RunExtractor] = [VFHExtractor()]


def resolve_run_dir(path_or_id: str) -> Path:
    """Resolve a run dir path or wandb ID to an absolute Path."""
    # If it looks like a path (contains / or is an existing dir)
    candidate = Path(path_or_id)
    if candidate.is_dir():
        return candidate.resolve()

    # Try as wandb ID — glob for matching run dirs
    matches = list(_LOGS_ROOT.glob(f"*/*/*_{path_or_id}"))
    if len(matches) == 1:
        return matches[0].resolve()
    elif len(matches) == 0:
        # Also try as a substring match
        matches = list(_LOGS_ROOT.glob(f"*/*/*{path_or_id}*"))
        if len(matches) == 1:
            return matches[0].resolve()
        raise ValueError(
            f"No run directory found matching ID '{path_or_id}' under {_LOGS_ROOT}. "
            f"Found {len(matches)} matches."
        )
    else:
        raise ValueError(
            f"Ambiguous ID '{path_or_id}': found {len(matches)} matches:\n"
            + "\n".join(f"  {m}" for m in matches)
        )


def find_extractor(run_dir: Path) -> RunExtractor:
    """Find the appropriate extractor for a run directory."""
    for extractor in _EXTRACTORS:
        if extractor.can_extract(path=run_dir):
            return extractor
    raise ValueError(
        f"No extractor can handle {run_dir}. "
        f"Expected run_metadata.json5 for VFH runs."
    )


# ---------------------------------------------------------------------------
# Interactive tag selection
# ---------------------------------------------------------------------------


def _fuzzy_search_tags(query: str, tags: list[CatalogTag], limit: int = 10) -> list[tuple[CatalogTag, float]]:
    """Fuzzy search tags by name. Returns (tag, score) pairs."""
    if not query or not tags:
        return [(t, 100.0) for t in tags[:limit]]

    try:
        from rapidfuzz import fuzz
        scored = [(t, fuzz.partial_ratio(query.lower(), t.name.lower())) for t in tags]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(t, s) for t, s in scored[:limit] if s > 40]
    except ImportError:
        # Fallback: substring match
        query_lower = query.lower()
        return [(t, 100.0) for t in tags if query_lower in t.name.lower()][:limit]


def interactive_tag_selection(catalog: Catalog) -> list[str] | None:
    """Interactive tag selection with fuzzy search. Returns None if cancelled."""
    selected_tags: list[str] = []

    print(f"\n{_colored('--- Tag Selection ---', _C.BOLD, _C.CYAN)}")
    print(f"  Type to fuzzy-search existing tags.")
    print(f"  Enter comma-separated indices to select, e.g.: {_colored('1,3', _C.YELLOW)}")
    print(f"  Type {_colored('+tag_name', _C.GREEN)} to create a new tag.")
    print(f"  Type {_colored('esc', _C.RED)} or {_colored('Ctrl+C', _C.RED)} to cancel and go back.")
    print(f"  Press {_colored('Enter', _C.DIM)} with no input when done.\n")

    while True:
        if catalog.tags:
            print(f"  {_colored('Existing tags:', _C.BOLD)}")
            for i, tag in enumerate(catalog.tags):
                if tag.name in selected_tags:
                    marker = _colored(" [selected]", _C.GREEN, _C.BOLD)
                else:
                    marker = ""
                print(f"    {_colored(str(i), _C.YELLOW)}: {_colored(tag.name, _C.MAGENTA)} — {_colored(tag.description, _C.DIM)}{marker}")

        try:
            query = _input_or_esc("\n  Search/select/+new (enter to finish, esc to cancel): ").strip()
        except _UserCancelled:
            print(f"  {_colored('Tag selection cancelled.', _C.YELLOW)}")
            return None
        if not query:
            break

        # Create new tag
        if query.startswith("+"):
            new_name = query[1:].strip()
            if not new_name:
                print("  Tag name cannot be empty.")
                continue
            # Check if already exists
            existing = [t for t in catalog.tags if t.name.lower() == new_name.lower()]
            if existing:
                print(f"  Tag '{new_name}' already exists.")
                if new_name not in selected_tags:
                    selected_tags.append(existing[0].name)
                    print(f"  Selected: {existing[0].name}")
                continue
            try:
                desc = _input_or_esc(f"  Description for '{new_name}' (esc to cancel): ").strip()
            except _UserCancelled:
                print(f"  {_colored('Tag creation cancelled.', _C.YELLOW)}")
                continue
            new_tag = CatalogTag(name=new_name, description=desc)
            catalog.tags.append(new_tag)
            selected_tags.append(new_name)
            print(f"  Created and selected: {new_name}")
            continue

        # Try as comma-separated indices
        if re.match(r"^[\d,\s]+$", query):
            indices = [int(x.strip()) for x in query.split(",") if x.strip().isdigit()]
            for idx in indices:
                if 0 <= idx < len(catalog.tags):
                    tag_name = catalog.tags[idx].name
                    if tag_name not in selected_tags:
                        selected_tags.append(tag_name)
                        print(f"  Selected: {tag_name}")
                    else:
                        selected_tags.remove(tag_name)
                        print(f"  Deselected: {tag_name}")
                else:
                    print(f"  Invalid index: {idx}")
            continue

        # Fuzzy search
        results = _fuzzy_search_tags(query=query, tags=catalog.tags)
        if results:
            print(f"  Search results for '{_colored(query, _C.YELLOW)}':")
            for tag, score in results:
                if tag.name in selected_tags:
                    marker = _colored(" [selected]", _C.GREEN, _C.BOLD)
                else:
                    marker = ""
                idx = catalog.tags.index(tag)
                print(f"    {_colored(str(idx), _C.YELLOW)}: {_colored(tag.name, _C.MAGENTA)} ({score:.0f}%) — {_colored(tag.description, _C.DIM)}{marker}")
        else:
            print(f"  No tags matching '{query}'. Use {_colored(f'+{query}', _C.GREEN)} to create.")

    return selected_tags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _copy_wandb_url(entry: CatalogEntry) -> None:
    """Print wandb URL and try to copy to clipboard."""
    if not entry.wandb_url:
        print(f"  {_colored('No W&B URL available.', _C.RED)}")
        return
    print(f"\n  {_colored(entry.wandb_url, _C.BLUE, _C.BOLD)}")
    # Try clipboard copy (works on macOS via pbcopy, Linux via xclip/xsel)
    try:
        import subprocess as _sp
        # Try pbcopy (macOS), then xclip, then xsel
        for cmd in [["pbcopy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]:
            try:
                _sp.run(cmd, input=entry.wandb_url.encode(), check=True, timeout=2)
                print(f"  {_colored('Copied to clipboard.', _C.GREEN)}")
                return
            except (FileNotFoundError, _sp.CalledProcessError, _sp.TimeoutExpired):
                continue
        print(f"  {_colored('(copy manually — no clipboard tool found)', _C.DIM)}")
    except Exception:
        print(f"  {_colored('(copy manually)', _C.DIM)}")


def _trunc(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _present_entry(entry: CatalogEntry) -> None:
    """Present extracted metadata to the user."""
    # Use short forms for long paths to avoid terminal line-wrap issues with ANSI codes
    short_model = Path(entry.base_model).name if "/" in entry.base_model else entry.base_model
    short_dir = _trunc(entry.run_dir, max_len=70)
    short_dataset = _trunc(entry.train_dataset, max_len=70)

    print(f"\n{_colored('=== Extracted Metadata ===', _C.BOLD, _C.CYAN)}")
    print(f"  {_colored('Run ID:', _C.BOLD)}           {_colored(entry.run_id, _C.YELLOW)}")
    print(f"  {_colored('Run dir:', _C.BOLD)}          {_colored(short_dir, _C.DIM)}")
    print(f"  {_colored('W&B URL:', _C.BOLD)}          {_colored(entry.wandb_url, _C.BLUE)}")
    print(f"  {_colored('Base model:', _C.BOLD)}       {_colored(short_model, _C.GREEN)}")
    print(f"  {_colored('Train dataset:', _C.BOLD)}    {_colored(short_dataset, _C.GREEN)}")
    if entry.checkpoint_range is not None:
        print(f"  {_colored('Ckpt range:', _C.BOLD)}      step {_colored(str(entry.checkpoint_range[0]), _C.YELLOW)} — {_colored(str(entry.checkpoint_range[1]), _C.YELLOW)}")
    else:
        print(f"  {_colored('Ckpt range:', _C.BOLD)}      {_colored('(none)', _C.DIM)}")
    if entry.rollout_range is not None:
        print(f"  {_colored('Rollout range:', _C.BOLD)}   step {_colored(str(entry.rollout_range[0]), _C.YELLOW)} — {_colored(str(entry.rollout_range[1]), _C.YELLOW)}")
    if entry.reward_config:
        print(f"  {_colored('Reward config:', _C.BOLD)}")
        for k, v in entry.reward_config.items():
            print(f"    {_colored(k, _C.MAGENTA)}: {_trunc(str(v), max_len=50)}")
    if entry.follows:
        print(f"  {_colored('Follows:', _C.BOLD)}         {', '.join(_colored(r, _C.CYAN) for r in entry.follows)}")
    if entry.preceded_by:
        print(f"  {_colored('Preceded by:', _C.BOLD)}     {', '.join(_colored(r, _C.CYAN) for r in entry.preceded_by)}")


@dataclass
class _DiscoveredRun:
    """A run discovered during lineage traversal."""
    run_id: str
    relation: str        # "ancestor" or "descendant"
    depth: int           # 1 = immediate parent/child, 2 = grandparent/grandchild, etc.
    entry: CatalogEntry  # extracted metadata
    discovered_via: str  # run_id of the run that linked to this one
    already_cataloged: bool = False


_MAX_LINEAGE_DEPTH = 20


def _traverse_lineage(
    root_entry: CatalogEntry,
    catalog: Catalog,
) -> list[_DiscoveredRun]:
    """BFS traversal of the run lineage DAG in both directions.

    Walks follows (ancestors) and preceded_by (descendants) recursively,
    up to _MAX_LINEAGE_DEPTH. Returns all discovered runs with extracted metadata.
    """
    cataloged_ids = {e.run_id for e in catalog.entries}
    visited: set[str] = {root_entry.run_id}
    discovered: list[_DiscoveredRun] = []

    # BFS queue: (run_id, relation_direction, depth, discovered_via_run_id)
    queue: list[tuple[str, str, int, str]] = []

    # Seed with immediate relatives
    for rid in root_entry.follows:
        queue.append((rid, "ancestor", 1, root_entry.run_id))
    for rid in root_entry.preceded_by:
        queue.append((rid, "descendant", 1, root_entry.run_id))

    while queue:
        rid, relation, depth, via = queue.pop(0)

        if rid in visited:
            continue
        if depth > _MAX_LINEAGE_DEPTH:
            continue
        visited.add(rid)

        # Already cataloged — include in list but mark, and continue traversal
        if rid in cataloged_ids:
            cat_entry = next(e for e in catalog.entries if e.run_id == rid)
            discovered.append(_DiscoveredRun(
                run_id=rid,
                relation=relation,
                depth=depth,
                entry=cat_entry,
                discovered_via=via,
                already_cataloged=True,
            ))
            # Continue traversal in the same direction
            if relation == "ancestor":
                for next_rid in cat_entry.follows:
                    queue.append((next_rid, "ancestor", depth + 1, rid))
            else:
                for next_rid in cat_entry.preceded_by:
                    queue.append((next_rid, "descendant", depth + 1, rid))
            continue

        # Try to resolve and extract metadata
        try:
            rel_dir = resolve_run_dir(path_or_id=rid)
            rel_extractor = find_extractor(run_dir=rel_dir)
            rel_entry = rel_extractor.extract(path=rel_dir)
        except (ValueError, FileNotFoundError):
            # Still try to read metadata for lineage traversal even if extraction fails
            try:
                rel_dir = resolve_run_dir(path_or_id=rid)
                meta_file = rel_dir / "run_metadata.json5"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = pyjson5.load(f)
                    origin = meta.get("origin", {})
                    child_ids: list[str] = meta.get("child_run_ids", [])
                    parent_id = origin.get("parent_run_id")
                    # Continue traversal through this empty run
                    if relation == "ancestor" and parent_id:
                        queue.append((parent_id, "ancestor", depth + 1, rid))
                    elif relation == "descendant":
                        for cid in child_ids:
                            queue.append((cid, "descendant", depth + 1, rid))
            except Exception:
                pass
            continue

        discovered.append(_DiscoveredRun(
            run_id=rid,
            relation=relation,
            depth=depth,
            entry=rel_entry,
            discovered_via=via,
        ))

        # Continue traversal in the same direction
        if relation == "ancestor":
            for next_rid in rel_entry.follows:
                queue.append((next_rid, "ancestor", depth + 1, rid))
        else:
            for next_rid in rel_entry.preceded_by:
                queue.append((next_rid, "descendant", depth + 1, rid))

    return discovered


def _discover_lineage(
    entry: CatalogEntry,
    catalog: Catalog,
    description: str,
    tags: list[str],
) -> None:
    """Traverse the full lineage DAG and offer to catalog discovered runs.

    BFS traversal up to depth 20 in both directions:
    - ancestors: follows → follows → ...
    - descendants: preceded_by → preceded_by → ...

    For each discovered uncataloged run, extracts and presents metadata,
    then offers to catalog under the same tags/description.
    """
    related_ids = entry.follows + entry.preceded_by
    if not related_ids:
        return

    # Update cross-references in already-cataloged immediate relatives
    for existing_entry in catalog.entries:
        if existing_entry.run_id in entry.follows:
            if entry.run_id not in existing_entry.preceded_by:
                existing_entry.preceded_by.append(entry.run_id)
        if existing_entry.run_id in entry.preceded_by:
            if entry.run_id not in existing_entry.follows:
                existing_entry.follows.append(entry.run_id)

    print(f"\n{_colored('--- Run Lineage (traversing DAG) ---', _C.BOLD, _C.CYAN)}")

    # BFS traversal
    discovered = _traverse_lineage(root_entry=entry, catalog=catalog)

    if not discovered:
        print(f"  All relatives are already cataloged or unresolvable.")
        return

    # Interactive loop: pick a run, see metadata, decide to add/edit/skip
    added_ids: set[str] = set()

    def _status_label(d: _DiscoveredRun) -> str:
        if d.run_id in added_ids:
            return _colored(" [just added]", _C.GREEN, _C.BOLD)
        if d.already_cataloged:
            return _colored(" [cataloged]", _C.GREEN)
        return ""

    def _print_list() -> None:
        tags_str = ", ".join(_colored(t, _C.MAGENTA) for t in tags) if tags else _colored("(none)", _C.DIM)
        print(f"\n{_colored('--- Lineage ---', _C.BOLD, _C.CYAN)}")
        print(f"  {len(discovered)} run(s). Tags: {tags_str}")
        print(f"  Select index to view/add/edit, {_colored('Enter', _C.DIM)} to finish, {_colored('esc', _C.RED)} to cancel.\n")
        for i, d in enumerate(discovered):
            ckpt_info = f"steps {d.entry.checkpoint_range[0]}–{d.entry.checkpoint_range[1]}" if d.entry.checkpoint_range else "no ckpts"
            rollout_info = f", rollout {d.entry.rollout_range[0]}–{d.entry.rollout_range[1]}" if d.entry.rollout_range else ""
            depth_label = _colored(f"d{d.depth}", _C.DIM)
            print(f"    {_colored(str(i), _C.YELLOW)}: {_colored(d.relation, _C.BOLD)} {_colored(d.run_id, _C.CYAN)} — {ckpt_info}{rollout_info} [{depth_label}]{_status_label(d)}")

    _print_list()

    while True:
        try:
            choice = _input_or_esc("\n  Index (Enter to finish, esc to cancel): ").strip().lower()
        except _UserCancelled:
            print(f"  {_colored('Cancelled.', _C.YELLOW)}")
            break

        if not choice:
            break

        if not choice.isdigit():
            print(f"  Enter a number 0–{len(discovered) - 1}.")
            continue
        idx = int(choice)
        if idx < 0 or idx >= len(discovered):
            print(f"  Invalid index: {idx}")
            continue

        d = discovered[idx]

        # Show full metadata
        _present_entry(d.entry)
        if d.already_cataloged or d.run_id in added_ids:
            print(f"  {_colored('Description:', _C.BOLD)} {d.entry.description}")
            cur_tags = ", ".join(_colored(t, _C.MAGENTA) for t in d.entry.tags) if d.entry.tags else _colored("(none)", _C.DIM)
            print(f"  {_colored('Tags:', _C.BOLD)}        {cur_tags}")

        # Different prompt depending on status
        if d.already_cataloged or d.run_id in added_ids:
            try:
                action = _input_or_esc(
                    f"\n  {_colored('[e]', _C.YELLOW)}dit desc, "
                    f"{_colored('[w]', _C.BLUE)}andb url, "
                    f"{_colored('[d]', _C.DIM)}on't change (esc to go back): "
                ).strip().lower()
            except _UserCancelled:
                _print_list()
                continue
            if action in ("w", "wandb"):
                _copy_wandb_url(entry=d.entry)
                continue
            if action in ("e", "edit"):
                try:
                    new_desc = _input_or_esc("  New description: ").strip()
                except _UserCancelled:
                    _print_list()
                    continue
                if new_desc:
                    d.entry.description = new_desc
                    print(f"  {_colored('Updated', _C.GREEN, _C.BOLD)} {_colored(d.run_id, _C.CYAN)}")
            _print_list()
            continue

        # Uncataloged run: add or skip
        try:
            action = _input_or_esc(
                f"\n  {_colored('[s]', _C.GREEN)}ame desc, "
                f"{_colored('[n]', _C.YELLOW)}ew desc, "
                f"{_colored('[w]', _C.BLUE)}andb url, "
                f"{_colored('[d]', _C.DIM)}on't add (esc to go back): "
            ).strip().lower()
        except _UserCancelled:
            _print_list()
            continue

        if action in ("w", "wandb"):
            _copy_wandb_url(entry=d.entry)
            continue
        elif action in ("d", "dont", "don't", ""):
            _print_list()
            continue
        elif action in ("s", "same"):
            d.entry.description = description
        elif action in ("n", "new"):
            try:
                new_desc = _input_or_esc("  New description: ").strip()
            except _UserCancelled:
                _print_list()
                continue
            if not new_desc:
                print(f"  Description cannot be empty. Skipped.")
                _print_list()
                continue
            d.entry.description = new_desc
        else:
            print(f"  Unknown option '{action}'. Skipped.")
            _print_list()
            continue

        d.entry.tags = list(tags)
        d.entry.cataloged_at = datetime.now(tz=timezone.utc).isoformat()

        # Update cross-references
        if d.relation == "ancestor":
            if d.discovered_via not in d.entry.preceded_by:
                d.entry.preceded_by.append(d.discovered_via)
        elif d.relation == "descendant":
            if d.discovered_via not in d.entry.follows:
                d.entry.follows.append(d.discovered_via)

        catalog.entries.append(d.entry)
        added_ids.add(d.run_id)
        print(f"  {_colored('Cataloged', _C.GREEN, _C.BOLD)} {_colored(d.run_id, _C.CYAN)}")
        _print_list()

    # Show remaining uncataloged
    still_uncataloged = [d.run_id for d in discovered if d.run_id not in added_ids and not d.already_cataloged]
    if still_uncataloged:
        print(f"\n  {_colored('Still uncataloged:', _C.YELLOW)} {', '.join(_colored(r, _C.CYAN) for r in still_uncataloged)}")
        print(f"  Catalog later: {_colored('python -m vfh.catalog --path <id>', _C.DIM)}")


def _handle_existing(
    catalog: Catalog,
    run_id: str,
    catalog_path: Path,
) -> Optional[str]:
    """Check if run is already cataloged. Returns action: 'edit', 'delete', or None."""
    existing = [e for e in catalog.entries if e.run_id == run_id]
    if not existing:
        return None

    entry = existing[0]
    print(f"\n  {_colored('This run is already cataloged:', _C.YELLOW, _C.BOLD)} {_colored(run_id, _C.YELLOW)}")
    _present_entry(entry)
    print(f"  {_colored('Description:', _C.BOLD)} {entry.description}")
    tags_str = ", ".join(_colored(t, _C.MAGENTA) for t in entry.tags) if entry.tags else _colored("(none)", _C.DIM)
    print(f"  {_colored('Tags:', _C.BOLD)} {tags_str}")

    while True:
        try:
            action = _input_or_esc("\n  [e]dit / [d]elete / [q]uit (esc to cancel)? ").strip().lower()
        except _UserCancelled:
            return "quit"
        if action in ("e", "edit"):
            return "edit"
        elif action in ("d", "delete"):
            catalog.entries.remove(entry)
            save_catalog(catalog=catalog, catalog_path=catalog_path)
            print(f"  Deleted entry for {run_id}.")
            return "delete"
        elif action in ("q", "quit"):
            return "quit"
        else:
            print("  Please enter 'e', 'd', or 'q'.")


def main() -> None:
    config = tyro.cli(CatalogConfig)
    catalog_path = _get_catalog_path()

    # Step 1: Resolve path
    try:
        run_dir = resolve_run_dir(path_or_id=config.path)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Run directory: {run_dir}")

    # Step 2: Find extractor and validate
    try:
        extractor = find_extractor(run_dir=run_dir)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Step 3: Load catalog and check for existing entry
    catalog = load_catalog(catalog_path=catalog_path)
    # Extract run_id early for existing check
    meta_path = run_dir / "run_metadata.json5"
    if meta_path.exists():
        with open(meta_path) as f:
            run_id = pyjson5.load(f)["run_id"]
    else:
        run_id = run_dir.name.split("_")[-1]

    action = _handle_existing(catalog=catalog, run_id=run_id, catalog_path=catalog_path)
    if action == "quit":
        return
    if action == "delete":
        return

    # Step 4: Extract metadata
    try:
        entry = extractor.extract(path=run_dir)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Step 5: Present and get description
    _present_entry(entry)

    if action == "edit":
        # Pre-fill with existing description
        existing = [e for e in catalog.entries if e.run_id == run_id]
        if existing:
            print(f"\n  Current description: {existing[0].description}")
            catalog.entries.remove(existing[0])

    try:
        description = _input_or_esc("\nDescription (esc to cancel): ").strip()
    except _UserCancelled:
        print(f"  {_colored('Cataloging cancelled.', _C.YELLOW)}")
        return
    if not description:
        print("Description cannot be empty.")
        sys.exit(1)
    entry.description = description

    # Step 6: Tag selection (returns None if cancelled)
    selected_tags = interactive_tag_selection(catalog=catalog)
    if selected_tags is None:
        print(f"  {_colored('Cataloging cancelled.', _C.YELLOW)}")
        return
    entry.tags = selected_tags

    # Step 7: Lineage discovery + cross-referencing
    _discover_lineage(entry=entry, catalog=catalog, description=description, tags=entry.tags)

    # Step 8: Save
    catalog.entries.append(entry)
    save_catalog(catalog=catalog, catalog_path=catalog_path)
    tags_str = ", ".join(_colored(t, _C.MAGENTA) for t in entry.tags) if entry.tags else _colored("(none)", _C.DIM)
    print(f"\n  {_colored('Cataloged', _C.GREEN, _C.BOLD)} {_colored(entry.run_id, _C.YELLOW)} with tags: {tags_str}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, _UserCancelled):
        print(f"\n{_colored('Cancelled.', _C.YELLOW)}")
