"""Generate a wandb workspace view URL for a selected set of runs.

Uses the ``wandb-workspaces`` package to create (and save) a Workspace whose
runset is filtered to the given run IDs. Workspaces are per (entity, project),
so all input runs must share the same entity+project — otherwise a
``MixedProjectError`` is raised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class WandbRunRef:
    """Minimal descriptor of a run for wandb view creation."""
    run_id: str
    entity: str
    project: str


def parse_wandb_url(url: str) -> tuple[str, str, str] | None:
    """Extract ``(entity, project, run_id)`` from a wandb URL.

    Expects the shape ``https://wandb.ai/{entity}/{project}/runs/{run_id}``.
    Returns ``None`` if the URL doesn't match that shape.
    """
    try:
        parts = urlparse(url).path.strip("/").split("/")
        # ['entity', 'project', 'runs', 'run_id', ...]
        if len(parts) >= 4 and parts[2] == "runs":
            return parts[0], parts[1], parts[3]
    except Exception:
        pass
    return None


class MixedProjectError(ValueError):
    """Raised when selected runs span more than one (entity, project) pair."""


def create_wandb_view_url(
    *, refs: list[WandbRunRef], view_name: str | None = None,
) -> str:
    """Create a saved wandb workspace view for ``refs`` and return its URL.

    Args:
        refs: runs to include in the view. Must all share the same
            ``(entity, project)``.
        view_name: optional name for the saved view. Auto-generated if None.

    Raises:
        ValueError: if ``refs`` is empty or any ref is missing required fields.
        MixedProjectError: if refs span more than one (entity, project).
        ImportError: if ``wandb-workspaces`` is not installed (error message
            includes the pip install hint).
    """
    if not refs:
        raise ValueError("create_wandb_view_url: no runs provided")

    missing = [r for r in refs if not (r.run_id and r.entity and r.project)]
    if missing:
        raise ValueError(
            f"create_wandb_view_url: {len(missing)} run(s) missing "
            "entity/project/run_id"
        )

    groups = {(r.entity, r.project) for r in refs}
    if len(groups) > 1:
        groups_str = ", ".join(sorted(f"{e}/{p}" for e, p in groups))
        raise MixedProjectError(
            f"Selected runs span multiple wandb projects ({groups_str}). "
            "wandb workspace views are per-project; pick runs from one project."
        )
    entity, project = next(iter(groups))

    # Lazy import — keep the viewer importable even if the package isn't installed.
    try:
        from wandb_workspaces.workspaces.interface import (
            Workspace,
            RunsetSettings,
            WorkspaceSettings,
        )
    except ImportError as exc:
        raise ImportError(
            "wandb-workspaces not installed. Install with: "
            "pip install wandb-workspaces"
        ) from exc

    # Preserve insertion order (deduplicate while keeping first occurrence)
    seen: set[str] = set()
    run_ids: list[str] = []
    for r in refs:
        if r.run_id not in seen:
            seen.add(r.run_id)
            run_ids.append(r.run_id)

    id_list_literal = "[" + ", ".join(f"'{rid}'" for rid in run_ids) + "]"
    # 'ID' is the frontend metric name for run id (maps to backend 'name').
    # See wandb_workspaces.expr.FE_METRIC_NAME_MAP.
    filter_expr = f"Metric('ID') in {id_list_literal}"

    name = view_name or f"viewer-selection-{run_ids[0][:8]}-{len(run_ids)}runs"
    # Wandb Workspace rejects names with angle brackets, emojis, etc.
    name = re.sub(r"[<>]", "", name).strip()
    if not name:
        name = f"viewer-selection-{run_ids[0][:8]}-{len(run_ids)}runs"
    # Wandb caps workspace names — truncate to 128 chars
    if len(name) > 128:
        name = name[:125] + "..."

    # Ordering: we tried patching a _view_order_{uid} config key onto each run
    # via the wandb API, then sorting with RunsetSettings(order=[Ordering(
    # item=Config(order_key), ascending=True)]). The config keys were written
    # correctly and the table sort worked, but the chart sidebar/legend order
    # (what you actually see) was unaffected — wandb groups runs by their
    # wandb group first, then uses an opaque internal sort (appears to be:
    # group root first, then forks/children by creation date, then same-name
    # continuations last). This is a wandb frontend limitation; the
    # programmatic workspaces API has no way to control legend ordering.

    # Default to 0.99 EMA smoothing for all line plots.
    # wandb's smoothing_weight is an integer percent (0-100); 99 == 0.99.
    ws = Workspace(
        entity=entity,
        project=project,
        name=name,
        runset_settings=RunsetSettings(filters=filter_expr),
        settings=WorkspaceSettings(
            smoothing_type="exponential",
            smoothing_weight=99,
        ),
        auto_generate_panels=True,
    )
    ws.save()  # required — .url is only valid after save
    return ws.url
