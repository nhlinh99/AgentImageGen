"""
Helpers to derive ordered Celery task names from canvas objects (chain / group / chord / signature)
for chain_process_result() and process() without maintaining duplicate task lists.
"""

from __future__ import annotations

from typing import Any, List, Set


def _is_chord(node: Any) -> bool:
    """True when Celery wrapped group+downstream steps as a chord (has .body)."""
    if getattr(node, "body", None) is None:
        return False
    if getattr(node, "task", None) == "celery.chord":
        return True
    return type(node).__name__ == "_chord"


def flatten_celery_canvas_task_names(canvas: Any) -> List[str]:
    """
    Walk a Celery canvas and return task name strings in execution order.
    Repeats names when the same task appears multiple times.
    Supports nested chain, group, chord, and leaf signatures.

    When a ``group`` is followed by more steps in a ``chain``, Celery turns that
    segment into a chord: ``.tasks`` only lists the parallel header signatures;
    the callback steps after the group are stored on ``.body`` and must be walked
    separately, or task names after the group are skipped.
    """
    names: List[str] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if _is_chord(node):
            # Header: parallel tasks (same as group members)
            for child in getattr(node, "tasks", None) or ():
                walk(child)
            walk(getattr(node, "body"))
            return
        subtasks = getattr(node, "tasks", None)
        if subtasks:
            for child in subtasks:
                walk(child)
            return
        task_name = getattr(node, "task", None)
        if task_name is not None:
            names.append(task_name)

    walk(canvas)
    return names


def collect_terminal_leaf_signatures(canvas: Any) -> List[Any]:
    """
    Leaf signatures that may complete last in a serialized pipeline sense:

    - Last step of a ``chain`` (recursive).
    - Every branch of a terminal ``group`` (parallel tail).
    - Only the ``body`` of a ``chord`` (callback after the header group).

    Used to set ``is_last_pipeline_task`` so the success handler finalizes the job
    only for these steps.
    """

    def walk(node: Any) -> List[Any]:
        if node is None:
            return []
        if _is_chord(node):
            return walk(getattr(node, "body"))
        subtasks = getattr(node, "tasks", None)
        if subtasks:
            tn = type(node).__name__
            if tn == "_chain":
                return walk(subtasks[-1])
            if tn == "group":
                out: List[Any] = []
                for child in subtasks:
                    out.extend(walk(child))
                return out
            return walk(subtasks[-1])
        if getattr(node, "task", None) is not None:
            return [node]
        return []

    return walk(canvas)


def collect_all_leaf_signatures(canvas: Any) -> List[Any]:
    """Every leaf ``Signature`` in execution preorder (same walk as flatten)."""

    def walk(node: Any) -> List[Any]:
        if node is None:
            return []
        if _is_chord(node):
            out: List[Any] = []
            for child in getattr(node, "tasks", None) or ():
                out.extend(walk(child))
            out.extend(walk(getattr(node, "body")))
            return out
        subtasks = getattr(node, "tasks", None)
        if subtasks:
            out = []
            for child in subtasks:
                out.extend(walk(child))
            return out
        if getattr(node, "task", None) is not None:
            return [node]
        return []

    return walk(canvas)


def mark_is_last_pipeline_task_kwargs(canvas: Any) -> None:
    """
    Set ``is_last_pipeline_task`` on every leaf: True only on terminal leaf signatures.

    Pipelines that never call this keep ``is_last_pipeline_task`` unset; signal handlers
    treat that as legacy behavior (still run completion checks).
    """
    terminal_ids: Set[int] = {id(s) for s in collect_terminal_leaf_signatures(canvas)}
    for leaf in collect_all_leaf_signatures(canvas):
        kwargs = getattr(leaf, "kwargs", None)
        if kwargs is None:
            continue
        kwargs["is_last_pipeline_task"] = id(leaf) in terminal_ids
