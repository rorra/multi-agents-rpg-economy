"""A quick read of finished or running logs: are the agents behaving?

Meant for a short check run before anything costly. It counts, per log, what
the agents did, what the engine refused and why, which outputs were invalid
and why, and, in Condition A, how the Director spread its tasks. Read-only,
from the logs alone, like the rest of the dashboard package.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LogCheck:
    """What one run log shows about the agents' behaviour."""

    path: Path
    condition: str = "?"
    last_tick: int = 0
    activations: int = 0
    actions: Counter = field(default_factory=Counter)
    invalid: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)
    tasks: Counter = field(default_factory=Counter)
    expeditions: int = 0
    trades: int = 0
    produced: int = 0

    @property
    def invalid_total(self) -> int:
        return sum(self.invalid.values())

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())


def _reason_key(reason: str) -> str:
    """The stable part of a refusal or invalid-output message."""
    text = reason.split("\n")[0]
    for cut in (": {", ": '", ": [", ' ("', " got "):
        if cut in text:
            text = text.split(cut)[0]
    return text[:80]


def check_log(path: Path) -> LogCheck:
    """Count one log's behaviour. A log still being written reads up to its last line."""
    result = LogCheck(path=path)
    with open(path, encoding="utf-8") as handle:
        try:
            header = json.loads(handle.readline())
            result.condition = str((header.get("payload") or {}).get("condition", "?"))
        except json.JSONDecodeError:
            pass
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                break  # a line still being written
            kind = event.get("type")
            payload = event.get("payload") or {}
            result.last_tick = max(result.last_tick, event.get("tick") or 0)
            if kind == "agent_output":
                result.activations += 1
                result.actions[str(payload.get("action"))] += 1
            elif kind == "agent_output_invalid":
                result.invalid[_reason_key(str(payload.get("reason")))] += 1
            elif kind == "action_rejected":
                code = payload.get("code") or "other"
                result.rejected[f"{code}: {_reason_key(str(payload.get('reason')))}"] += 1
            elif kind == "director_task_assigned":
                result.tasks[str(event.get("agent_id"))] += 1
            elif kind == "expedition_started":
                result.expeditions += 1
            elif kind == "trade_executed":
                result.trades += 1
            elif kind == "production_completed":
                result.produced += int(payload.get("quantity") or 0)
    return result


def check_directory(directory: Path) -> list[LogCheck]:
    """Every run log under ``directory``, by path."""
    paths = sorted(p for p in Path(directory).rglob("*.jsonl") if "previous-" not in str(p))
    return [check_log(p) for p in paths]


def render_checks(checks: list[LogCheck], t) -> str:
    """The checks as text, in the reader's language (``t`` is the catalog lookup)."""
    lines: list[str] = []
    for check in checks:
        share = check.invalid_total / check.activations if check.activations else 0.0
        lines.append(
            t(
                "check.header",
                path=str(check.path),
                condition=check.condition,
                tick=check.last_tick,
                activations=check.activations,
                invalid=check.invalid_total,
                share=f"{share:.0%}",
                rejected=check.rejected_total,
            )
        )
        lines.append(
            t(
                "check.activity",
                expeditions=check.expeditions,
                trades=check.trades,
                produced=check.produced,
            )
        )
        if check.actions:
            mix = ", ".join(f"{k} {v}" for k, v in check.actions.most_common())
            lines.append(t("check.actions", actions=mix))
        if check.tasks:
            spread = ", ".join(f"{k} {v}" for k, v in check.tasks.most_common())
            lines.append(t("check.tasks", agents=len(check.tasks), spread=spread))
        for reason, count in check.invalid.most_common(6):
            lines.append(t("check.invalid", count=count, reason=reason))
        for reason, count in check.rejected.most_common(6):
            lines.append(t("check.rejected", count=count, reason=reason))
        lines.append("")
    return "\n".join(lines)
