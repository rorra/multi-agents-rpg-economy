"""Auditing a real run: every activation checked against the obligations.

Reads a run log only. Each activation's delivered observation is paired with
the action the agent finally gave (an activation with no valid answer counts
as a wait, which is what the engine executed). Nothing here calls a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.behavior.obligations import OBLIGATIONS, View, judge
from rpg_economy.config.models import SimulationConfig

WAIT = {"action": "wait"}


@dataclass
class Activation:
    agent_id: str
    tick: int
    seq: int
    observation: dict[str, Any]
    task: dict[str, Any] | None
    action: dict[str, Any] | None = None


@dataclass
class ObligationTally:
    name: str
    applicable: int = 0
    met: int = 0
    #: (agent_id, tick) of every activation that missed it.
    misses: list[tuple[str, int]] = field(default_factory=list)

    @property
    def share(self) -> float | None:
        return self.met / self.applicable if self.applicable else None


@dataclass
class AuditReport:
    log: str
    condition: str
    prompt_version: str | None
    activations: int
    tallies: dict[str, ObligationTally]

    def to_payload(self) -> dict[str, Any]:
        return {
            "log": self.log,
            "condition": self.condition,
            "prompt_version": self.prompt_version,
            "activations": self.activations,
            "obligations": {
                name: {
                    "applicable": tally.applicable,
                    "met": tally.met,
                    "share": None if tally.share is None else round(tally.share, 4),
                    "misses": [list(m) for m in tally.misses],
                }
                for name, tally in self.tallies.items()
            },
        }


def read_activations(path: Path) -> tuple[dict[str, Any], list[Activation]]:
    """The run header and its agent activations, each with its final action."""
    header: dict[str, Any] = {}
    found: dict[tuple[str, int], Activation] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            kind = event["type"]
            if kind == "run_metadata":
                header = event["payload"]
            elif kind == "observation_delivered":
                payload = event["payload"]
                key = (event["agent_id"], payload["activation"]["seq"])
                found[key] = Activation(
                    agent_id=event["agent_id"],
                    tick=event["tick"],
                    seq=payload["activation"]["seq"],
                    observation=payload["observation"],
                    task=payload.get("task"),
                )
            elif kind == "agent_output":
                payload = event["payload"]
                key = (event["agent_id"], payload["activation"]["seq"])
                if key in found:
                    found[key].action = payload
    return header, list(found.values())


def audit_log(path: Path, config: SimulationConfig) -> AuditReport:
    """Tally, per obligation, the activations it applied to and those that met it."""
    header, activations = read_activations(path)
    tallies = {o.name: ObligationTally(o.name) for o in OBLIGATIONS}
    for activation in activations:
        view = View(config, activation.observation, activation.task)
        for name, met in judge(view, activation.action or WAIT).items():
            if met is None:
                continue
            tally = tallies[name]
            tally.applicable += 1
            if met:
                tally.met += 1
            else:
                tally.misses.append((activation.agent_id, activation.tick))
    hashes = header.get("prompt_hashes")
    version = hashes.get("version") if isinstance(hashes, dict) else None
    return AuditReport(
        log=str(path),
        condition=str(header.get("condition")),
        prompt_version=version,
        activations=len(activations),
        tallies=tallies,
    )
