"""Real moments from a run log, rebuilt exactly, as behaviour-check jobs.

The run is executed again offline, every model call answered with the answer
the log recorded for it, in order: the engine is deterministic, so it walks
through exactly the states the real run did. At the chosen moments the prompt
today's code builds is kept, with the observation it was built from, so the
model can be asked again in the very situation where it once failed, with
whatever prompts are current. A log made under another configuration cannot
be rebuilt and is refused; so is a replay that drifts from the log.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

from rpg_economy.behavior.audit import AuditReport
from rpg_economy.behavior.cases import Situation
from rpg_economy.behavior.obligations import View
from rpg_economy.behavior.runner import Job, agent_parser
from rpg_economy.config.loader import config_hash
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime, Scenario
from rpg_economy.llm.client import LLMResponse


class ReplayError(Exception):
    """The log cannot be rebuilt with the current code and configuration."""


def config_for_log(
    base: SimulationConfig,
    header: dict[str, Any],
    earlier: tuple[SimulationConfig, ...] = (),
) -> SimulationConfig:
    """The run's configuration, rebuilt and checked by hash.

    Tried from the base configuration and from ``earlier`` ones (past
    configurations kept under configs/history), each as is and per pilot stage.
    """
    from rpg_economy.campaign.pilot import STAGES, stage_config

    condition = Condition(header["condition"])
    candidates = [
        variant
        for config in (base, *earlier)
        for variant in (config, *(stage_config(config, stage) for stage in STAGES.values()))
    ]
    for candidate in candidates:
        cell = candidate.for_cell(
            scenario=Scenario(header["scenario"]),
            experiment=ExperimentRegime(header["experiment"]),
            seed=header["seed"],
        ).for_condition(condition, seed=header["seed"])
        if config_hash(cell) == header["config_hash"]:
            return cell
    raise ReplayError("the log was made under another configuration")


class _SequenceClient:
    """Answers the k-th call with the k-th answer the log recorded."""

    def __init__(self, calls: list[dict[str, Any]], targets: dict[int, tuple[str, int]]):
        self._calls = calls
        self._targets = targets
        self._next = 0
        self.captured: dict[tuple[str, int], tuple[str, str]] = {}

    @property
    def model(self) -> str:
        return "replay"

    def complete(self, *, system: str, user: str) -> LLMResponse:
        if self._next >= len(self._calls):
            raise ReplayError("the replay made more calls than the log")
        call = self._calls[self._next]
        agent = call["agent_id"]
        if agent and f'"agent_id":"{agent}"' not in user:
            raise ReplayError(f"call {self._next} drifted: expected a prompt for {agent}")
        if self._next in self._targets:
            self.captured[self._targets[self._next]] = (system, user)
        self._next += 1
        return LLMResponse(text=call["raw"], input_tokens=0, output_tokens=0, model="replay")


def pick_moments(audit: AuditReport, per_obligation: int) -> list[tuple[str, int]]:
    """Up to ``per_obligation`` missed activations per obligation, spread over the run."""
    picked: list[tuple[str, int]] = []
    for tally in audit.tallies.values():
        misses = tally.misses
        if not misses or per_obligation < 1:
            continue
        step = max(1, len(misses) // per_obligation)
        for moment in misses[::step][:per_obligation]:
            if moment not in picked:
                picked.append(moment)
    return picked


def replay_jobs(
    log: Path,
    base: SimulationConfig,
    moments: list[tuple[str, int]],
    *,
    label: str,
    earlier: tuple[SimulationConfig, ...] = (),
) -> list[Job]:
    """Rebuild ``log`` and return a job for each (agent, tick) moment."""
    from rpg_economy.runner import RunInterrupted, run_condition

    header: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event["type"] == "run_metadata":
                header = event["payload"]
            elif event["type"] == "llm_call":
                calls.append(
                    {
                        "agent_id": event.get("agent_id"),
                        "tick": event["tick"],
                        "attempt": event["payload"].get("attempt", 1),
                        "raw": event["payload"]["raw"],
                    }
                )
    cell = config_for_log(base, header, earlier)
    wanted = set(moments)
    targets: dict[int, tuple[str, int]] = {}
    for index, call in enumerate(calls):
        key = (call["agent_id"], call["tick"])
        if key in wanted and call["attempt"] == 1 and key not in targets.values():
            targets[index] = key
    client = _SequenceClient(calls, targets)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "replay.jsonl"
        # Stop once every moment is captured: the rest of the run is not needed.
        with contextlib.suppress(RunInterrupted):
            run_condition(
                cell,
                path,
                client=client,
                director_client=client,
                should_abort=lambda: len(client.captured) == len(targets),
            )
        observations: dict[tuple[str, int], dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                key = (event.get("agent_id"), event["tick"])
                if event["type"] == "observation_delivered" and key in wanted:
                    observations.setdefault(key, event["payload"])
    condition = cell.run.condition
    jobs = []
    for key in moments:
        if key not in client.captured or key not in observations:
            continue
        system, user = client.captured[key]
        delivered = observations[key]
        agent, tick = key
        jobs.append(
            Job(
                situation=Situation(
                    name=f"log:{label}:{agent}@{tick}",
                    build=None,
                    conditions=(condition,),
                    origin=f"{log}",
                ),
                condition=condition,
                agent=agent,
                system=system,
                user=user,
                parse=agent_parser(cell),
                view=View(cell, delivered["observation"], delivered.get("task")),
            )
        )
    return jobs
