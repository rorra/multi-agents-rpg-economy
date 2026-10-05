"""Running the behaviour cases against a model, and pricing them first."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from rpg_economy.agents.action import AgentAction
from rpg_economy.agents.observation import build_observation, should_include_bulletin
from rpg_economy.agents.parsing import parse_action
from rpg_economy.agents.prompts import build_agent_prompt
from rpg_economy.behavior.cases import DIRECTOR_THRESHOLD, SITUATIONS, Situation
from rpg_economy.behavior.obligations import BY_NAME, View, applicable
from rpg_economy.config.models import ModelPricing, SimulationConfig
from rpg_economy.domain import Condition, Priority, TaskType
from rpg_economy.engine import Engine, build_world
from rpg_economy.llm.client import LLMClient
from rpg_economy.llm.tokens import TokenLedger, estimate_tokens
from rpg_economy.logging_ import EventLogWriter
from rpg_economy.messaging import MessageBus
from rpg_economy.orchestration import build_directory
from rpg_economy.orchestration.global_view import build_global_summary

#: Output tokens per call when pricing: the rate measured in the check's own
#: real runs (about 690), above a pilot activation's because its situations
#: are the harder ones.
OUTPUT_PER_CALL = 700
#: Counted answers an aggregate rule needs in a group before that group is judged.
MIN_AGGREGATE = 8
#: The largest share of invalid answers, over every call, the check accepts.
MAX_INVALID_SHARE = 0.05
CENT = Decimal("0.01")


@dataclass
class Job:
    """One situation in one condition: the prompt, and how to read and judge an answer."""

    situation: Situation
    condition: Condition
    agent: str
    system: str
    user: str
    parse: Callable[[str], AgentAction | None]
    #: The agent's observation, judged by the obligations; None for the Director.
    view: View | None = None

    @property
    def name(self) -> str:
        return self.situation.name

    def obligations(self) -> list[str]:
        """The obligations every answer here is judged on."""
        if self.view is None:
            return [VALID_DECISION]
        return [o.name for o in applicable(self.view)]

    def threshold(self, obligation: str) -> float:
        return DIRECTOR_THRESHOLD if obligation == VALID_DECISION else BY_NAME[obligation].threshold


#: The Director's one obligation: a valid decision.
VALID_DECISION = "director_valid_decision"


def _is_aggregate(name: str) -> bool:
    return name in BY_NAME and BY_NAME[name].aggregate


@dataclass
class CaseResult:
    """A job's answers, and per obligation how many of them met it."""

    case: str
    condition: str
    agent: str
    origin: str
    thresholds: dict[str, float]
    answers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def invalid(self) -> int:
        return sum(1 for a in self.answers if a["outcome"] == "invalid")

    def met(self, obligation: str) -> int:
        return sum(1 for a in self.answers if a["judged"].get(obligation))

    def counted(self, obligation: str) -> int:
        """Answers the rule judged: all but those it does not count (None)."""
        return sum(1 for a in self.answers if a["judged"].get(obligation) is not None)

    def share(self, obligation: str) -> float:
        counted = self.counted(obligation)
        return self.met(obligation) / counted if counted else 1.0

    @property
    def failing(self) -> list[str]:
        """Rules below threshold here; aggregate rules are judged on the total."""
        return [
            name
            for name, threshold in self.thresholds.items()
            if not _is_aggregate(name) and (not self.answers or self.share(name) < threshold)
        ]

    @property
    def passed(self) -> bool:
        return bool(self.answers) and not self.failing

    def to_payload(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "condition": self.condition,
            "agent": self.agent,
            "origin": self.origin,
            "invalid": self.invalid,
            "total": len(self.answers),
            "obligations": {
                name: {
                    "met": self.met(name),
                    "counted": self.counted(name),
                    "share": round(self.share(name), 4),
                    "threshold": threshold,
                    "passed": name not in self.failing,
                }
                for name, threshold in self.thresholds.items()
            },
            "passed": self.passed,
            "answers": self.answers,
        }


@dataclass
class BehaviorReport:
    model: str
    reps: int
    results: list[CaseResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def calls(self) -> int:
        return sum(len(r.answers) for r in self.results)

    @property
    def invalid_share(self) -> float:
        return sum(r.invalid for r in self.results) / self.calls if self.calls else 0.0

    @property
    def failing_aggregates(self) -> list[str]:
        """Aggregate rules below threshold over the built situations or the real moments.

        Each group is judged on its own, once it has enough counted answers:
        the built situations alone can be easier than the real ones.
        """
        failing = []
        for name in sorted({n for r in self.results for n in r.thresholds if _is_aggregate(n)}):
            for real in (False, True):
                group = [r for r in self.results if r.case.startswith("log:") is real]
                counted = sum(r.counted(name) for r in group if name in r.thresholds)
                met = sum(r.met(name) for r in group if name in r.thresholds)
                if counted >= MIN_AGGREGATE and met / counted < BY_NAME[name].threshold:
                    failing.append(name)
                    break
        return failing

    @property
    def passed(self) -> bool:
        return (
            all(r.passed for r in self.results)
            and not self.failing_aggregates
            and self.invalid_share <= MAX_INVALID_SHARE
        )

    def by_obligation(self) -> dict[str, dict[str, int]]:
        """Each obligation over every job it applied in: answers counted, met, failing jobs."""
        totals: dict[str, dict[str, int]] = {}
        for result in self.results:
            for name in result.thresholds:
                entry = totals.setdefault(name, {"answers": 0, "met": 0, "jobs": 0, "failed": 0})
                entry["answers"] += result.counted(name)
                entry["met"] += result.met(name)
                entry["jobs"] += 1
                entry["failed"] += name in result.failing
        return dict(sorted(totals.items()))

    def to_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reps": self.reps,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "invalid_share": round(self.invalid_share, 4),
            "max_invalid_share": MAX_INVALID_SHARE,
            "passed": self.passed,
            "failing_aggregates": self.failing_aggregates,
            "by_obligation": self.by_obligation(),
            "results": [r.to_payload() for r in self.results],
        }


def _agent_job(
    situation: Situation, config: SimulationConfig, engine: Engine, messages, agent
) -> Job:
    condition = config.run.condition
    task = None
    if situation.task is not None:
        from rpg_economy.orchestration.protocol import Task

        kind, params = situation.task
        task = Task(
            task_id="task-1",
            type=TaskType(kind),
            agent_id=agent,
            params=params,
            priority=Priority.HIGH,
        ).to_agent_payload()
    if should_include_bulletin(condition):
        engine.publish_bulletin(build_global_summary(config, engine))
    observation = build_observation(
        config,
        engine,
        agent,
        messages=messages,
        include_bulletin=should_include_bulletin(condition),
    )
    prompt = build_agent_prompt(observation, config, condition=condition, task=task)
    return Job(
        situation=situation,
        condition=condition,
        agent=agent,
        system=prompt.system,
        user=prompt.user,
        parse=agent_parser(config),
        view=View(config, observation.to_payload(), task),
    )


def agent_parser(config: SimulationConfig) -> Callable[[str], AgentAction | None]:
    condition = config.run.condition
    max_orders = config.market.max_orders_per_activation
    return lambda text: parse_action(text, condition=condition, max_orders=max_orders)


def _director_job(situation: Situation, config: SimulationConfig, engine: Engine) -> Job:
    from rpg_economy.orchestration.director import Director, DirectorCycleReport

    director = Director(config, engine, _NoClient(), TokenLedger(), activate=_no_activation)

    def parse(text: str) -> AgentAction | None:
        director._parse(text)  # raises on an invalid decision
        return None

    return Job(
        situation=situation,
        condition=Condition.A,
        agent="director",
        system=director._system,
        user=director._user_prompt(DirectorCycleReport()),
        parse=parse,
    )


class _NoClient:
    """The Director's client is never called here: only its prompt is built."""

    model = "none"

    def complete(self, *, system: str, user: str):  # pragma: no cover - never called
        raise AssertionError("the behaviour check calls the model itself")


def _no_activation(engine, agent_id, task):  # pragma: no cover - never called
    raise AssertionError("no agent is activated in a behaviour check")


def build_jobs(config: SimulationConfig, situations: list[Situation] | None = None) -> list[Job]:
    """Every situation's prompt in every condition it runs in. Offline: no model is called."""
    from rpg_economy.runner import _metadata

    jobs: list[Job] = []
    with tempfile.TemporaryDirectory() as tmp:
        for situation in SITUATIONS if situations is None else situations:
            for condition in situation.conditions:
                cell = config.for_condition(condition, seed=config.run.seed)
                path = Path(tmp) / f"{situation.name.replace(':', '-')}-{condition.value}.jsonl"
                with EventLogWriter(path, _metadata(cell)) as log:
                    engine = Engine(cell, build_world(cell), log)
                    messages = MessageBus(build_directory(engine))
                    agent = situation.build(engine)
                    if situation.director:
                        jobs.append(_director_job(situation, cell, engine))
                    else:
                        jobs.append(_agent_job(situation, cell, engine, messages, agent))
    return jobs


def estimate_behavior_cost(jobs: list[Job], reps: int, pricing: ModelPricing) -> dict[str, Any]:
    """Projected tokens and cost of asking every job ``reps`` times."""
    input_tokens = sum(estimate_tokens(job.system) + estimate_tokens(job.user) for job in jobs)
    input_tokens *= reps
    output_tokens = OUTPUT_PER_CALL * len(jobs) * reps
    million = Decimal(1_000_000)
    cost = (
        Decimal(input_tokens) / million * pricing.input_per_1m_tokens
        + Decimal(output_tokens) / million * pricing.output_per_1m_tokens
    ).quantize(CENT, rounding=ROUND_HALF_UP)
    return {
        "calls": len(jobs) * reps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": cost,
        "currency": pricing.currency,
    }


def _ask(client: LLMClient, job: Job, reps: int) -> tuple[Job, list[dict[str, Any]], int, int]:
    answers: list[dict[str, Any]] = []
    tokens_in = tokens_out = 0
    names = job.obligations()
    for _ in range(reps):
        response = client.complete(system=job.system, user=job.user)
        tokens_in += response.input_tokens
        tokens_out += response.output_tokens
        try:
            action = job.parse(response.text)
        except Exception as exc:  # an invalid answer is a result, not an error
            answers.append(
                {
                    "outcome": "invalid",
                    "detail": str(exc)[:200],
                    "judged": {name: None if _is_aggregate(name) else False for name in names},
                    "raw": response.text,
                }
            )
            continue
        if job.view is None:
            judged = {VALID_DECISION: True}
        else:
            payload = action.to_payload()
            judged = {o.name: o.met(job.view, payload) for o in applicable(job.view)}
        answers.append(
            {
                "outcome": (
                    "unexpected" if any(v is False for v in judged.values()) else "expected"
                ),
                "action": action.to_payload() if action is not None else None,
                "judged": judged,
                "raw": response.text,
            }
        )
    return job, answers, tokens_in, tokens_out


def run_behavior_check(
    jobs: list[Job], client: LLMClient, reps: int, *, workers: int = 8
) -> BehaviorReport:
    """Ask the model ``reps`` times per job and judge every answer."""
    report = BehaviorReport(model=client.model, reps=reps)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for job, answers, tokens_in, tokens_out in pool.map(
            lambda job: _ask(client, job, reps), jobs
        ):
            report.input_tokens += tokens_in
            report.output_tokens += tokens_out
            report.results.append(
                CaseResult(
                    case=job.name,
                    condition=job.condition.value,
                    agent=job.agent,
                    origin=job.situation.origin,
                    thresholds={name: job.threshold(name) for name in job.obligations()},
                    answers=answers,
                )
            )
    return report
