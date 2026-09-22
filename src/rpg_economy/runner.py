"""Assembling and executing runs, singly and as seed-paired trios.

One seed produces a trio: conditions A, B and C over the same world. They share
it by construction rather than by copying -- the same configuration builds the
same agents with the same endowments, the shock tick derives from the configured
phase structure, and every random stream is derived by name from the run seed, so
the n-th expedition of one condition draws exactly what the n-th expedition of
another draws (design v1.0 s12; the named-stream rationale is in engine.rng).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rpg_economy.agents.llm_agent import LLMAgentRunner
from rpg_economy.agents.offline import (
    director_responder,
    rule_based_responder,
)
from rpg_economy.agents.prompts import PROMPT_VERSION
from rpg_economy.cli_support import build_metadata
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime
from rpg_economy.engine import Engine, WorldState, build_world
from rpg_economy.integrity import verify_trio
from rpg_economy.llm.budget import TokenBudget, common_active_horizon
from rpg_economy.llm.client import LLMClient, Responder, build_client
from rpg_economy.llm.tokens import TokenLedger
from rpg_economy.logging_ import EventLogWriter, EventType, code_digest
from rpg_economy.messaging import MessageBus
from rpg_economy.orchestration import DecentralizedOrchestrator, build_directory
from rpg_economy.orchestration.centralized import TaskActivationRunner
from rpg_economy.orchestration.director import Director, prompt_hashes
from rpg_economy.orchestration.global_view import GLOBAL_VIEW_VERSION, build_global_summary


@dataclass
class RunResult:
    """What a completed run leaves behind."""

    config: SimulationConfig
    state: WorldState
    log_path: Path
    ledger: TokenLedger
    client: LLMClient
    budget: TokenBudget
    orchestrator: object | None = None
    director: Director | None = None

    @property
    def exhaustion_tick(self) -> int | None:
        """The tick at which the token budget ran out, if it did (s11.2)."""
        return self.budget.exhausted_tick


#: Why a run or trio was interrupted (``RunInterrupted.reason``).
INTERRUPT_ABORT = "abort"
INTERRUPT_STOP = "stop"


@dataclass(eq=False, repr=False)
class RunInterrupted(Exception):
    """A run or trio ended early at the caller's request, not by failure.

    Raised by :func:`run_condition` when ``should_abort`` returns True, and by
    :func:`run_trio` when either callback ends it:

    * ``reason == INTERRUPT_ABORT``: a run was cut mid-way. Its log carries a
      ``run_aborted`` event and no ``run_end``, so it reads as unfinished and a
      resume re-runs it from the start. ``interrupted`` names its condition.
    * ``reason == INTERRUPT_STOP``: ``should_stop`` returned True between
      conditions. No run was cut; the conditions not yet started were never
      begun, and ``interrupted`` is None.

    Either way ``completed`` holds the conditions that finished (each with its
    run_end written), which the caller keeps. An interrupted trio is not
    verified: it is not a trio yet.
    """

    reason: str
    completed: dict[Condition, RunResult] = field(default_factory=dict)
    interrupted: Condition | None = None
    log_path: Path | None = None

    def __post_init__(self) -> None:
        super().__init__(self.reason, self.interrupted)

    def __str__(self) -> str:
        finished = ", ".join(c.value for c in self.completed) or "none"
        where = "" if self.interrupted is None else f" during {self.interrupted.value}"
        return f"run interrupted ({self.reason}{where}); finished: {finished}"

    def __repr__(self) -> str:
        # Short on purpose: the generated repr would dump every completed
        # RunResult, world state included.
        where = None if self.interrupted is None else self.interrupted.value
        return f"RunInterrupted(reason={self.reason!r}, interrupted={where!r})"


def _metadata(config: SimulationConfig):
    return build_metadata(config).model_copy(
        update={
            "phase": config.run.phase,
            "prompt_hashes": {**prompt_hashes(), "version": PROMPT_VERSION},
            "global_view_version": GLOBAL_VIEW_VERSION,
            "code_digest": code_digest(),
        }
    )


def run_condition(
    config: SimulationConfig,
    log_path: Path | str,
    *,
    responder: Responder | None = None,
    director_response: Responder | None = None,
    client: LLMClient | None = None,
    director_client: LLMClient | None = None,
    cycles: int | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> RunResult:
    """Execute one run under whichever condition the configuration names.

    ``should_abort`` is asked once per macro-cycle, before the cycle runs. When
    it returns True the run stops at once: a ``run_aborted`` event is logged,
    ``run_end`` is not, and :class:`RunInterrupted` (reason ``abort``) is
    raised, so the log stays unfinished and a resume re-runs it. Without it the
    run is exactly what it would otherwise be.
    """
    # model_copy performs no validation, so a condition passed as a bare string
    # would never match any identity check and the run would silently execute as
    # something else (design v1.0 s12). Fail here instead.
    if not isinstance(config.run.condition, Condition):
        raise TypeError(
            f"run.condition is {type(config.run.condition).__name__} "
            f"{config.run.condition!r}, not a Condition enum member; use "
            f"SimulationConfig.for_condition() to derive a per-condition config"
        )
    state = build_world(config)
    total_cycles = cycles if cycles is not None else config.time.total_cycles

    with EventLogWriter(log_path, _metadata(config)) as log:
        engine = Engine(config, state, log)
        messages = MessageBus(build_directory(engine))
        ledger = TokenLedger()
        # Experiment 1 has no token cap; Experiment 2 takes it from the
        # config, whose validator refuses to load without one (design s8.4).
        budget = TokenBudget(
            cap=(
                config.budget.experiment_2_total_tokens
                if config.run.experiment is ExperimentRegime.EQUAL_COST
                else None
            )
        )

        # build_client honours config.llm.mock, so a config with mock=False
        # never runs offline while its metadata records llm_mock=False.
        agent_client = client or build_client(
            config.llm, responder or rule_based_responder(config, engine, messages=messages)
        )
        agents = LLMAgentRunner(config, agent_client, ledger, messages=messages, budget=budget)

        orchestrator: object | None = None
        director: Director | None = None

        if config.run.condition is Condition.A:
            boss_client = director_client or build_client(
                config.llm, director_response or director_responder(config, engine)
            )
            director = Director(
                config,
                engine,
                boss_client,
                ledger,
                TaskActivationRunner(config, agents),
                budget=budget,
            )
            decision_phase = director
        else:
            # The same budget the agents charge: once it is spent the scheduler
            # stops serving activations, as the Director stops calling in A
            # (design v1.0 s8.4).
            orchestrator = DecentralizedOrchestrator(
                config, engine, agents, messages=messages, budget=budget
            )
            decision_phase = orchestrator
            if config.run.condition is Condition.C:
                decision_phase = _with_bulletin(config, orchestrator)

        for _ in range(total_cycles):
            if should_abort is not None and should_abort():
                # Not a run_end: the log must read as unfinished, so a resume
                # re-runs it rather than analyse half a run.
                # The spend so far, as tokens_consumed carries it at run_end:
                # an aborted run's cost is then in its own log.
                engine.log_event(
                    EventType.RUN_ABORTED,
                    payload={
                        "cycles_completed": engine.cycle,
                        "cycles_planned": total_cycles,
                        **ledger.snapshot(),
                        "budget": budget.snapshot(),
                    },
                )
                raise RunInterrupted(
                    reason=INTERRUPT_ABORT,
                    interrupted=config.run.condition,
                    log_path=Path(log_path),
                )
            engine.run_cycle(decision_phase)

        engine.log_event(
            EventType.TOKENS_CONSUMED,
            payload={**ledger.snapshot(), "budget": budget.snapshot()},
        )
        engine.log_run_end()

    return RunResult(
        config=config,
        state=state,
        log_path=Path(log_path),
        ledger=ledger,
        client=agent_client,
        budget=budget,
        orchestrator=orchestrator,
        director=director,
    )


def run_decentralized(
    config: SimulationConfig,
    log_path: Path | str,
    *,
    responder: Responder | None = None,
    client: LLMClient | None = None,
    cycles: int | None = None,
) -> RunResult:
    """Execute a Condition B or C run. Kept for callers that predate trios."""
    return run_condition(config, log_path, responder=responder, client=client, cycles=cycles)


def _with_bulletin(config: SimulationConfig, orchestrator: DecentralizedOrchestrator):
    """Publish the bulletin as the decision phase opens, then decide (design s3.3).

    The same moment the Director reads its global view in A: after the cycle's
    mechanical ticks, before anyone acts. Publishing before the ticks would hand
    C a view one tick older than A's, and the ablation would compare different
    information rather than the same information delivered differently.
    """

    def decision_phase(engine: Engine) -> None:
        engine.publish_bulletin(build_global_summary(config, engine))
        orchestrator(engine)

    return decision_phase


def run_trio(
    base_config: SimulationConfig,
    directory: Path | str,
    *,
    seed: int | None = None,
    cycles: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> dict[Condition, RunResult]:
    """Run conditions A, B and C on one seed, as a paired trio.

    The three runs differ only in their condition. Everything the design
    holds constant across a trio -- agents, ids, traits, endowments, recipes,
    thresholds, the shock and its tick, and the engine's random streams -- comes
    from the shared configuration and the shared seed.

    Two levels of interruption, both raising :class:`RunInterrupted` with the
    conditions already finished in ``completed``:

    * ``should_stop`` is asked before each condition starts. When True, the
      remaining conditions are not started (reason ``stop``).
    * ``should_abort`` is passed to every run and asked once per macro-cycle.
      When True, the running condition stops at once with an unfinished log
      (reason ``abort``).
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)

    runs: dict[Condition, RunResult] = {}
    for condition in (Condition.A, Condition.B, Condition.C):
        if should_stop is not None and should_stop():
            raise RunInterrupted(reason=INTERRUPT_STOP, completed=dict(runs))
        # The enum, not its value: model_copy does not coerce, and a string here
        # would silently fail every `is Condition.X` check downstream, running
        # all three conditions as the same one.
        config = base_config.for_condition(condition, seed=seed)
        try:
            runs[condition] = run_condition(
                config,
                destination / f"{condition.value}.jsonl",
                cycles=cycles,
                should_abort=should_abort,
            )
        except RunInterrupted as exc:
            exc.completed = dict(runs)
            raise

    # Definitional, not empirical: a trio that fails these did not run as three
    # conditions, whatever its numbers say (design v1.0 s12).
    verify_trio(runs)
    return runs


def trio_common_active_horizon(runs: dict[Condition, RunResult]) -> int:
    """The trio's common active horizon (design v1.0 s8.4).

    Ends at the earliest budget exhaustion across A, B and C, or at the planned
    horizon if none exhausted. Economic metrics are recomputed over this interval
    as Experiment 2's secondary analysis, so the three architectures are compared
    while all of them still had decision capacity -- the primary analysis keeps
    the full horizon, because running out early is itself a result.
    """
    planned = next(iter(runs.values())).config.time.horizon_ticks
    return common_active_horizon([run.exhaustion_tick for run in runs.values()], planned)
