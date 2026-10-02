"""Parallel campaign execution (IMPLEMENTATION.md section 5).

Wall-clock is the binding resource once the provider is real: at ~3.5s per call
and roughly a million calls across pilot and final, a serial campaign is weeks.
Runs are independent -- a run's outcome depends only on its own seed, condition,
scenario and experiment -- so the campaign parallelises across runs even though
each run is inherently serial inside itself.

Three properties matter more than throughput here:

Isolation. Each worker builds its own config, engine, state, client and ledger.
Nothing mutable is shared, so parallelism cannot change any run's outcome.

Unbilled rejections never enter the ledger. A 429 or a 400 is charged nothing
by the provider, so it costs nothing in our accounting either; otherwise
retries would inflate the Experiment 2 budget against spend that never happened.

Resumability. A crashed campaign resumes from completed runs and never re-pays
for a finished one: not for a finished trio, and not for a finished condition of
a trio that failed part-way.

Interruption. An ``Interrupt`` (campaign.interrupt) is shared by every worker. Its
first level starts nothing new -- no queued trio, no retry, no next condition --
while each running condition finishes; its second level aborts the running
conditions at their next macro-cycle. The pool is always joined before the
caller returns, so the output directory's lock outlives every worker.
"""

from __future__ import annotations

import os
import random
import time
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.campaign.interrupt import Interrupt
from rpg_economy.campaign.plan import CampaignPhase, CampaignPlan, PlannedTrio, cell_config
from rpg_economy.campaign.runner import (
    execute_trio,
    trio_directory,
    trio_is_complete,
)
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition
from rpg_economy.llm.client import ProviderError

DEFAULT_WORKERS = 8

#: Whole-trio retries, a layer above the client's per-request retries
#: (llm.max_transient_retries): a trio whose provider failure outlasted those is
#: re-run from scratch after a longer wait. Rate limits and transient 5xx are the
#: retryable class; a 400 is a bug in the request and retrying it just spends
#: latency.
MAX_PROVIDER_ATTEMPTS = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_CAP_SECONDS = 60.0


def is_retryable(error: BaseException) -> bool:
    """Whether a failed trio is worth re-running after a wait.

    Only a provider failure the client already classified as transient: a rate
    limit or an outage that outlasted its own retries. Anything else (a refused
    key, an exhausted quota, a bug) fails the trio at once.
    """
    return isinstance(error, ProviderError) and error.transient


def backoff_delay(attempt: int, *, rng: random.Random) -> float:
    """Exponential backoff with full jitter.

    Jittered because eight workers that all back off for exactly four seconds
    return in lockstep and re-trigger the same rate limit together.
    """
    ceiling = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
    return rng.uniform(0, ceiling)


@dataclass
class TrioOutcome:
    """What happened to one planned trio."""

    trio: PlannedTrio
    status: str  # "completed" | "skipped" | "failed" | "interrupted"
    detail: str = ""
    seconds: float = 0.0
    attempts: int = 1

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.trio.to_payload(),
            "status": self.status,
            "detail": self.detail,
            "seconds": round(self.seconds, 1),
            "attempts": self.attempts,
        }


@dataclass
class ParallelResult:
    directory: Path
    workers: int
    outcomes: list[TrioOutcome] = field(default_factory=list)
    wall_seconds: float = 0.0

    def by_status(self, status: str) -> list[TrioOutcome]:
        return [o for o in self.outcomes if o.status == status]

    @property
    def failed(self) -> list[TrioOutcome]:
        return self.by_status("failed")

    @property
    def interrupted(self) -> list[TrioOutcome]:
        """Trios stopped by Ctrl-C: not failures, and resumable."""
        return self.by_status("interrupted")

    def to_payload(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "workers": self.workers,
            "wall_seconds": round(self.wall_seconds, 1),
            "completed": len(self.by_status("completed")),
            "skipped": len(self.by_status("skipped")),
            "failed": len(self.failed),
            "interrupted": len(self.interrupted),
            "outcomes": [o.to_payload() for o in self.outcomes],
        }


def _run_one(
    base_config: SimulationConfig,
    trio: PlannedTrio,
    destination: Path,
    *,
    rng_seed: int,
    resume: bool = True,
    interrupt: Interrupt | None = None,
    phase: CampaignPhase | str | None = None,
    concurrent: bool = False,
) -> TrioOutcome:
    """Execute one trio, retrying the provider's transient failures.

    ``phase`` is the campaign's (``cell_config``); without it the configured
    phase is kept.

    A retry re-runs only the condition that failed: the conditions that
    finished in an earlier attempt are kept (``execute_trio``), so a transient
    failure in C costs a second C and never a second A and B. The failed
    attempt's partial log is set aside, not deleted, and counted as spend by
    the billing verification. Once ``interrupt`` asks to stop, no further
    attempt or condition starts, and the trio is reported ``interrupted``:
    the conditions it finished are kept for a resume.
    """
    from rpg_economy.runner import RunInterrupted

    started = time.monotonic()
    rng = random.Random(rng_seed)
    config = cell_config(base_config, trio, phase)

    def interrupted(attempts: int, detail: str = "") -> TrioOutcome:
        return TrioOutcome(
            trio=trio,
            status="interrupted",
            detail=detail,
            seconds=time.monotonic() - started,
            attempts=attempts,
        )

    for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
        if interrupt is not None and interrupt.should_stop():
            return interrupted(attempt - 1)
        try:
            # --no-resume discards earlier logs once, on the first attempt;
            # later attempts always keep what this one finished.
            ran = execute_trio(
                config,
                destination,
                reuse=resume or attempt > 1,
                should_stop=interrupt.should_stop if interrupt is not None else None,
                should_abort=interrupt.should_abort if interrupt is not None else None,
                concurrent=concurrent,
                only=trio.only,
            )
            from rpg_economy.integrity import has_integrity_marker

            return TrioOutcome(
                trio=trio,
                status="partial"
                if trio.only is not None and not has_integrity_marker(destination)
                else "completed",
                detail="" if ran else "verified existing logs",
                seconds=time.monotonic() - started,
                attempts=attempt,
            )
        except RunInterrupted as exc:
            return interrupted(attempt, str(exc)[:300])
        except Exception as exc:  # classified below, never swallowed
            if interrupt is not None and interrupt.should_stop():
                # Whatever failed, no retry starts after a Ctrl-C.
                return interrupted(attempt, f"{type(exc).__name__}: {exc}"[:300])
            if not is_retryable(exc) or attempt == MAX_PROVIDER_ATTEMPTS:
                return TrioOutcome(
                    trio=trio,
                    status="failed",
                    detail=f"{type(exc).__name__}: {exc}"[:300],
                    seconds=time.monotonic() - started,
                    attempts=attempt,
                )
            delay = backoff_delay(attempt, rng=rng)
            if interrupt is not None:
                # Woken at once by an interruption instead of sleeping it out.
                interrupt.wait(delay)
            else:
                time.sleep(delay)

    return TrioOutcome(trio=trio, status="failed", detail="exhausted attempts")


def run_campaign_parallel(
    base_config: SimulationConfig,
    plan: CampaignPlan,
    directory: Path | str,
    *,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
    progress: Callable[[TrioOutcome, int, int], None] | None = None,
    interrupt: Interrupt | None = None,
) -> ParallelResult:
    """Run every trio in a plan, up to ``workers`` at a time.

    Threads rather than processes: the work is almost entirely waiting on the
    provider, and threads keep the log writers and file handles in one address
    space where their append-only guarantees still hold.

    ``interrupt`` is the caller's two-level Ctrl-C (see ``Interrupt``). Trios
    that had not started when it asked to stop are reported ``interrupted``
    with no attempt, and without a progress line.
    """
    interrupt = interrupt if interrupt is not None else Interrupt()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    workers_requested = workers
    workers = max(1, min(workers, len(plan.trios) or 1))

    pending: list[PlannedTrio] = []
    result = ParallelResult(directory=root, workers=workers)

    for trio in plan.trios:
        destination = trio_directory(root, trio)
        cell = plan.cell(base_config, trio)
        if resume and trio_is_complete(destination, cell):
            result.outcomes.append(TrioOutcome(trio=trio, status="skipped"))
        else:
            pending.append(trio)

    started = time.monotonic()
    total = len(pending)
    if total:
        # Created explicitly, not in a ``with``: leaving a ``with`` waits for
        # every queued trio, so an interrupted real batch would keep running
        # (and billing) all of them. On any exception, KeyboardInterrupt
        # included, the queue is cancelled and only the trios already running
        # are waited for; the interrupt keeps those from starting another
        # attempt or condition.
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {
                pool.submit(
                    _run_one,
                    base_config,
                    trio,
                    trio_directory(root, trio),
                    resume=resume,
                    # Derived from the trio, so a resumed campaign backs off the
                    # same way a fresh one would.
                    # crc32, not hash(): hash() of a string changes between
                    # processes, so a resumed campaign would back off differently.
                    rng_seed=zlib.crc32(
                        f"{trio.seed}:{trio.scenario.value}:{trio.experiment.value}".encode()
                    ),
                    interrupt=interrupt,
                    phase=plan.phase,
                    # Spare workers run a trio's conditions side by side: a
                    # one-trio check then takes as long as A, not A + B + C.
                    # A full stage keeps one stream per trio, as rate limits
                    # were sized for.
                    concurrent=len(pending) * 3 <= workers_requested,
                ): trio
                for trio in pending
            }
            done = 0
            for future in as_completed(futures):
                outcome = future.result()
                result.outcomes.append(outcome)
                if outcome.status == "interrupted" and outcome.attempts == 0:
                    continue  # never started: nothing to report as progress
                done += 1
                if progress is not None:
                    progress(outcome, done, total)
        except BaseException:
            interrupt.request_stop()
            join_pool(pool, interrupt, cancel=True)
            raise
        join_pool(pool, interrupt)

    result.wall_seconds = time.monotonic() - started
    return result


def join_pool(pool: ThreadPoolExecutor, interrupt: Interrupt, *, cancel: bool = False) -> None:
    """Wait for every worker of ``pool``, however many Ctrl-C arrive meanwhile.

    A KeyboardInterrupt while joining would otherwise return early, and the
    caller would release the output directory's lock with workers still
    writing (and billing) under it. Each one escalates the interrupt instead,
    and the join resumes.
    """
    while True:
        try:
            pool.shutdown(wait=True, cancel_futures=cancel)
            return
        except KeyboardInterrupt:
            interrupt.escalate()
            cancel = True


def run_trio_parallel(
    base_config: SimulationConfig,
    directory: Path | str,
    *,
    seed: int | None = None,
    interrupt: Interrupt | None = None,
) -> dict[Condition, Any]:
    """Run one trio's three conditions concurrently, then verify it.

    The three conditions of a trio are as independent of one another as two
    different seeds are -- they share only the configuration and the seed, both
    read-only -- so the same isolation argument that licenses campaign-level
    parallelism licenses this. It matters because a trio's critical path is
    otherwise A + B + C in series, and A alone is the longest run.

    The trio integrity gate (design v1.0 s12) still runs, on the same results, after all three
    have finished: a trio that fails it is not a trio, however it was scheduled.

    All three conditions start together, so a first Ctrl-C (``interrupt``
    asking to stop) lets them finish and the trio is verified as usual. A
    second one aborts them at their next macro-cycle; the trio is then not
    verified, and ``RunInterrupted`` is raised once all three have returned.
    """
    from rpg_economy.integrity import verify_trio, write_integrity_marker
    from rpg_economy.runner import INTERRUPT_STOP, RunInterrupted, run_condition

    interrupt = interrupt if interrupt is not None else Interrupt()
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    if interrupt.should_stop():
        raise RunInterrupted(reason=INTERRUPT_STOP)

    runs: dict[Condition, Any] = {}
    errors: list[BaseException] = []
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        futures = {
            pool.submit(
                run_condition,
                base_config.for_condition(condition, seed=seed),
                destination / f"{condition.value}.jsonl",
                should_abort=interrupt.should_abort,
            ): condition
            for condition in (Condition.A, Condition.B, Condition.C)
        }
        for future in as_completed(futures):
            try:
                runs[futures[future]] = future.result()
            except Exception as exc:  # re-raised once every condition returned
                errors.append(exc)
    finally:
        join_pool(pool, interrupt)

    if errors:
        # An abort outranks the failures it may have caused in the others.
        for error in errors:
            if isinstance(error, RunInterrupted):
                error.completed = dict(runs)
                raise error
        raise errors[0]

    write_integrity_marker(destination, verify_trio(runs))
    return runs


def suggested_workers() -> int:
    """A default that respects the machine without pretending to know the provider's limits."""
    return min(DEFAULT_WORKERS, max(1, (os.cpu_count() or 4)))
