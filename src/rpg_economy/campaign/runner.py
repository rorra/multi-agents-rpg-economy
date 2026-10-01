"""Executing a planned campaign (IMPLEMENTATION.md phase 7).

One trio at a time, each written to its own directory, each verified by the
trio integrity gate (design v1.0 s12) before it counts. A campaign is resumable:
a trio whose logs finished and passed the gate is skipped rather than re-run,
and a trio interrupted part-way keeps the conditions that finished, so an
interrupted real-provider batch does not restart from zero.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.campaign.plan import CampaignPlan, PlannedTrio
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition
from rpg_economy.logging_ import log_is_finished
from rpg_economy.logging_.writer import SET_ASIDE_PREFIX

#: Called after each trio with (index, total, trio). Progress reporting only --
#: it can observe a campaign but never steer one.
ProgressHook = Callable[[int, int, PlannedTrio], None]


@dataclass
class CampaignResult:
    """What a campaign execution produced."""

    directory: Path
    completed: list[PlannedTrio] = field(default_factory=list)
    skipped: list[PlannedTrio] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "completed": [trio.to_payload() for trio in self.completed],
            "skipped": [trio.to_payload() for trio in self.skipped],
        }


def trio_directory(root: Path, trio: PlannedTrio) -> Path:
    """Where one trio's three logs live.

    Scenario and experiment are in the path because the same seed runs in
    several cells, and a flat layout would have them overwrite each other.
    """
    return Path(root) / trio.scenario.value / trio.experiment.value / f"seed_{trio.seed}"


def current_prompt_hashes() -> dict[str, str]:
    """The prompt hashes a run started now records in its metadata.

    Built as ``rpg_economy.runner`` builds them: every agent and Director
    template, the global view format included, plus the prompt version.
    """
    from rpg_economy.agents.prompts import PROMPT_VERSION
    from rpg_economy.orchestration.director import prompt_hashes

    return {**prompt_hashes(), "version": PROMPT_VERSION}


def condition_log_matches(
    directory: Path, condition: Condition, config: SimulationConfig | None = None
) -> bool:
    """Whether one condition's log finished cleanly, and ran this cell if given.

    With ``config`` (the trio's cell configuration), the log must have run
    exactly that configuration for this condition and the cell's own seed, in
    the same mode (mock or real), under the code running now: the same prompt
    templates and global view format, the same interpreter release, and the
    same package source (``code_digest``; a log that predates the digest has
    none, and is not vouched for). A
    finished log from another stage or setup in the same directory -- a 120-tick
    calibration run where a 340-tick one is planned -- is not this trio's; nor is
    one written before a prompt changed, which kept beside fresh runs would pair
    two versions of the code in one trio.
    """
    from rpg_economy import logging_
    from rpg_economy.config import config_hash
    from rpg_economy.logging_ import current_python_version, read_metadata

    path = Path(directory) / f"{condition.value}.jsonl"
    if not log_is_finished(path):
        return False
    if config is None:
        return True
    try:
        metadata = read_metadata(path)
    except Exception:  # an unreadable log is not a finished run of this cell
        return False
    # The cell's seed, not the one the log reports: a log of another seed
    # would otherwise vouch for itself.
    expected = config_hash(config.for_condition(condition, seed=config.run.seed))
    prompts = current_prompt_hashes()
    return (
        metadata.config_hash == expected
        and metadata.seed == config.run.seed
        and metadata.llm_mock == config.llm.mock
        # The whole map, so the global view format is compared with the rest.
        and metadata.prompt_hashes == prompts
        and metadata.python_version == current_python_version()
        # Last: it reads the whole package's source.
        and metadata.code_digest is not None
        and metadata.code_digest == logging_.code_digest()
    )


def trio_is_complete(directory: Path, config: SimulationConfig | None = None) -> bool:
    """A trio counts as done only when all three logs finished and passed the gate.

    Each log must satisfy ``condition_log_matches``, and the trio directory
    must hold the integrity marker written when the three logs passed
    ``verify_trio`` (design v1.0 s12) -- on exactly the bytes they hold now. A
    trio that finished but failed the gate is therefore never skipped.
    """
    from rpg_economy.integrity import has_integrity_marker

    for condition in (Condition.A, Condition.B, Condition.C):
        if not condition_log_matches(Path(directory), condition, config):
            return False
    return has_integrity_marker(directory)


def pending_trios(
    base_config: SimulationConfig, plan: CampaignPlan, directory: Path | str, *, resume: bool
) -> CampaignPlan:
    """The part of ``plan`` that still has to run under ``directory``.

    What a real-provider batch is priced and confirmed on: a resumed campaign
    pays only for the trios it will actually run, never again for finished ones.
    """
    import dataclasses

    if not resume:
        return plan
    root = Path(directory)
    remaining = [
        trio
        for trio in plan.trios
        if not trio_is_complete(
            trio_directory(root, trio),
            plan.cell(base_config, trio),
        )
    ]
    return dataclasses.replace(plan, trios=remaining, warnings=list(plan.warnings))


def set_aside_previous(
    directory: Path, conditions: Iterable[Condition] | None = None
) -> Path | None:
    """Move a trio's existing logs out of the way before they are run again.

    Used for an unfinished run and for a deliberate re-run. Moved, never
    deleted: a partial real run was billed, and its log is the only record of
    what that spend bought. ``run_logs`` never reads the set-aside copies.
    With ``conditions``, only those conditions' logs move; the others stay
    where they are. The trio's integrity marker is dropped either way, since
    it no longer describes the logs in place. Returns where they went, if
    anywhere.
    """
    from rpg_economy.integrity import clear_integrity_marker

    directory = Path(directory)
    selected = tuple(conditions) if conditions is not None else tuple(Condition)
    logs = [
        directory / f"{condition.value}.jsonl"
        for condition in (Condition.A, Condition.B, Condition.C)
        if condition in selected and (directory / f"{condition.value}.jsonl").exists()
    ]
    if not logs:
        return None
    clear_integrity_marker(directory)
    index = 1
    while (directory / f"{SET_ASIDE_PREFIX}{index}").exists():
        index += 1
    target = directory / f"{SET_ASIDE_PREFIX}{index}"
    target.mkdir(parents=True)
    for path in logs:
        path.rename(target / path.name)
    return target


def execute_trio(
    config: SimulationConfig,
    directory: Path | str,
    *,
    reuse: bool = True,
    should_stop: Callable[[], bool] | None = None,
    should_abort: Callable[[], bool] | None = None,
    concurrent: bool = False,
    only: Condition | None = None,
) -> tuple[Condition, ...]:
    """Run whatever a trio still lacks, verify it, and mark it verified.

    With ``only``, just that condition runs if it is missing; the trio is
    verified and marked only once all three conditions have finished logs.

    With ``concurrent`` the missing conditions run at the same time, as
    ``run_trio_parallel`` does: a trio's wall time is then its slowest
    condition (A) instead of A + B + C. A condition that fails does not stop
    the others; the first failure is raised once all have returned, and the
    finished logs are kept for a retry.

    ``config`` is the trio's cell configuration. With ``reuse``, a condition
    whose log already finished for this cell is kept, and only the others run:
    given the seed, each condition of a trio is as independent of the others as
    two seeds are (the same argument ``run_trio_parallel`` rests on), so a
    transient provider failure in C re-runs C, and never re-pays A and B.
    Without ``reuse`` every existing log is set aside and all three run.

    Verification always covers all three logs. A trio assembled partly from
    earlier runs that fails the gate is set aside whole and run again, once;
    a freshly run trio that fails it raises ``TrioIntegrityError``. Returns the
    conditions actually run.

    ``should_stop`` is asked before each condition starts: once it returns
    True no further condition begins, and ``RunInterrupted`` (reason
    ``stop``) is raised with the finished conditions' logs left in place, so a
    resume runs only the missing ones. ``should_abort`` is handed to each run,
    which asks it once per macro-cycle and raises ``RunInterrupted`` (reason
    ``abort``), leaving that run's log unfinished. An interrupted trio is
    neither verified nor marked.
    """
    from rpg_economy.integrity import (
        TrioIntegrityError,
        has_integrity_marker,
        verify_trio_logs,
        write_integrity_marker,
    )
    from rpg_economy.logging_ import LogError
    from rpg_economy.runner import INTERRUPT_STOP, RunInterrupted, run_condition

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    conditions = (Condition.A, Condition.B, Condition.C)
    kept = {c for c in conditions if reuse and condition_log_matches(destination, c, config)}
    if len(kept) == len(conditions) and has_integrity_marker(destination):
        return ()

    def stop_requested() -> None:
        if should_stop is not None and should_stop():
            raise RunInterrupted(reason=INTERRUPT_STOP)

    def run(missing: tuple[Condition, ...]) -> None:
        stop_requested()
        set_aside_previous(destination, missing)
        if concurrent and len(missing) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(missing)) as pool:
                futures = [
                    pool.submit(
                        run_condition,
                        config.for_condition(condition, seed=config.run.seed),
                        destination / f"{condition.value}.jsonl",
                        should_abort=should_abort,
                    )
                    for condition in missing
                ]
            errors = [f.exception() for f in futures if f.exception() is not None]
            if errors:
                raise errors[0]
            return
        for condition in missing:
            # Between conditions, never inside one: a condition that started
            # is paid for, so it runs to its end unless aborted.
            stop_requested()
            run_condition(
                config.for_condition(condition, seed=config.run.seed),
                destination / f"{condition.value}.jsonl",
                should_abort=should_abort,
            )

    ran = tuple(c for c in conditions if c not in kept and (only is None or c == only))
    run(ran)
    if only is not None and not all(
        condition_log_matches(destination, c, config) for c in conditions
    ):
        return ran  # the trio is not whole yet: nothing to verify
    try:
        report = verify_trio_logs(destination)
    except TrioIntegrityError, LogError:
        if not kept:
            raise
        # Logs that finished separately do not make a trio if together they
        # fail the gate, and a kept log that cannot be read whole is no log at
        # all; none of them is evidence, so the trio runs again whole.
        ran = conditions
        run(ran)
        report = verify_trio_logs(destination)
    write_integrity_marker(destination, report)
    return ran


def run_campaign(
    base_config: SimulationConfig,
    plan: CampaignPlan,
    directory: Path | str,
    *,
    progress: ProgressHook | None = None,
    resume: bool = True,
) -> CampaignResult:
    """Execute every trio in a plan.

    The scenario and experiment come from the plan, the condition from
    ``execute_trio``; nothing about a run's behaviour is decided here.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    result = CampaignResult(directory=root)

    for index, trio in enumerate(plan.trios, start=1):
        destination = trio_directory(root, trio)
        cell = plan.cell(base_config, trio)
        if resume and trio_is_complete(destination, cell):
            result.skipped.append(trio)
        else:
            # Runs each missing condition and verifies the trio against the
            # integrity gate before marking it complete.
            execute_trio(cell, destination, reuse=resume)
            result.completed.append(trio)
        if progress is not None:
            progress(index, len(plan.trios), trio)

    return result
