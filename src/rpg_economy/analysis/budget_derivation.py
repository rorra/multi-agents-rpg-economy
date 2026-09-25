"""Deriving the Experiment 2 token budget from the pilot (design v1.0 s8.4).

The rule is fixed by the design and deliberately neutral: the budget is the
pooled median of total token consumption across the pilot runs of A, B and C
in the primary scenario under Experiment 1. Pooled, so no condition is
privileged; median, so one runaway run cannot set the budget for everyone.

Computed by this script from the pilot logs, recorded in the
configuration, and never hand-tuned (s8.4, s15).

Per-run totals are read from each log's authoritative figure -- the sum of what
the provider reported -- not from bucket attribution (s8.4).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from rpg_economy.analysis.errors import AnalysisError
from rpg_economy.analysis.logchecks import run_end_ticks, with_run_phase
from rpg_economy.config import config_hash
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime, Scenario
from rpg_economy.logging_ import (
    EventType,
    log_is_finished,
    read_events,
    read_metadata,
    run_logs,
)

#: The primary scenario, whose pilot runs alone determine the budget (s8.4).
PRIMARY_SCENARIO = Scenario.RESOURCE_SCARCITY


#: The run phase a log must record to set the budget: only the pilot's runs do.
PILOT_PHASE = "pilot"


class BudgetDerivationError(AnalysisError):
    """Raised when the pilot logs cannot support a budget derivation.

    Carries a catalog ``key`` and its ``params``, like every analysis error.
    """


@dataclass
class PilotRun:
    """One pilot run's contribution."""

    path: Path
    condition: Condition
    seed: int
    total_tokens: int


@dataclass
class BudgetDerivation:
    """The derived budget and the evidence behind it."""

    budget: int
    runs: list[PilotRun] = field(default_factory=list)

    @property
    def per_condition(self) -> dict[str, list[int]]:
        totals: dict[str, list[int]] = {}
        for run in self.runs:
            totals.setdefault(run.condition.value, []).append(run.total_tokens)
        return {k: sorted(v) for k, v in sorted(totals.items())}

    def to_payload(self) -> dict:
        return {
            "budget": self.budget,
            "rule": "pooled median of per-run provider totals, primary scenario, "
            "experiment_1 pilot runs of A, B and C",
            "scenario": PRIMARY_SCENARIO.value,
            "runs": len(self.runs),
            "seeds": sorted({run.seed for run in self.runs}),
            "per_condition_totals": self.per_condition,
            "pooled_totals": sorted(run.total_tokens for run in self.runs),
        }


def run_total_tokens(path: Path) -> int:
    """The authoritative token total recorded by a completed run."""
    for event in read_events(path):
        if event.type is EventType.TOKENS_CONSUMED:
            payload = event.payload
            if "authoritative_total" not in payload:
                raise BudgetDerivationError(
                    f"{path}: log predates authoritative token accounting; the "
                    f"budget must be derived from provider totals (design v1.0 s8.4)",
                    key="error.budget.no_authoritative_total",
                    path=str(path),
                )
            return int(payload["authoritative_total"])
    raise BudgetDerivationError(
        f"{path}: no token accounting recorded; run incomplete?",
        key="error.budget.no_token_accounting",
        path=str(path),
    )


def collect_pilot_runs(
    paths: Iterable[Path | str],
    *,
    config: SimulationConfig | None = None,
    allow_mock: bool = False,
) -> list[PilotRun]:
    """Select the pilot runs that the budget rule admits, refusing any that do not fit.

    Runs of another scenario or experiment are skipped: they are legitimately
    present in a pilot directory. Anything else that does not fit is an error,
    never silently dropped, because a wrong budget carries into Experiment 2:

    * an unfinished run, a mock run (unless ``allow_mock``, for tests);
    * with ``config``: a seed outside the pilot set, a length other than the
      full horizon (the calibration stage runs shorter), a run whose logged
      config hash is not what ``config`` derives for it, or a run whose
      recorded phase is not ``pilot`` (a final, ad hoc or older unmarked run);
    * with ``config``: a run whose trio directory carries no integrity marker
      for the logs it holds now (the trio never passed the gate, or a log
      changed after it did), or whose prompts or interpreter are not the ones
      the final runs will use, or whose code digest is not the current
      code's. The budget is spent by the final runs, so it must come from
      runs of the same code.
    """
    admitted: list[PilotRun] = []
    gated: dict[Path, bool] = {}
    for candidate in paths:
        path = Path(candidate)
        metadata = read_metadata(path)
        if metadata.scenario is not PRIMARY_SCENARIO:
            continue
        if metadata.experiment is not ExperimentRegime.EQUAL_ACTIVITY:
            continue
        if not log_is_finished(path):
            raise BudgetDerivationError(
                f"{path}: the run did not finish", key="error.budget.unfinished", path=str(path)
            )
        if metadata.llm_mock and not allow_mock:
            raise BudgetDerivationError(
                f"{path}: a mock run; the budget is derived from real spend (design v1.0 s8.4)",
                key="error.budget.mock_run",
                path=str(path),
            )
        if not metadata.llm_mock:
            _check_billed(path)
        if config is not None:
            _check_against_config(path, metadata, config)
            _check_integrity(path, gated)
            _check_code_version(path, metadata)
        admitted.append(
            PilotRun(
                path=path,
                condition=metadata.condition,
                seed=metadata.seed,
                total_tokens=run_total_tokens(path),
            )
        )
    return admitted


def _check_billed(path: Path) -> None:
    """A real run's total must be what the provider billed, call for call.

    A call without provider usage falls back to a local estimate; a budget
    derived partly from estimates would not be the pilot's real spend.
    """
    for event in read_events(path):
        if event.type is EventType.TOKENS_CONSUMED:
            payload = event.payload
            billed = payload.get("billed_total")
            if billed is None or int(billed) != int(payload.get("authoritative_total", -1)):
                raise BudgetDerivationError(
                    f"{path}: not every call was billed by the provider; the budget "
                    f"must come from billed spend (design v1.0 s8.4)",
                    key="error.budget.unbilled",
                    path=str(path),
                )
            return


def _check_against_config(path: Path, metadata, config: SimulationConfig) -> None:
    if metadata.seed not in config.seeds.pilot:
        raise BudgetDerivationError(
            f"{path}: seed {metadata.seed} is not a pilot seed",
            key="error.budget.not_pilot_seed",
            path=str(path),
            seed=metadata.seed,
        )
    ticks = run_end_ticks(path)
    if ticks != config.time.horizon_ticks:
        raise BudgetDerivationError(
            f"{path}: ran {ticks} ticks, not the full {config.time.horizon_ticks}; the "
            f"budget is derived at the horizon the final runs use (design v1.0 s11.4)",
            key="error.budget.short_horizon",
            path=str(path),
            ticks=str(ticks),
            horizon=config.time.horizon_ticks,
        )
    derived = config.for_cell(
        scenario=metadata.scenario, experiment=metadata.experiment, seed=metadata.seed
    ).for_condition(metadata.condition, seed=metadata.seed)
    derived = with_run_phase(derived, getattr(metadata, "phase", None))
    if config_hash(derived) != metadata.config_hash:
        raise BudgetDerivationError(
            f"{path}: the run used a different configuration",
            key="error.budget.config_mismatch",
            path=str(path),
        )
    # A final or ad hoc run on a pilot seed, or a log from before the phase was
    # recorded, is not part of the pilot the rule pools (s8.4).
    phase = getattr(metadata, "phase", None)
    if phase != PILOT_PHASE:
        raise BudgetDerivationError(
            f"{path}: its recorded phase is {phase or 'none'}, not {PILOT_PHASE}; only "
            f"pilot runs set the budget (design v1.0 s8.4)",
            key="error.budget.not_pilot_phase",
            path=str(path),
            phase=str(phase or "-"),
        )


def _check_integrity(path: Path, gated: dict[Path, bool]) -> None:
    """Only trios that passed the integrity gate, on the exact logs they hold."""
    from rpg_economy.integrity import has_integrity_marker

    directory = path.parent
    if directory not in gated:
        gated[directory] = has_integrity_marker(directory)
    if not gated[directory]:
        raise BudgetDerivationError(
            f"{path}: its trio ({directory}) has not passed the integrity gate, or a "
            f"log changed after it did; only verified trios set the budget",
            key="error.budget.integrity",
            path=str(path),
            directory=str(directory),
        )


def current_prompt_hashes() -> dict[str, str]:
    """The prompt hashes a run of the current code records in its metadata."""
    from rpg_economy.agents.prompts import PROMPT_VERSION
    from rpg_economy.orchestration.director import prompt_hashes

    return {**prompt_hashes(), "version": PROMPT_VERSION}


def _check_code_version(path: Path, metadata) -> None:
    """The run's code, prompts and interpreter must be the ones in use now."""
    from rpg_economy.logging_ import metadata as run_metadata

    python = run_metadata.current_python_version()
    if metadata.python_version != python:
        raise BudgetDerivationError(
            f"{path}: ran on Python {metadata.python_version}, not {python}; the budget "
            f"is derived in the environment the final runs use",
            key="error.budget.python_version",
            path=str(path),
            recorded=metadata.python_version,
            current=python,
        )
    # The whole package, not only the prompts: an engine or accounting change
    # moves token spend as surely as a prompt edit does.
    digest = run_metadata.code_digest()
    recorded_digest = getattr(metadata, "code_digest", None)
    if recorded_digest != digest:
        raise BudgetDerivationError(
            f"{path}: its code digest {recorded_digest or 'none recorded'} is not the "
            f"current code's {digest}; a budget measured on other code does not fit the "
            f"final runs",
            key="error.budget.code_digest",
            path=str(path),
            recorded=recorded_digest or "-",
            current=digest,
        )
    expected = current_prompt_hashes()
    recorded = dict(metadata.prompt_hashes or {})
    if recorded != expected:
        changed = sorted(
            name
            for name in set(expected) | set(recorded)
            if expected.get(name) != recorded.get(name)
        )
        raise BudgetDerivationError(
            f"{path}: its prompts differ from the current code ({', '.join(changed)}); "
            f"a budget measured on other prompts does not fit the final runs",
            key="error.budget.prompts_differ",
            path=str(path),
            prompts=", ".join(changed),
        )


def derive_budget(
    paths: Iterable[Path | str],
    *,
    config: SimulationConfig | None = None,
    allow_mock: bool = False,
) -> BudgetDerivation:
    """Derive the Experiment 2 budget from a set of pilot logs.

    With ``config``, the pilot must be whole: every configured pilot seed of
    the primary scenario under Experiment 1, each with A, B and C at the full
    horizon. A partial pilot is refused, naming the seeds it lacks: the pooled
    median of whichever runs happened to finish is not the pre-registered rule
    (s8.4), and which runs finish first is not a neutral selection.
    """
    runs = collect_pilot_runs(paths, config=config, allow_mock=allow_mock)
    if config is not None:
        complete = {
            seed
            for seed in {run.seed for run in runs}
            if {run.condition for run in runs if run.seed == seed} == set(Condition)
        }
        missing_seeds = sorted(set(config.seeds.pilot) - complete)
        if missing_seeds:
            raise BudgetDerivationError(
                f"the pilot is incomplete: pilot seeds {missing_seeds} have no complete "
                f"A/B/C trio of {PRIMARY_SCENARIO.value} under experiment_1 at the full "
                f"horizon; the budget is the pooled median of the whole pilot (design v1.0 s8.4)",
                key="error.budget.pilot_incomplete",
                seeds=", ".join(map(str, missing_seeds)),
                scenario=PRIMARY_SCENARIO.value,
            )

    if not runs:
        raise BudgetDerivationError(
            f"no admissible pilot runs: the budget is derived only from "
            f"{PRIMARY_SCENARIO.value} runs under experiment_1 (design v1.0 s8.4)",
            key="error.budget.no_runs",
            scenario=PRIMARY_SCENARIO.value,
        )

    present = {run.condition for run in runs}
    missing = set(Condition) - present
    if missing:
        raise BudgetDerivationError(
            f"the pooled median must not privilege a condition, but no runs were "
            f"found for {sorted(c.value for c in missing)} (design v1.0 s8.4)",
            key="error.budget.condition_missing",
            conditions=", ".join(sorted(c.value for c in missing)),
        )

    seen: dict[tuple[Condition, int], Path] = {}
    for run in runs:
        key = (run.condition, run.seed)
        if key in seen:
            raise BudgetDerivationError(
                f"seed {run.seed} condition {run.condition.value} appears twice "
                f"({seen[key]} and {run.path})",
                key="error.budget.duplicate_run",
                seed=run.seed,
                condition=run.condition.value,
                first=str(seen[key]),
                second=str(run.path),
            )
        seen[key] = run.path
    for seed in sorted({run.seed for run in runs}):
        have = {run.condition for run in runs if run.seed == seed}
        if have != set(Condition):
            raise BudgetDerivationError(
                f"seed {seed} has {sorted(c.value for c in have)}; every seed must "
                f"contribute A, B and C, or the pool would favour a condition",
                key="error.budget.seed_incomplete",
                seed=seed,
                conditions=", ".join(sorted(c.value for c in have)),
            )

    # Pooled: every run contributes one value, regardless of its condition.
    # Rounded, not truncated: an even count gives a half-token median.
    median = Decimal(str(statistics.median(run.total_tokens for run in runs)))
    budget = int(median.quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return BudgetDerivation(budget=budget, runs=runs)


def find_logs(directory: Path | str) -> list[Path]:
    """Every JSONL run log under ``directory``."""
    return run_logs(directory)
