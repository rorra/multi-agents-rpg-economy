"""From a directory of run logs to one results set per scenario and experiment.

The analysis reads logs and nothing else (IMPLEMENTATION.md section 4.12). Logs
are first selected, then checked:

* selected: only final seeds from runs recorded as the final phase enter the
  inference (design s12: pilot seeds are never used for inference). With
  ``include_pilot``, pilot seeds and pilot-phase runs enter too, for
  exploration. Everything else -- another seed, an ad hoc run, a log from
  before the phase was recorded -- is left out and listed with its reason;
* checked, every selected log: it must be finished, at the full horizon, from
  a real model unless mock runs are allowed, and it must have run the
  configuration it claims (by hash);
* per trio: it must have passed the integrity gate on the exact logs it holds,
  and the common active horizon the gate recorded must be the one computed
  here (design s12, s8.4). A trio that fails is rejected and reported;
* per cell: its runs must share one code version (code digest, prompt hashes,
  interpreter, global-view format). A cell that mixes them is refused and
  reported, since it would compare versions, not architectures.
"""

from __future__ import annotations

import json
import statistics
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.analysis.errors import AnalysisError
from rpg_economy.analysis.logchecks import run_end_ticks, with_run_phase
from rpg_economy.analysis.results import Results, build_results
from rpg_economy.config import config_hash
from rpg_economy.config.hashing import canonical_json
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime
from rpg_economy.llm.budget import common_active_horizon
from rpg_economy.logging_ import (
    EventType,
    LogError,
    RunMetadata,
    log_is_finished,
    read_events,
    read_metadata,
    run_logs,
)
from rpg_economy.metrics.run_metrics import (
    RunLog,
    compute_run_metrics,
    target_series,
)
from rpg_economy.metrics.targets import primary_target

__all__ = [
    "Analysis",
    "AnalysisError",
    "Cell",
    "Collected",
    "IncompleteTrioWarning",
    "RejectedTrioWarning",
    "analyse",
    "analyse_cell",
    "collect_cells",
    "complete_trios",
    "gate_trios",
    "incomplete_seeds",
    "mixed_versions",
]

#: The run phase whose logs enter the inference, and the one ``include_pilot``
#: admits besides it.
FINAL_PHASE = "final"
PILOT_PHASE = "pilot"

#: Metadata fields every run of a cell must share: the same simulation code,
#: prompts, interpreter and global-view format. A cell whose trios were run by
#: different versions would compare those versions, not architectures.
VERSION_FIELDS = ("code_digest", "prompt_hashes", "python_version", "global_view_version")


class IncompleteTrioWarning(UserWarning):
    """A seed lacking one of A, B and C, left out of the analysis."""


class RejectedTrioWarning(IncompleteTrioWarning):
    """A trio or a cell refused by the integrity or version checks, left out.

    A subclass of :class:`IncompleteTrioWarning`, so a caller that reports
    left-out seeds itself silences both with one filter.
    """


@dataclass
class Cell:
    """The runs of one scenario under one experiment."""

    scenario: str
    experiment: str
    logs: dict[tuple[str, int], Path] = field(default_factory=dict)
    #: The metadata of each run in ``logs``, read once when it was collected.
    metadata: dict[tuple[str, int], RunMetadata] = field(default_factory=dict)


@dataclass
class Collected:
    """The selected logs grouped into cells, and every log left out, with why."""

    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    #: Seeds outside the analysed seed set.
    skipped_seeds: list[int] = field(default_factory=list)
    #: One entry per log left out before any check (seed or run phase).
    skipped_logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Analysis:
    """One results set per cell, plus what was left out and why."""

    results: dict[tuple[str, str], Results] = field(default_factory=dict)
    price_series: dict[tuple[str, str], dict[str, list[float | None]]] = field(default_factory=dict)
    skipped_seeds: list[int] = field(default_factory=list)
    #: Every log left out before the checks, with its seed, phase and reason.
    skipped_logs: list[dict[str, Any]] = field(default_factory=list)
    #: Per cell, the seeds left out because their A/B/C trio was incomplete.
    incomplete_trios: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    #: Per cell, the complete trios refused by the integrity gate, with why.
    rejected_trios: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    #: Cells refused whole because their runs mix code versions, with the
    #: metadata fields that differ.
    refused_cells: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    #: Cells with no complete trio at all, left out whole, with the conditions
    #: their logs did cover. The other cells are still analysed.
    skipped_cells: dict[tuple[str, str], list[str]] = field(default_factory=dict)


def exhaustion_tick(path: Path) -> int | None:
    """The tick the token budget ran out in this run, if it did."""
    for event in read_events(path):
        if event.type is EventType.BUDGET_EXHAUSTED:
            return event.tick
    return None


def _skip_reason(seed_allowed: bool, phase: str | None, admitted: set[str]) -> str | None:
    """Why a log is left out before any check, or None when it is selected."""
    if not seed_allowed:
        return "seed_not_analysed"
    if phase in admitted:
        return None
    if phase is None:
        return "phase_missing"
    if phase in ("adhoc", PILOT_PHASE):
        return f"phase_{phase}"
    return "phase_unknown"


def collect_cells(
    logs: Iterable[Path],
    config: SimulationConfig,
    *,
    include_pilot: bool = False,
    allow_mock: bool = False,
) -> Collected:
    """Select the logs the inference admits, check each, and group them by cell.

    Selection comes first, by seed and run phase, so a log that was never
    meant for this analysis -- a stray smoke test, an ad hoc or mock run, a
    pilot run -- is listed as left out instead of aborting the report. Every
    selected log is then checked, and any failure refuses the analysis: a
    selected log that is unfinished, short of the horizon, from the mock
    (unless allowed) or from another configuration is a defect, not noise.
    """
    allowed = set(config.seeds.final)
    admitted = {FINAL_PHASE}
    if include_pilot:
        allowed |= set(config.seeds.pilot)
        admitted.add(PILOT_PHASE)

    collected = Collected()
    skipped_seeds: set[int] = set()
    for candidate in logs:
        path = Path(candidate)
        try:
            metadata = read_metadata(path)
        except (LogError, OSError, ValueError) as exc:
            raise AnalysisError(
                f"{path}: unreadable run metadata ({exc})",
                key="error.report.unreadable",
                path=str(path),
            ) from exc
        phase = getattr(metadata, "phase", None)
        reason = _skip_reason(metadata.seed in allowed, phase, admitted)
        if reason is not None:
            if reason == "seed_not_analysed":
                skipped_seeds.add(metadata.seed)
            collected.skipped_logs.append(
                {
                    "path": str(path),
                    "scenario": metadata.scenario.value,
                    "experiment": metadata.experiment.value,
                    "condition": metadata.condition.value,
                    "seed": metadata.seed,
                    "phase": phase,
                    "mock": metadata.llm_mock,
                    "reason": reason,
                }
            )
            continue
        _check_log(path, metadata, config, allow_mock=allow_mock)

        key = (metadata.scenario.value, metadata.experiment.value)
        cell = collected.cells.setdefault(key, Cell(scenario=key[0], experiment=key[1]))
        run_key = (metadata.condition.value, metadata.seed)
        if run_key in cell.logs:
            raise AnalysisError(
                f"{path}: a second run of condition {run_key[0]} seed {run_key[1]} "
                f"(first: {cell.logs[run_key]})",
                key="error.report.duplicate_run",
                path=str(path),
                condition=run_key[0],
                seed=run_key[1],
                first=str(cell.logs[run_key]),
            )
        cell.logs[run_key] = path
        cell.metadata[run_key] = metadata
    collected.skipped_seeds = sorted(skipped_seeds)
    return collected


def _check_log(
    path: Path, metadata: RunMetadata, config: SimulationConfig, *, allow_mock: bool
) -> None:
    """The checks every selected log passes, in the order derive-budget applies them."""
    if not log_is_finished(path):
        raise AnalysisError(
            f"{path}: the run did not finish", key="error.report.unfinished", path=str(path)
        )
    if metadata.llm_mock and not allow_mock:
        raise AnalysisError(
            f"{path}: a mock run, analysed only when mock runs are allowed",
            key="error.report.mock_run",
            path=str(path),
        )
    ticks = run_end_ticks(path)
    if ticks != config.time.horizon_ticks:
        raise AnalysisError(
            f"{path}: ran {ticks} ticks, not the full {config.time.horizon_ticks}; the "
            f"analysis compares runs at the planned horizon",
            key="error.report.short_horizon",
            path=str(path),
            ticks=str(ticks),
            horizon=config.time.horizon_ticks,
        )
    _check_hash(path, metadata, config)


def _check_hash(path: Path, metadata: RunMetadata, config: SimulationConfig) -> None:
    try:
        derived = _derived(config, metadata)
    except ValueError as exc:
        raise AnalysisError(
            f"{path}: the run used a different configuration ({exc})",
            key="error.report.config_mismatch",
            path=str(path),
        ) from exc
    if config_hash(derived) != metadata.config_hash:
        raise AnalysisError(
            f"{path}: the run used a different configuration",
            key="error.report.config_mismatch",
            path=str(path),
        )


def _derived(config: SimulationConfig, metadata: RunMetadata) -> SimulationConfig:
    derived = config.for_cell(
        scenario=metadata.scenario, experiment=metadata.experiment, seed=metadata.seed
    ).for_condition(metadata.condition, seed=metadata.seed)
    return with_run_phase(derived, getattr(metadata, "phase", None))


def incomplete_seeds(cell: Cell) -> list[int]:
    """Seeds of a cell that lack one of A, B and C.

    The unit of comparison is the seed trio (design s11.4): a seed with only
    two conditions would enter some contrasts and not others, and its common
    active horizon (s8.4) would be cut at the earliest exhaustion among the
    runs that happen to exist, not among the three.
    """
    wanted = {c.value for c in Condition}
    by_seed: dict[int, set[str]] = {}
    for condition, seed in cell.logs:
        by_seed.setdefault(seed, set()).add(condition)
    return sorted(seed for seed, conditions in by_seed.items() if conditions != wanted)


def _restricted(cell: Cell, dropped: set[int]) -> Cell:
    return Cell(
        scenario=cell.scenario,
        experiment=cell.experiment,
        logs={key: path for key, path in cell.logs.items() if key[1] not in dropped},
        metadata={key: meta for key, meta in cell.metadata.items() if key[1] not in dropped},
    )


def complete_trios(cell: Cell) -> Cell:
    """The cell restricted to the seeds that ran all three conditions."""
    return _restricted(cell, set(incomplete_seeds(cell)))


def trio_horizons(cell: Cell, config: SimulationConfig) -> dict[int, int]:
    """Per seed, the trio's common active horizon (design s8.4).

    The earliest budget exhaustion among A, B and C, or the planned horizon
    when none exhausted -- as in Experiment 1, where there is no budget.
    """
    by_seed: dict[int, list[int | None]] = {}
    for (_, seed), path in cell.logs.items():
        by_seed.setdefault(seed, []).append(exhaustion_tick(path))
    return {
        seed: common_active_horizon(ticks, config.time.horizon_ticks)
        for seed, ticks in by_seed.items()
    }


def _marker_horizon(directory: Path) -> Any:
    """The common active horizon a trio's integrity marker recorded."""
    from rpg_economy.integrity import INTEGRITY_MARKER

    try:
        payload = json.loads((directory / INTEGRITY_MARKER).read_text(encoding="utf-8"))
        return payload["horizon"]["common_active_horizon"]
    except OSError, ValueError, KeyError, TypeError:
        return None


def gate_trios(
    cell: Cell, config: SimulationConfig
) -> tuple[Cell, list[dict[str, Any]], dict[int, int]]:
    """The trios of a cell that passed the integrity gate (design s12).

    A trio is admitted only when its three logs sit in one trio directory
    whose integrity marker pins exactly those bytes, and when the common
    active horizon the marker recorded is the one computed here from the same
    logs. Anything else is rejected, with its reason, never repaired: a trio
    that fails the check is not part of the experiment.

    Returns the cell restricted to the admitted trios, the rejections, and the
    admitted trios' common active horizons.
    """
    from rpg_economy.integrity import has_integrity_marker

    by_seed: dict[int, dict[str, Path]] = {}
    for (condition, seed), path in cell.logs.items():
        by_seed.setdefault(seed, {})[condition] = path
    computed = trio_horizons(cell, config)

    rejected: list[dict[str, Any]] = []
    for seed, paths in sorted(by_seed.items()):
        directories = {path.parent for path in paths.values()}
        directory = min(directories)
        named = all(path.name == f"{condition}.jsonl" for condition, path in paths.items())
        if len(directories) != 1 or not named or not has_integrity_marker(directory):
            rejected.append(
                {"seed": seed, "reason": "integrity_marker", "directory": str(directory)}
            )
            continue
        recorded = _marker_horizon(directory)
        if recorded != computed[seed]:
            rejected.append(
                {
                    "seed": seed,
                    "reason": "horizon_mismatch",
                    "directory": str(directory),
                    "recorded": recorded,
                    "computed": computed[seed],
                }
            )
    dropped = {entry["seed"] for entry in rejected}
    horizons = {seed: value for seed, value in computed.items() if seed not in dropped}
    return _restricted(cell, dropped), rejected, horizons


def mixed_versions(cell: Cell) -> list[str]:
    """The version fields (``VERSION_FIELDS``) on which a cell's runs disagree."""
    differing: list[str] = []
    for name in VERSION_FIELDS:
        values = {
            canonical_json(getattr(metadata, name, None)) for metadata in cell.metadata.values()
        }
        if len(values) > 1:
            differing.append(name)
    return differing


def analyse_cell(
    cell: Cell,
    config: SimulationConfig,
    *,
    final_seeds: Iterable[int] | None,
    horizons: dict[int, int] | None = None,
):
    """Metrics for every run of one cell, then its contrasts.

    The cell must hold complete trios only (see :func:`complete_trios`).
    ``horizons`` are the trios' common active horizons when already computed
    (:func:`gate_trios`); they apply in Experiment 2 only.
    """
    incomplete = incomplete_seeds(cell)
    if incomplete:
        raise AnalysisError(
            f"{cell.scenario}/{cell.experiment}: incomplete trios for seeds "
            f"{incomplete}; every seed needs A, B and C",
            key="error.report.incomplete_trios",
            scenario=cell.scenario,
            experiment=cell.experiment,
            seeds=", ".join(map(str, incomplete)),
        )
    active: dict[int, int] = {}
    if cell.experiment == ExperimentRegime.EQUAL_COST.value:
        # Design s8.4: the trio is cut at the earliest exhaustion among its runs.
        active = horizons if horizons is not None else trio_horizons(cell, config)

    runs = []
    series: dict[str, list[list[float | None]]] = {}
    for (condition, seed), path in sorted(cell.logs.items()):
        metadata = cell.metadata.get((condition, seed)) or read_metadata(path)
        derived = _derived(config, metadata)
        runs.append(compute_run_metrics(path, derived, common_active_horizon=active.get(seed)))
        target = target_series(RunLog(path), derived, primary_target(derived))
        series.setdefault(condition, []).append(
            [None if value is None else float(value) for value in target]
        )

    results = build_results(runs, final_seeds=final_seeds)
    return results, {condition: _pointwise_median(rows) for condition, rows in series.items()}


def _pointwise_median(rows: list[list[float | None]]) -> list[float | None]:
    """Per tick, the median across seeds of the values that exist."""
    length = max((len(row) for row in rows), default=0)
    out: list[float | None] = []
    for tick in range(length):
        values = [row[tick] for row in rows if tick < len(row) and row[tick] is not None]
        out.append(statistics.median(values) if values else None)
    return out


def _cell_key(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}"


def analyse(
    directory: Path,
    config: SimulationConfig,
    *,
    include_pilot: bool = False,
    allow_mock: bool = False,
) -> Analysis:
    logs = run_logs(directory)
    if not logs:
        raise AnalysisError(
            f"{directory}: no run logs", key="error.report.no_logs", path=str(directory)
        )
    collected = collect_cells(logs, config, include_pilot=include_pilot, allow_mock=allow_mock)
    if not collected.cells:
        reasons = sorted({entry["reason"] for entry in collected.skipped_logs})
        raise AnalysisError(
            f"{directory}: none of its {len(logs)} logs is admitted to the analysis "
            f"(left out: {', '.join(reasons)}); inference reads final-phase runs on "
            f"the final seeds",
            key="error.report.no_admitted_logs",
            path=str(directory),
            count=len(logs),
        )
    final_seeds = None if include_pilot else config.seeds.final
    pilot = set(config.seeds.pilot)
    analysis = Analysis(
        skipped_seeds=collected.skipped_seeds, skipped_logs=list(collected.skipped_logs)
    )
    for key, cell in sorted(collected.cells.items()):
        incomplete = incomplete_seeds(cell)
        if incomplete:
            # Left out of the horizon and of every contrast, and said so: the
            # caller reports it (design s11.3, pairs used and dropped).
            analysis.incomplete_trios[key] = incomplete
            warnings.warn(
                f"{cell.scenario}/{cell.experiment}: seeds {incomplete} lack one of "
                "A, B and C and are left out of the analysis",
                IncompleteTrioWarning,
                stacklevel=2,
            )
        trios = complete_trios(cell)
        if not trios.logs:
            # Nothing in this cell can be paired, but that says nothing about
            # the others: skip it, record it, and let the caller report it.
            conditions = sorted({condition for condition, _ in cell.logs})
            analysis.skipped_cells[key] = conditions
            warnings.warn(
                f"{cell.scenario}/{cell.experiment}: no complete trio (conditions "
                f"present: {conditions}); the cell is left out of the analysis",
                IncompleteTrioWarning,
                stacklevel=2,
            )
            continue
        # Design s12: a trio that fails the integrity check is rejected.
        trios, rejected, horizons = gate_trios(trios, config)
        if rejected:
            analysis.rejected_trios[key] = rejected
            warnings.warn(
                f"{cell.scenario}/{cell.experiment}: the trios of seeds "
                f"{[entry['seed'] for entry in rejected]} failed the integrity check "
                f"({sorted({entry['reason'] for entry in rejected})}) and are left out",
                RejectedTrioWarning,
                stacklevel=2,
            )
        if not trios.logs:
            continue
        differing = mixed_versions(trios)
        if differing:
            analysis.refused_cells[key] = differing
            warnings.warn(
                f"{cell.scenario}/{cell.experiment}: its runs differ in {differing}; "
                "a cell mixing code versions compares versions, not architectures, "
                "and is left out",
                RejectedTrioWarning,
                stacklevel=2,
            )
            continue
        results, series = analyse_cell(trios, config, final_seeds=final_seeds, horizons=horizons)
        # Exploration only: pilot seeds never enter inference (design s11.4).
        results.pilot_seeds_included = include_pilot and bool(set(results.seeds) & pilot)
        analysis.results[key] = results
        analysis.price_series[key] = series
    # Every results file names what was left out, not only the console (s11.3).
    cells_skipped = {
        _cell_key(key): conditions for key, conditions in sorted(analysis.skipped_cells.items())
    }
    trios_rejected = {
        _cell_key(key): entries for key, entries in sorted(analysis.rejected_trios.items())
    }
    cells_refused = {
        _cell_key(key): fields for key, fields in sorted(analysis.refused_cells.items())
    }
    # The metric parameters are applied to the logs, not by the runs, so they
    # are not part of a run's identity: the results record the ones used here.
    parameters = config.metrics.model_dump(mode="json")
    for key, results in analysis.results.items():
        results.incomplete_trios = list(analysis.incomplete_trios.get(key, []))
        results.skipped_seeds = list(collected.skipped_seeds)
        results.skipped_cells = dict(cells_skipped)
        results.skipped_logs = list(analysis.skipped_logs)
        results.rejected_trios = dict(trios_rejected)
        results.refused_cells = dict(cells_refused)
        results.metric_parameters = dict(parameters)
    if not analysis.results:
        raise AnalysisError(
            f"no cell has an admissible complete trio (conditions present: {cells_skipped}; "
            f"trios rejected by the integrity check: {trios_rejected}; cells refused for "
            f"mixed versions: {cells_refused}); need A, B and C",
            key="error.report.no_complete_trio",
            skipped=len(cells_skipped),
            rejected=sum(len(entries) for entries in trios_rejected.values()),
            refused=len(cells_refused),
        )
    return analysis
