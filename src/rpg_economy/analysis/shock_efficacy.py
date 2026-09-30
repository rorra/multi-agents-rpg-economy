"""Does the shock move the economy? A calibration diagnostic (design v1.0 s10).

A shock that leaves its target untouched makes every contrast on it a
comparison of noise. Before the final runs, each scenario is checked offline with
rule-based agents: the same seed runs once with its shock and once without, and
the difference over the shock phase is the shock's effect on that world. Pairing
on the seed removes the drift a base-versus-shock comparison would mix in.

The same comparison can be read from the pilot's own logs, where no
unshocked twin exists: there the last W ticks of the base phase stand in for
the control, scaled to the shock phase's length. Only the last W: the start of
a run is the economy settling in from its endowments, which is not what the
shock is measured against (the recovery reference uses the same window,
design s11.1). Coarser, since it mixes in drift, but free, and it answers the
question calibration cannot: does the shock bite with LLM agents?

Not part of the experiment's metrics, and no LLM is called here.
"""

from __future__ import annotations

import statistics
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from rpg_economy.analysis.logchecks import with_run_phase
from rpg_economy.config import config_hash
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import ExperimentRegime, Resource, Scenario
from rpg_economy.logging_ import EventType
from rpg_economy.metrics.run_metrics import (
    PHASE_BASE,
    PHASE_SHOCK,
    RunLog,
    stockout_ticks_between,
)
from rpg_economy.metrics.targets import primary_target

#: What counts as a shock that bites, as the median over seeds of the paired
#: shocked-minus-control difference. Diagnostic thresholds, not experimental
#: parameters: they decide whether a calibration is worth running.
MIN_PRICE_CHANGE = 0.10
MIN_STOCKOUT_TICKS = 10


@dataclass(frozen=True)
class ShockPhase:
    """The target's behaviour over the shock phase of one run."""

    mean_price: float | None
    stockout_ticks: int | None
    traded_value: float
    #: Units of the target gathered on expeditions during the shock (resources only).
    gathered: int | None = None
    #: Units of the target held across all agents when the shock ends.
    held_at_end: int | None = None


@dataclass(frozen=True)
class SeedEffect:
    seed: int
    shocked: ShockPhase
    control: ShockPhase

    @property
    def price_change(self) -> float | None:
        """Relative change of the target's mean shock-phase price."""
        before, after = self.control.mean_price, self.shocked.mean_price
        if before is None or after is None or before == 0:
            return None
        return (after - before) / before

    @property
    def stockout_change(self) -> int | None:
        if self.shocked.stockout_ticks is None or self.control.stockout_ticks is None:
            return None
        return self.shocked.stockout_ticks - self.control.stockout_ticks

    @property
    def supply_change(self) -> float | None:
        """Relative change of the units gathered during the shock."""
        before, after = self.control.gathered, self.shocked.gathered
        if before is None or after is None or before == 0:
            return None
        return (after - before) / before

    @property
    def value_change(self) -> float | None:
        if self.control.traded_value == 0:
            return None
        return (self.shocked.traded_value - self.control.traded_value) / self.control.traded_value


@dataclass
class Efficacy:
    scenario: str
    target: str | None
    effects: list[SeedEffect] = field(default_factory=list)
    #: The runs' shock phase over the full-length one: below 1 for the pilot's
    #: compressed calibration stage, where the stockout threshold (10 ticks of
    #: a full shock phase) scales in proportion.
    phase_scale: float = 1.0

    def median(self, name: str) -> float | None:
        values = [getattr(e, name) for e in self.effects]
        present = [v for v in values if v is not None]
        return statistics.median(present) if present else None

    def effective(
        self,
        *,
        min_price_change: float = MIN_PRICE_CHANGE,
        min_stockout_ticks: int = MIN_STOCKOUT_TICKS,
    ) -> bool:
        """Whether the shock moves its pre-registered target (design s11.3).

        The target's price moves, or its stockouts grow. Lower supply alone does
        not count: a scarcity shock lowers supply by construction, so it would
        pass whether or not the market ever noticed.
        """
        if self.target is None:  # liquidity_loss: aggregate traded value
            change = self.median("value_change")
            return change is not None and abs(change) >= min_price_change
        price = self.median("price_change")
        stockouts_added = self.median("stockout_change")
        threshold = max(1, round(min_stockout_ticks * self.phase_scale))
        return (price is not None and abs(price) >= min_price_change) or (
            stockouts_added is not None and stockouts_added >= threshold
        )


def phase_stats(
    log: RunLog, config: SimulationConfig, good: str | None, phase: str = PHASE_SHOCK
) -> ShockPhase:
    """The target's behaviour over one phase of one run."""
    bounds = {
        PHASE_BASE: (0, config.phases.shock_tick),
        PHASE_SHOCK: (config.phases.shock_tick, config.phases.reversion_tick),
    }
    lo, hi = bounds.get(phase, (config.phases.reversion_tick, config.time.horizon_ticks))
    return window_stats(log, good, lo, hi)


def window_stats(log: RunLog, good: str | None, lo: int, hi: int) -> ShockPhase:
    """The target's behaviour over the ticks ``[lo, hi)`` of one run."""
    prices = [
        float(trade.price)
        for trade in log.trades
        if good is not None and trade.good == good and lo <= trade.tick < hi
    ]
    value = sum(float(trade.value) for trade in log.trades if lo <= trade.tick < hi)
    gathered = None
    if good is not None and good in {r.value for r in Resource}:
        gathered = sum(
            int(e.payload["yielded"].get(good, 0))
            for e in log.of_type(EventType.EXPEDITION_RESOLVED)
            if lo <= e.tick < hi
        )
    return ShockPhase(
        gathered=gathered,
        mean_price=statistics.fmean(prices) if prices else None,
        stockout_ticks=(stockout_ticks_between(log, good, lo, hi) if good is not None else None),
        traded_value=value,
    )


def control_window(config: SimulationConfig) -> tuple[int, int]:
    """The last W ticks of the base phase: the control for a pilot log.

    W is the metrics window (``metrics.volatility_window_w``), clipped to the
    base phase.
    """
    end = config.phases.shock_tick
    return max(0, end - config.metrics.volatility_window_w), end


def shock_phase(log: RunLog, config: SimulationConfig, good: str | None) -> ShockPhase:
    from rpg_economy.domain import parse_good
    from rpg_economy.engine import build_world

    stats = phase_stats(log, config, good)
    if good is None:
        return stats
    world = build_world(config)
    initial = sum(world.account(a).quantity(parse_good(good)) for a in world.agent_ids)
    return replace(stats, held_at_end=_held_at(log, config.phases.reversion_tick, good, initial))


def _scaled(stats: ShockPhase, factor: float) -> ShockPhase:
    """Counts and flows rescaled to another phase length; prices unchanged."""
    return ShockPhase(
        mean_price=stats.mean_price,
        stockout_ticks=(
            None if stats.stockout_ticks is None else round(stats.stockout_ticks * factor)
        ),
        traded_value=stats.traded_value * factor,
        gathered=None if stats.gathered is None else round(stats.gathered * factor),
    )


def from_logs(
    logs: list[Path], config: SimulationConfig, *, allow_mock: bool = False
) -> tuple[dict[tuple[str, str], Efficacy], list[Path]]:
    """Base phase against shock phase, per finished run, grouped by scenario and condition.

    Only the pilot is read: Experiment 1 runs on pilot seeds, recorded as the
    pilot phase where the log records a phase at all, and from a real model
    unless ``allow_mock``. Anything else is left out: a final or Experiment 2
    run is not the pilot, and a mock run says nothing about LLM agents.

    Each log is matched to the configuration it actually ran: the base one or
    a pilot stage derived from it (the calibration stage runs a shorter
    horizon, with its own phase ticks). A log that matches none, or for which
    no candidate configuration can be derived at all, is returned as unmatched
    rather than read with the wrong phases. A seed counts once per scenario and
    condition: when both stages ran it, the run of the configuration tried
    first (the base one, at the full horizon) is the one read. The control is
    the last W base ticks (:func:`control_window`), rescaled to the shock
    phase's length so counts compare.
    """
    from rpg_economy.campaign.pilot import STAGES, stage_config
    from rpg_economy.logging_ import log_is_finished, read_metadata

    candidates = [config]
    for stage in STAGES.values():
        try:
            candidates.append(stage_config(config, stage))
        except ValueError:
            continue

    pilot_seeds = set(config.seeds.pilot)
    #: Per (scenario, condition, seed), the chosen run: its candidate's rank and path.
    chosen: dict[tuple[str, str, int], tuple[int, Path, SimulationConfig]] = {}
    unmatched: list[Path] = []
    for path in logs:
        if not log_is_finished(path):
            continue
        metadata = read_metadata(path)
        if metadata.experiment is not ExperimentRegime.EQUAL_ACTIVITY:
            continue
        if metadata.seed not in pilot_seeds:
            continue
        phase = getattr(metadata, "phase", None)
        if phase is not None and phase != "pilot":
            continue
        if metadata.llm_mock and not allow_mock:
            continue
        match = _matching_config(metadata, candidates)
        if match is None:
            unmatched.append(path)
            continue
        rank, derived = match
        key = (metadata.scenario.value, metadata.condition.value, metadata.seed)
        if key not in chosen or rank < chosen[key][0]:
            chosen[key] = (rank, path, derived)

    grouped: dict[tuple[str, str], Efficacy] = {}
    for (scenario, condition, seed), (_, path, derived) in sorted(chosen.items()):
        lo, hi = control_window(derived)
        factor = derived.phases.shock_ticks / (hi - lo)
        target = primary_target(derived)
        good = target.good.value if target.good is not None else None
        log = RunLog(path)
        efficacy = grouped.setdefault(
            (scenario, condition),
            Efficacy(
                scenario=scenario,
                target=good,
                phase_scale=derived.phases.shock_ticks / config.phases.shock_ticks,
            ),
        )
        efficacy.effects.append(
            SeedEffect(
                seed=seed,
                shocked=phase_stats(log, derived, good, PHASE_SHOCK),
                control=_scaled(window_stats(log, good, lo, hi), factor),
            )
        )
    return dict(sorted(grouped.items())), unmatched


def _matching_config(
    metadata, candidates: list[SimulationConfig]
) -> tuple[int, SimulationConfig] | None:
    """The first candidate configuration whose derivation for this run has its hash.

    Returns its rank among the candidates and the derived configuration, or
    None. A candidate that cannot be derived for the run (an Experiment 2 cell
    of a configuration without a budget, say) simply does not match.
    """
    for rank, candidate in enumerate(candidates):
        try:
            attempt = candidate.for_cell(
                scenario=metadata.scenario, experiment=metadata.experiment, seed=metadata.seed
            ).for_condition(metadata.condition, seed=metadata.seed)
            attempt = with_run_phase(attempt, getattr(metadata, "phase", None))
        except ValueError:
            continue
        if config_hash(attempt) == metadata.config_hash:
            return rank, attempt
    return None


def _held_at(log: RunLog, tick: int, good: str, initial: int) -> int:
    """Units of ``good`` across all inventories just before ``tick``, from the log."""
    held = initial
    for event in log.events:
        if event.tick >= tick:
            break
        payload = event.payload
        match event.type:
            case EventType.EXPEDITION_RESOLVED:
                held += int(payload["yielded"].get(good, 0))
            case EventType.PRODUCTION_COMPLETED:
                held -= int((payload.get("consumed") or {}).get(good, 0))
                if payload.get("product") == good:
                    held += int(payload.get("quantity", 0))
            case EventType.CONSUMPTION if payload.get("product") == good:
                held -= int(payload.get("quantity", 0))
            case EventType.EQUIPMENT_EQUIPPED if payload.get("product") == good:
                held -= 1
    return held


def run_rule_based(config: SimulationConfig, path: Path, *, shock: bool = True) -> RunLog:
    """One offline run with rule-based agents, activated as the experiment does it.

    Triggers, the neutral scheduler and K activations per cycle, as in
    Condition B. Activating every agent every cycle instead would calibrate an
    economy several times busier than the one the experiment runs.
    """
    from rpg_economy.agents.decision_phase import RuleBasedActivation
    from rpg_economy.cli_support import build_metadata
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from rpg_economy.orchestration import DecentralizedOrchestrator

    with EventLogWriter(path, build_metadata(config)) as log:
        engine = Engine(config, build_world(config), log, shock_enabled=shock)
        engine.run(DecentralizedOrchestrator(config, engine, RuleBasedActivation(config)))
    return RunLog(path)


_run = run_rule_based


def measure(config: SimulationConfig, seeds: list[int], directory: Path | None = None) -> Efficacy:
    """Run each seed with and without the shock and compare the shock phase."""
    target = primary_target(config)
    good = target.good.value if target.good is not None else None
    out = directory or Path(tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)

    efficacy = Efficacy(scenario=config.run.scenario.value, target=good)
    for seed in seeds:
        seeded = config.for_cell(
            scenario=config.run.scenario, experiment=config.run.experiment, seed=seed
        )
        shocked = _run(seeded, out / f"{seed}-shock.jsonl", shock=True)
        control = _run(seeded, out / f"{seed}-control.jsonl", shock=False)
        efficacy.effects.append(
            SeedEffect(
                seed=seed,
                shocked=shock_phase(shocked, seeded, good),
                control=shock_phase(control, seeded, good),
            )
        )
    return efficacy


def measure_scenario(
    config: SimulationConfig,
    scenario: Scenario,
    *,
    seeds: list[int] | None = None,
    directory: Path | None = None,
) -> Efficacy:
    """The offline efficacy check for one scenario, whatever ``config`` runs.

    Lets a caller loop over the primary and the secondary scenarios with one
    configuration. ``seeds`` defaults to the pilot seeds.
    """
    scoped = config.for_cell(
        scenario=scenario, experiment=config.run.experiment, seed=config.run.seed
    )
    chosen = list(config.seeds.pilot) if seeds is None else list(seeds)
    out = None if directory is None else directory / scenario.value
    return measure(scoped, chosen, out)


def efficacy_scenarios(config: SimulationConfig) -> list[Scenario]:
    """The primary scenario, then the secondary scenarios the campaign plans (s10, s11.4).

    The same list the campaign plan runs, so no planned shock goes unchecked.
    """
    from rpg_economy.campaign.plan import planned_secondary_scenarios

    ordered = [Scenario.RESOURCE_SCARCITY]
    ordered += [s for s in planned_secondary_scenarios(config) if s not in ordered]
    return ordered


__all__ = [
    "Efficacy",
    "SeedEffect",
    "ShockPhase",
    "control_window",
    "efficacy_scenarios",
    "measure",
    "measure_scenario",
    "shock_phase",
    "window_stats",
]
