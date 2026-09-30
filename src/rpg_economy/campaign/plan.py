"""The pre-registered scenario matrix and seed sets (design v1.0 s10, s11.4).

The hierarchy below is pre-registered: it is a property of the design,
not a runtime choice. It is encoded here rather than in a shell script so that
"which scenarios get the full seed set" and "which experiments each scenario
runs" are never decided ad hoc at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime, Scenario

PLAN_SCHEMA_VERSION = "1.0"

#: Design v1.0 s11.4: the primary scenario takes the full seed set in both
#: experiments; secondary scenarios take at least eight seeds, Experiment 1 only.
PRIMARY_SCENARIO = Scenario.RESOURCE_SCARCITY
SECONDARY_SCENARIOS = (
    Scenario.DEMAND_SURGE,
    Scenario.PRODUCTIVITY_DROP,
    Scenario.LIQUIDITY_LOSS,
)

MIN_PRIMARY_SEEDS = 10
TARGET_PRIMARY_SEEDS = 15
MIN_SECONDARY_SEEDS = 8
MIN_PILOT_SEEDS = 8
MAX_PILOT_SEEDS = 10


class ScenarioTier(StrEnum):
    """Where a scenario sits in the pre-registered hierarchy (s11.4)."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    OPTIONAL = "optional"


class CampaignPhase(StrEnum):
    PILOT = "pilot"
    FINAL = "final"


class Notice(str):
    """An English message for payloads and logs, with the catalog key that renders it.

    A ``str``, so it serialises and compares as the text it carries; the CLI
    renders ``key`` and ``params`` in the reader's language instead
    (IMPLEMENTATION.md section 2).
    """

    key: str
    params: dict[str, Any]

    def __new__(cls, text: str, key: str, **params: Any) -> Notice:
        notice = super().__new__(cls, text)
        notice.key = key
        notice.params = params
        return notice


class PlanError(ValueError):
    """The requested campaign violates the pre-registered design.

    ``key`` and ``params`` name the catalog message the CLI prints; the English
    text stays the exception's message for library callers.
    """

    def __init__(self, message: str, *, key: str | None = None, **params: Any) -> None:
        super().__init__(message)
        self.key = key
        self.params = params


def scenario_tier(scenario: Scenario) -> ScenarioTier:
    if scenario is PRIMARY_SCENARIO:
        return ScenarioTier.PRIMARY
    if scenario in SECONDARY_SCENARIOS:
        return ScenarioTier.SECONDARY
    # noisy_info is admissible only if defined before the final runs
    # (s10); it is never included by default.
    return ScenarioTier.OPTIONAL


def planned_secondary_scenarios(config: SimulationConfig) -> list[Scenario]:
    """The secondary scenarios a campaign plans, in order.

    The single source of that list: the plan and the shock-efficacy check both
    read it, so a scenario can never be run without its shock being checked.
    The configured scope (design v1.0 s10), or the whole secondary tier when
    none is configured.
    """
    if config.campaign.primary_only:
        return []
    configured = list(dict.fromkeys(config.campaign.secondary_scenarios))
    return configured or list(SECONDARY_SCENARIOS)


def experiments_for(
    scenario: Scenario,
    phase: CampaignPhase = CampaignPhase.FINAL,
    *,
    primary_only: bool = False,
) -> tuple[ExperimentRegime, ...]:
    """Which budget regimes a scenario runs under (s11.4, s8.4).

    Only the primary scenario runs Experiment 2. Running the secondary scenarios
    under both regimes would multiply cost without addressing the research
    question, which is why the design confines the equal-cost comparison to the
    scenario the primary contrast lives in.

    The pilot is always Experiment 1 only, in every scenario. Experiment 2's
    budget is the pooled median of the primary scenario's Experiment 1 pilot
    runs (s8.4), so a pilot that tried to run Experiment 2 would need the number
    it exists to produce.
    """
    if phase is CampaignPhase.PILOT or primary_only:
        return (ExperimentRegime.EQUAL_ACTIVITY,)
    if scenario_tier(scenario) is ScenarioTier.PRIMARY:
        return (ExperimentRegime.EQUAL_ACTIVITY, ExperimentRegime.EQUAL_COST)
    return (ExperimentRegime.EQUAL_ACTIVITY,)


@dataclass(frozen=True)
class PlannedTrio:
    """One seed's A/B/C trio for one scenario and experiment.

    The trio, not the run, is the unit of planning: the design's pairing means a
    seed is only usable when all three conditions ran it, so a campaign that
    could stop between conditions would produce unpairable data.
    """

    scenario: Scenario
    experiment: ExperimentRegime
    seed: int
    #: Run only this condition of the trio now (a check run one condition at a
    #: time); the trio is whole, and verified, once all three have run.
    only: Condition | None = None

    @property
    def conditions(self) -> tuple[Condition, ...]:
        if self.only is not None:
            return (self.only,)
        return (Condition.A, Condition.B, Condition.C)

    @property
    def runs(self) -> int:
        return len(self.conditions)

    def to_payload(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.value,
            "experiment": self.experiment.value,
            "seed": self.seed,
            "conditions": [c.value for c in self.conditions],
        }


@dataclass
class CampaignPlan:
    """Every trio a campaign will run, and the warnings it carries."""

    phase: CampaignPhase
    trios: list[PlannedTrio] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_runs(self) -> int:
        return sum(trio.runs for trio in self.trios)

    def cell(self, base_config: SimulationConfig, trio: PlannedTrio) -> SimulationConfig:
        """The configuration of one of this plan's trios (``cell_config``)."""
        return cell_config(base_config, trio, self.phase)

    def by_scenario(self) -> dict[Scenario, list[PlannedTrio]]:
        grouped: dict[Scenario, list[PlannedTrio]] = {}
        for trio in self.trios:
            grouped.setdefault(trio.scenario, []).append(trio)
        return grouped

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "phase": self.phase.value,
            "total_trios": len(self.trios),
            "total_runs": self.total_runs,
            "trios": [trio.to_payload() for trio in self.trios],
            "warnings": list(self.warnings),
        }


def cell_config(
    base_config: SimulationConfig, trio: PlannedTrio, phase: CampaignPhase | str | None
) -> SimulationConfig:
    """A planned trio's cell configuration, marked with the campaign phase.

    Every run a campaign or a pilot stage makes goes through here, so each one
    records (``run.phase``, then its metadata) whether it was a pilot or a
    final run. Without ``phase`` the configured one is kept; one-off runs
    never come through here and stay ``adhoc`` by default.
    """
    return base_config.for_cell(
        scenario=trio.scenario,
        experiment=trio.experiment,
        seed=trio.seed,
        phase=None if phase is None else CampaignPhase(phase).value,
    )


def build_plan(
    config: SimulationConfig,
    *,
    phase: CampaignPhase | str = CampaignPhase.PILOT,
    scenarios: tuple[Scenario, ...] | None = None,
) -> CampaignPlan:
    """Compose the pre-registered matrix into a concrete list of trios.

    Raises rather than trimming when the configured seed sets cannot support the
    design: a campaign that silently ran eight seeds where the design requires
    ten would produce an underpowered result indistinguishable from a planned
    one.
    """
    phase = CampaignPhase(phase)
    seeds = config.seeds.pilot if phase is CampaignPhase.PILOT else config.seeds.final
    if not seeds:
        raise PlanError(
            f"no {phase.value} seeds configured", key="error.plan.no_seeds", phase=phase.value
        )

    if scenarios is not None:
        selected = scenarios
    else:
        # The configured scope, not the full hierarchy: the secondary scenarios
        # in scope (design v1.0 s10) are config, so the choice is in the hash.
        selected = (PRIMARY_SCENARIO, *planned_secondary_scenarios(config))
    # Each scenario once, in order. A scenario named twice (or the primary also
    # listed as a secondary) would plan the same trio twice, and two workers
    # would then write the same trio directory at once.
    selected = tuple(dict.fromkeys(selected))
    plan = CampaignPlan(phase=phase)
    planned: set[PlannedTrio] = set()

    for scenario in selected:
        tier = scenario_tier(scenario)
        if tier is ScenarioTier.OPTIONAL:
            # s10: admissible only if defined before the final runs. It is
            # never planned implicitly, so asking for it is always deliberate.
            plan.warnings.append(
                Notice(
                    f"{scenario.value} is outside the pre-registered hierarchy and is "
                    f"included only because it was named explicitly",
                    "campaign.warning.optional_scenario",
                    scenario=scenario.value,
                )
            )
        scenario_seeds = seeds
        if phase is CampaignPhase.FINAL and tier is ScenarioTier.SECONDARY:
            # Secondary scenarios run a fixed subset of the final seeds, the same
            # first N for every secondary scenario, so their trios stay paired.
            scenario_seeds = seeds[: config.campaign.secondary_final_seeds]
        for experiment in experiments_for(
            scenario, phase, primary_only=config.campaign.primary_only
        ):
            for seed in scenario_seeds:
                trio = PlannedTrio(scenario=scenario, experiment=experiment, seed=seed)
                # A repeated seed is the same trio directory; plan it once.
                if trio not in planned:
                    planned.add(trio)
                    plan.trios.append(trio)

    plan.warnings.extend(
        _seed_count_warnings(
            phase,
            selected,
            len(seeds),
            secondary=min(len(seeds), config.campaign.secondary_final_seeds),
            reduced=config.campaign.primary_only,
        )
    )

    needs_budget = any(trio.experiment is ExperimentRegime.EQUAL_COST for trio in plan.trios)
    if needs_budget and config.budget.experiment_2_total_tokens is None:
        # The config validator rejects this too, but only when a run is
        # constructed -- which in a campaign means part-way through a batch,
        # after the Experiment 1 trios have already been paid for.
        raise PlanError(
            "plan includes Experiment 2 but budget.experiment_2_total_tokens is not "
            "derived; run derive-budget on the pilot logs and add its layer to the "
            "configuration (design v1.0 s8.4)",
            key="error.plan.no_budget",
        )

    return plan


def _seed_count_warnings(
    phase: CampaignPhase,
    scenarios: tuple[Scenario, ...],
    available: int,
    *,
    secondary: int | None = None,
    reduced: bool = False,
) -> list[str]:
    """Check the seed counts against s11.4, hard-failing where it is a floor."""
    warnings: list[str] = []

    if phase is CampaignPhase.PILOT:
        if not MIN_PILOT_SEEDS <= available <= MAX_PILOT_SEEDS:
            raise PlanError(
                f"pilot uses {available} seeds; design v1.0 s11.4 specifies "
                f"{MIN_PILOT_SEEDS}-{MAX_PILOT_SEEDS}",
                key="error.plan.pilot_seeds",
                count=available,
                min=MIN_PILOT_SEEDS,
                max=MAX_PILOT_SEEDS,
            )
        return warnings

    for scenario in scenarios:
        tier = scenario_tier(scenario)
        floor = MIN_PRIMARY_SEEDS if tier is ScenarioTier.PRIMARY else MIN_SECONDARY_SEEDS
        used = available if tier is ScenarioTier.PRIMARY or secondary is None else secondary
        if used < floor and reduced:
            warnings.append(
                Notice(
                    f"reduced campaign: {scenario.value} runs {used} final seeds, below "
                    f"the design's floor of {floor}",
                    "campaign.warning.reduced",
                    scenario=scenario.value,
                    floor=floor,
                    used=used,
                )
            )
            continue
        if used < floor:
            raise PlanError(
                f"{scenario.value} is {tier.value} and needs at least {floor} final "
                f"seeds; {used} configured",
                key="error.plan.seed_floor",
                scenario=scenario.value,
                tier=tier.value,
                floor=floor,
                used=used,
            )
        if tier is ScenarioTier.PRIMARY and available < TARGET_PRIMARY_SEEDS:
            # A target, not a floor: below it the campaign is valid but weaker
            # than the design asked for, and that should be visible.
            warnings.append(
                Notice(
                    f"primary scenario has {available} seeds; design v1.0 s11.4 targets "
                    f"{TARGET_PRIMARY_SEEDS}",
                    "campaign.warning.below_target",
                    count=available,
                    target=TARGET_PRIMARY_SEEDS,
                )
            )
    return warnings
