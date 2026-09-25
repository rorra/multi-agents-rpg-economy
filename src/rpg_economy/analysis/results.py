"""Aggregating runs into a language-neutral results file (design v1.0 s11.3, s11.4).

One value per run per metric, then contrasts per scenario. The results file holds
enum values and numbers only -- no rendered text -- so the English and the
Spanish report are produced from the very same file at render time, and neither
language can drift from the other's numbers (IMPLEMENTATION.md section 2).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.metrics.run_metrics import PHASE_SHOCK, RunMetrics

RESULTS_SCHEMA_VERSION = "1.0"

#: Metrics compared with the paired Wilcoxon. Recovery time is deliberately
#: absent: it may be censored, and s11.1 forbids treating a censored
#: observation as a measurement. It is analysed by survival methods instead.
CONTINUOUS_METRICS = (
    "volatility_shock",
    "volatility_shock_phase",
    "stockout_ticks",
    "traded_value",
    "useful_production",
    "gini",
    "coordination_efficiency",
    "activation_concentration",
    "unanswered_requests",
    "tokens_total",
    "llm_calls",
)

#: Pre-registered secondary economic metrics, contrasted like the others
#: (design s11.1, s11.3): the inventory-based stockout variant and the
#: multi-good aggregates, each good normalized by its base phase.
SECONDARY_ECONOMIC_METRICS = (
    "stockout_ticks_inventory",
    "volatility_all_goods",
    "volume_all_goods",
    "stockouts_all_goods",
)

#: The shock's propagation to the main finished product of the affected raw
#: material (secondary, design s11.3). Only where the target is a resource.
PROPAGATION_METRICS = (
    "propagation_price_shock",
    "propagation_stockout_ticks",
)

#: Economic metrics recomputed up to the common active horizon of each trio,
#: the secondary analysis of Experiment 2 (design s8.4). A condition that
#: spent its budget early keeps ticking mechanically; cutting all three runs at
#: the earliest exhaustion compares them over the stretch where all could act.
#: Every economic metric is recomputed there, not only the primary ones
#: (design s8.4, s11.2): each is its full-horizon namesake plus ``_common``.
COMMON_HORIZON_METRICS = (
    "volatility_shock_common",
    "volatility_shock_phase_common",
    "stockout_ticks_common",
    "traded_value_common",
    "useful_production_common",
    "gini_common",
    "stockout_ticks_inventory_common",
    "volatility_all_goods_common",
    "volume_all_goods_common",
    "stockouts_all_goods_common",
    "propagation_price_shock_common",
    "propagation_stockout_ticks_common",
)

#: Metrics defined on the scenario's target good. Where the scenario has none
#: (liquidity_loss, design s11.3) they do not exist, so they are neither
#: contrasted nor listed as unmeasurable: that would read as a thin market.
TARGET_GOOD_METRICS = frozenset(
    {
        "volatility_shock",
        "volatility_shock_phase",
        "stockout_ticks",
        "stockout_ticks_inventory",
        "volatility_shock_common",
        "volatility_shock_phase_common",
        "stockout_ticks_common",
        "stockout_ticks_inventory_common",
    }
)

#: Primary metric names in ``targets.primary_metrics`` mapped to the metric the
#: contrasts carry (design s11.3).
PRIMARY_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "volatility": ("volatility_shock", "volatility_shock_phase"),
    "stockouts": ("stockout_ticks",),
    "traded_value": ("traded_value",),
}

#: The three condition contrasts of one metric: one Holm family (s11.4).
CONDITION_PAIRS: tuple[tuple[str, str], ...] = (("A", "B"), ("A", "C"), ("B", "C"))

#: Metrics contrasted on fewer pairs, because a condition cannot vary on them:
#: A has no peer-to-peer messages, so its unanswered requests are always 0 and
#: A-B, A-C would test nothing. A is reported descriptively; the family is B-C.
METRIC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "unanswered_requests": (("B", "C"),),
}


#: Metrics whose missingness is caused by the treatment rather than by chance.
#: Market thinness is an *effect* of orchestration architecture, so restricting
#: a contrast to the surviving pairs would condition it on "the decentralized
#: conditions happened to trade" -- a selection bias on a primary metric
#: (design s11.3). Where measurability differs between the two conditions of a
#: pair, that pair's contrast is suppressed. For the window volatility the
#: phase-level companion carries the comparison (the value named here); the
#: propagated product's price and the all-goods volatility have no such
#: companion (None), so their measurability, reported as an outcome, carries it.
#: The Holm family stays the metric's three contrasts: a suppressed pair is
#: withheld, not untestable.
SELECTION_SENSITIVE_METRICS: dict[str, str | None] = {
    "volatility_shock": "volatility_shock_phase",
    "volatility_shock_common": "volatility_shock_phase_common",
    "propagation_price_shock": None,
    "propagation_price_shock_common": None,
    "volatility_all_goods": None,
    "volatility_all_goods_common": None,
}

#: Metrics whose measurability is reported as a companion outcome and tested
#: with the paired binary test (design s11.3). The phase-level companions are
#: here but never suppressed: s11.3 has the volatility comparison "sustained by
#: the phase volatility", so they are always contrasted on their measurable
#: pairs, with the dropped pairs reported.
MEASURABILITY_OUTCOME_METRICS = (
    "volatility_shock",
    "volatility_shock_phase",
    "volatility_shock_common",
    "volatility_shock_phase_common",
    "propagation_price_shock",
    "propagation_price_shock_common",
    "volatility_all_goods",
    "volatility_all_goods_common",
)


def applicable_metrics(runs: Iterable[RunMetrics]) -> tuple[str, ...]:
    """The contrast metrics that exist for this set of runs, in report order.

    A metric on the target good does not exist without one, the propagation
    block only exists for a raw-material target, the common-horizon block
    only in Experiment 2, and the inventory stockout variant only where the
    log supported the replay. Leaving them out is not dropping data: there was
    never a quantity to measure.
    """
    collected = list(runs)
    has_target = any(run.target_good is not None for run in collected)
    metrics = list(CONTINUOUS_METRICS)
    # Secondary blocks enter only where they were computed at all: a results
    # set built before they existed carries no such quantity, not a missing one.
    if any(run.stockouts_inventory is not None for run in collected):
        metrics.append("stockout_ticks_inventory")
    if any(run.multi_good for run in collected):
        metrics += ["volatility_all_goods", "volume_all_goods"]
    # Where the scenario has no target good (liquidity_loss), stockouts only
    # enter as this aggregate (design s11.3). A results set built before it
    # existed carries no such quantity.
    if any("stockouts" in (run.multi_good or {}) for run in collected):
        metrics.append("stockouts_all_goods")
    if any(run.propagation is not None for run in collected):
        metrics += PROPAGATION_METRICS
    # Each common-horizon metric enters only where it was computed (a results
    # set built before it existed carries no such key, not a missing value) and
    # only where its full-horizon namesake exists at all.
    present = set(metrics)
    metrics += [
        metric
        for metric in COMMON_HORIZON_METRICS
        if metric.removesuffix("_common") in present
        and any(metric.removesuffix("_common") in (run.common_horizon or {}) for run in collected)
    ]
    return tuple(m for m in metrics if has_target or m not in TARGET_GOOD_METRICS)


def metric_value(run: RunMetrics, metric: str) -> float | None:
    """One run's value for a metric -- the single primary value per run (s11.3).

    ``None`` means the run does not carry a usable value for this metric, not
    that the value is zero (design s11.3). Callers must drop the whole
    seed pair rather than substitute anything.
    """
    match metric:
        case "volatility_shock":
            value = run.volatility.get(PHASE_SHOCK)
            return None if value is None else float(value)
        case "volatility_shock_phase":
            value = run.volatility_phase.get(PHASE_SHOCK)
            return None if value is None else float(value)
        # One value per run, over the shock phase (design s11.3): measured over
        # the whole run, the base phase -- same world and seed across a trio,
        # though the agents' choices already differ there -- would dilute the
        # effect the contrast is looking for.
        case "stockout_ticks":
            # No target good (liquidity_loss): there is no stockout to count.
            if run.stockouts.get("good") is None:
                return None
            return float(run.stockouts["by_phase"][PHASE_SHOCK])
        case "traded_value":
            return float(run.volume_value[PHASE_SHOCK])
        case _ if metric in COMMON_HORIZON_METRICS:
            if run.common_horizon is None:
                return None
            return _optional(run.common_horizon.get(metric.removesuffix("_common")))
        case "stockout_ticks_inventory":
            if run.stockouts_inventory is None:
                return None
            return float(run.stockouts_inventory["by_phase"][PHASE_SHOCK])
        case "propagation_price_shock":
            if run.propagation is None:
                return None
            return _optional(run.propagation["mean_price"].get(PHASE_SHOCK))
        case "propagation_stockout_ticks":
            if run.propagation is None:
                return None
            return float(run.propagation["stockout_ticks"][PHASE_SHOCK])
        case "volatility_all_goods":
            return _optional((run.multi_good or {}).get("volatility"))
        case "volume_all_goods":
            return _optional((run.multi_good or {}).get("volume"))
        case "stockouts_all_goods":
            return _optional((run.multi_good or {}).get("stockouts"))
        case "useful_production":
            return _optional(run.useful_production.get("useful_fraction"))
        case "gini":
            return None if run.gini is None else float(run.gini)
        case "coordination_efficiency":
            return _optional(run.coordination.get("coordination_efficiency"))
        case "activation_concentration":
            return _optional(run.coordination.get("activation_concentration"))
        case "unanswered_requests":
            return float(run.unanswered_requests)
        case "tokens_total":
            return float(run.tokens["authoritative_total"])
        case "llm_calls":
            return float(run.coordination["llm_calls"])
    raise KeyError(f"unknown metric {metric!r}")


def _optional(value: float | None) -> float | None:
    return None if value is None else float(value)


def primary_metric_keys(run: RunMetrics) -> tuple[str, ...]:
    """The contrast metrics that are primary for this run's scenario (s11.3)."""
    keys: list[str] = []
    for name in run.primary_metrics:
        keys.extend(PRIMARY_METRIC_KEYS.get(name, ()))
    return tuple(keys)


@dataclass
class Results:
    """Everything the report needs, with no language in it."""

    scenario: str
    experiment: str
    runs: list[RunMetrics] = field(default_factory=list)
    contrasts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    survival: dict[str, Any] = field(default_factory=dict)
    measurability: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    suppressed_contrasts: dict[str, dict[str, Any]] = field(default_factory=dict)
    companion_outcomes: dict[str, Any] = field(default_factory=dict)
    survival_common_horizon: dict[str, Any] | None = None
    #: Recovery of the propagated product (secondary, design s11.3).
    survival_propagation: dict[str, Any] | None = None
    #: The same, at the trio's common active horizon (design s8.4).
    survival_propagation_common_horizon: dict[str, Any] | None = None
    #: Median units and value per good and phase, per condition (s11.1).
    volume_by_good: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)
    #: True when pilot seeds entered this analysis. Exploration only: pilot
    #: seeds are never used for inference (design s11.4), so a report built
    #: on them must say so.
    pilot_seeds_included: bool = False
    #: Seeds of this cell left out because their A/B/C trio was incomplete
    #: (design s11.3: pairs used and dropped are reported).
    incomplete_trios: list[int] = field(default_factory=list)
    #: Seeds outside the analysed seed set, left out of the whole analysis.
    skipped_seeds: list[int] = field(default_factory=list)
    #: Other cells ("scenario/experiment") left out whole for want of any
    #: complete trio, with the conditions their logs did cover.
    skipped_cells: dict[str, list[str]] = field(default_factory=dict)
    #: Logs left out before any check (seed or run phase), each with its reason.
    skipped_logs: list[dict[str, Any]] = field(default_factory=list)
    #: Per cell ("scenario/experiment"), the complete trios refused by the
    #: integrity check (design s12), each with its seed and reason.
    rejected_trios: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: Cells ("scenario/experiment") refused whole because their runs mix code
    #: versions, with the metadata fields that differ.
    refused_cells: dict[str, list[str]] = field(default_factory=dict)
    #: The metric parameters (W, X, M, T, D, ...) this analysis applied. They
    #: are not part of a run's identity, so the results record them.
    metric_parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_metrics(self) -> tuple[str, ...]:
        return primary_metric_keys(self.runs[0]) if self.runs else ()

    @property
    def conditions(self) -> list[str]:
        return sorted({run.condition for run in self.runs})

    @property
    def seeds(self) -> list[int]:
        return sorted({run.seed for run in self.runs})

    def by_condition(self, metric: str) -> dict[str, dict[int, float]]:
        """Per-seed values by condition, omitting seeds with no usable value.

        A missing value is absent from the mapping rather than present as zero,
        so a paired test drops that seed pair instead of comparing against a
        fabricated measurement.
        """
        table: dict[str, dict[int, float]] = {}
        for run in self.runs:
            value = metric_value(run, metric)
            table.setdefault(run.condition, {})
            if value is not None:
                table[run.condition][run.seed] = value
        return table

    def seeds_by_condition(self) -> dict[str, set[int]]:
        """Which seeds each condition ran, irrespective of any metric's value."""
        table: dict[str, set[int]] = {}
        for run in self.runs:
            table.setdefault(run.condition, set()).add(run.seed)
        return table

    def metric_measurability(self, metric: str) -> dict[str, dict[str, int]]:
        """How many runs per condition carried a usable value for a metric.

        Reported beside every summary the way the censoring rate is: a median
        computed over few runs must be read knowing how few.
        """
        census: dict[str, dict[str, int]] = {}
        for run in self.runs:
            entry = census.setdefault(run.condition, {"runs": 0, "measured": 0, "missing": 0})
            entry["runs"] += 1
            if metric_value(run, metric) is None:
                entry["missing"] += 1
            else:
                entry["measured"] += 1
        return census

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RESULTS_SCHEMA_VERSION,
            "scenario": self.scenario,
            "experiment": self.experiment,
            "conditions": self.conditions,
            "seeds": self.seeds,
            "primary_metrics": list(self.primary_metrics),
            "runs": [run.to_payload() for run in self.runs],
            "contrasts": self.contrasts,
            "survival": self.survival,
            "measurability": self.measurability,
            "suppressed_contrasts": self.suppressed_contrasts,
            "companion_outcomes": self.companion_outcomes,
            "survival_common_horizon": self.survival_common_horizon,
            "survival_propagation": self.survival_propagation,
            "survival_propagation_common_horizon": self.survival_propagation_common_horizon,
            "volume_by_good": self.volume_by_good,
            "system": self.system,
            "pilot_seeds_included": self.pilot_seeds_included,
            "incomplete_trios": sorted(self.incomplete_trios),
            "skipped_seeds": sorted(self.skipped_seeds),
            "skipped_cells": dict(sorted(self.skipped_cells.items())),
            "skipped_logs": list(self.skipped_logs),
            "rejected_trios": dict(sorted(self.rejected_trios.items())),
            "refused_cells": dict(sorted(self.refused_cells.items())),
            "metric_parameters": dict(sorted(self.metric_parameters.items())),
        }

    def write(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return destination
