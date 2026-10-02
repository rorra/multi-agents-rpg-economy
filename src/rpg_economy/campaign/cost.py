"""Cost projection for a planned campaign (IMPLEMENTATION.md section 7).

"Never launch multi-seed real-LLM batches without an explicit cost estimate
printed first." That rule is enforced here and in the CLI: an unpriced batch
cannot be undone by re-running something.

The projection multiplies measured mock token counts by the configured list
price. Mock runs consume the same prompts and the same number of calls as real
ones -- the mock replaces the provider's response, not the orchestration -- so
the call structure is exact. What the mock cannot know is how long a real
model's replies run, so output tokens are the estimate's soft spot and are
reported separately rather than folded invisibly into one number.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from rpg_economy.campaign.parallel import MAX_PROVIDER_ATTEMPTS
from rpg_economy.campaign.plan import CampaignPlan, Notice
from rpg_economy.config.models import ModelPricing, SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime, Scenario
from rpg_economy.logging_ import last_event, log_is_finished, read_metadata, run_logs
from rpg_economy.metrics.run_metrics import RunLog, token_totals

CENT = Decimal("0.01")

#: How much the measured mock output length is allowed to understate a real
#: model's. Applied only to the *upper* bound, never to the headline figure, so
#: the estimate is never quietly inflated -- but a batch is never authorised
#: against a number that assumed the cheapest possible outcome either.
OUTPUT_UNCERTAINTY_MULTIPLIER = Decimal("3")

#: Completion tokens per call, measured on the real provider at
#: reasoning_effort=low. The mock's own output is ~50 tokens per call, an order
#: of magnitude low, because a mock does not reason; a profile built from mock
#: logs therefore supplies the input side only, and output is taken from here
#: (IMPLEMENTATION.md section 7).
MEASURED_OUTPUT_PER_CALL = 478


class CostError(ValueError):
    """A campaign cannot be priced.

    ``key`` and ``params`` name the catalog message the CLI prints; the English
    text stays the exception's message for library callers.
    """

    def __init__(self, message: str, *, key: str | None = None, **params: Any) -> None:
        super().__init__(message)
        self.key = key
        self.params = params


@dataclass(frozen=True)
class RunTokens:
    """One measured run's input and output tokens."""

    condition: Condition
    scenario: Scenario
    experiment: ExperimentRegime
    seed: int
    input_tokens: int
    output_tokens: int
    calls: int = 0
    #: The run's length in ticks, from its run_end. None when unknown, in which
    #: case the run is taken to have the planned horizon.
    ticks: int | None = None

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class TokenProfile:
    """Measured per-run token consumption, keyed by what varies between runs.

    Keyed on (scenario, experiment, condition) rather than pooled, because the
    conditions differ in cost by design -- A carries the Director, C carries the
    bulletin -- and pooling them would price a campaign at an average no
    individual run has.
    """

    measured: dict[tuple[str, str, str], list[RunTokens]] = field(default_factory=dict)
    #: Logs left out because they never reached run_end.
    skipped: int = 0
    #: Every measured run used the real provider: its output is a forecast.
    real: bool = False

    def record(self, run: RunTokens) -> None:
        key = (run.scenario.value, run.experiment.value, run.condition.value)
        self.measured.setdefault(key, []).append(run)

    def median_for(
        self,
        scenario: Scenario,
        experiment: ExperimentRegime,
        condition: Condition,
        *,
        horizon: int | None = None,
    ) -> tuple[int, int] | None:
        """Median input and output tokens for one cell, or None if unmeasured.

        With ``horizon``, each run is scaled to that many ticks first: a profile
        measured on short runs must not price a long campaign as if it were short.
        """
        runs = self.measured.get((scenario.value, experiment.value, condition.value))
        if not runs:
            return None
        return (
            int(statistics.median(_scaled(r.input_tokens, r, horizon) for r in runs)),
            int(statistics.median(_scaled(r.output_tokens, r, horizon) for r in runs)),
        )

    def fallback(
        self, condition: Condition, *, horizon: int | None = None
    ) -> tuple[int, int] | None:
        """Median across every scenario for one condition.

        Used when a planned cell has no pilot measurement of its own. Recorded in
        the estimate as an extrapolation so the number is never mistaken for a
        measurement of the cell it prices.
        """
        runs = [
            r
            for (_, _, cond), values in self.measured.items()
            if cond == condition.value
            for r in values
        ]
        if not runs:
            return None
        return (
            int(statistics.median(_scaled(r.input_tokens, r, horizon) for r in runs)),
            int(statistics.median(_scaled(r.output_tokens, r, horizon) for r in runs)),
        )

    def calls_for(
        self,
        condition: Condition,
        *,
        scenario: Scenario | None = None,
        experiment: ExperimentRegime | None = None,
        horizon: int | None = None,
    ) -> int:
        """Median call count for one cell, or for one condition across every cell.

        With ``scenario`` and ``experiment``, the median of that cell's own runs,
        the same runs ``median_for`` prices its input from; an Experiment 2 cell
        stops calling at its budget, so pooling it with Experiment 1 would price
        its output at calls it never makes. A cell with no measurement of its
        own falls back to the condition across every cell, as ``fallback`` does.
        """
        runs: list[RunTokens] = []
        if scenario is not None and experiment is not None:
            runs = self.measured.get((scenario.value, experiment.value, condition.value), [])
        if not runs:
            runs = [
                r
                for (_, _, cond), values in self.measured.items()
                if cond == condition.value
                for r in values
            ]
        return int(statistics.median(_scaled(r.calls, r, horizon) for r in runs)) if runs else 0

    @property
    def runs_measured(self) -> int:
        return sum(len(runs) for runs in self.measured.values())


def _scaled(value: int, run: RunTokens, horizon: int | None) -> float:
    """``value`` as it would be over ``horizon`` ticks, at the run's own rate."""
    if horizon is None or not run.ticks:
        return value
    return value * horizon / run.ticks


def _run_ticks(path: Path) -> int | None:
    """Ticks of a finished run, from its closing run_end; None if unfinished."""
    if not log_is_finished(path):
        return None
    ticks = (last_event(path) or {}).get("payload", {}).get("ticks")
    return int(ticks) if ticks else None


def measure_profile(log_directory: Path | str) -> TokenProfile:
    """Read every run log under a directory into a token profile.

    Reads logs, never live state, so a profile can be built from any past mock
    batch without re-running it.
    """
    profile = TokenProfile()
    mocks: list[bool] = []
    for path in run_logs(log_directory):
        try:
            metadata = read_metadata(path)
        except Exception:  # a non-run .jsonl is skipped, not fatal
            continue
        ticks = _run_ticks(path)
        if ticks is None:
            # An unfinished run understates what a whole one costs.
            profile.skipped += 1
            continue
        tokens = token_totals(RunLog(path))
        mocks.append(bool(metadata.llm_mock))
        profile.record(
            RunTokens(
                condition=metadata.condition,
                scenario=metadata.scenario,
                experiment=metadata.experiment,
                seed=metadata.seed,
                input_tokens=sum(tokens["input_by_bucket"].values()),
                output_tokens=sum(tokens["output_by_bucket"].values()),
                calls=sum(tokens["calls_by_bucket"].values()),
                ticks=ticks,
            )
        )
    profile.real = bool(mocks) and not any(mocks)
    return profile


@dataclass
class CostEstimate:
    """A priced campaign plan."""

    currency: str
    quoted_on: str
    model: str
    total_runs: int
    input_tokens: int
    output_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    upper_bound: Decimal
    #: Cost if every run used every call the caps allow (K agent activations
    #: and, in A, K + 1 Director calls per cycle, each agent call retried once)
    #: with every reply at the output cap. What a provider spend limit must
    #: cover so a stage is never cut off half-way.
    worst_case: Decimal | None = None
    #: Attempts a run may take when the provider fails transiently (see
    #: ``campaign.parallel``). A retry re-runs only the failed condition, but
    #: each failed attempt was billed up to where it stopped, so a run can cost
    #: up to this many times its own price.
    retry_attempts: int = 1
    extrapolated_cells: list[str] = field(default_factory=list)
    per_scenario: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: The plan's warnings among ``warnings``, which a caller printing the plan
    #: has already shown.
    plan_warnings: list[str] = field(default_factory=list)
    #: Priced from real runs (their measured output), not from the mock.
    real: bool = False

    @property
    def total_cost(self) -> Decimal:
        return self.input_cost + self.output_cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def worst_case_with_retries(self) -> Decimal | None:
        """The worst case if every run failed transiently on every attempt but the last."""
        if self.worst_case is None:
            return None
        return (self.worst_case * self.retry_attempts).quantize(CENT, rounding=ROUND_HALF_UP)

    def to_payload(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "quoted_on": self.quoted_on,
            "model": self.model,
            "total_runs": self.total_runs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": str(self.input_cost),
            "output_cost": str(self.output_cost),
            "total_cost": str(self.total_cost),
            "upper_bound": str(self.upper_bound),
            "worst_case": None if self.worst_case is None else str(self.worst_case),
            "retry_attempts": self.retry_attempts,
            "worst_case_with_retries": (
                None if self.worst_case_with_retries is None else str(self.worst_case_with_retries)
            ),
            "extrapolated_cells": list(self.extrapolated_cells),
            "per_scenario": self.per_scenario,
            "warnings": list(self.warnings),
        }


def _worst_case_run(
    config: SimulationConfig,
    condition: Condition,
    input_per_call: float,
    horizon: int | None,
    *,
    budget: int | None = None,
) -> tuple[int, int]:
    """Input and output tokens of one run at every cap (see ``worst_case``).

    ``input_per_call`` is the measured input of one call, taken before any
    Experiment 2 cap: a capped run's input over its uncapped call count would
    understate every call. With ``budget``, the run stops calling once the
    budget is spent, so it can exceed it by at most the one call that crossed
    it: its input and a reply at the output cap.
    """
    ticks = horizon if horizon is not None else config.time.horizon_ticks
    cycles = -(-ticks // config.time.ticks_per_cycle)
    k = config.time.k_activations
    calls = k * cycles * 2  # every activation, each retried once
    if condition is Condition.A:
        calls += (k + 1) * cycles  # every Director call the cap allows
    worst_input = int(input_per_call * calls)
    worst_output = calls * config.llm.max_output_tokens
    if budget is not None:
        bound = budget + int(input_per_call) + config.llm.max_output_tokens
        if worst_input + worst_output > bound:
            share = Decimal(bound) / Decimal(worst_input + worst_output)
            worst_input = int(worst_input * share)
            worst_output = bound - worst_input
    return worst_input, worst_output


def _upper_run(run_input: int, run_output: int, *, ceiling: int | None) -> tuple[int, int]:
    """Input and output tokens of one run at the upper bound.

    Only output is inflated (``OUTPUT_UNCERTAINTY_MULTIPLIER``): input length
    is known exactly. With ``ceiling`` (an Experiment 2 run's budget plus the
    one call that can cross it), the inflated run stops there too, as the real
    one would: a budget run cannot cost more than its budget allows, however
    long the model's replies turn out to be.
    """
    upper_output = int(run_output * OUTPUT_UNCERTAINTY_MULTIPLIER)
    if ceiling is not None and run_input + upper_output > ceiling:
        share = Decimal(ceiling) / Decimal(run_input + upper_output)
        run_input = int(run_input * share)
        upper_output = ceiling - run_input
    return run_input, upper_output


def estimate_cost(
    plan: CampaignPlan,
    profile: TokenProfile,
    pricing: ModelPricing,
    *,
    model: str,
    horizon: int | None = None,
    config: SimulationConfig | None = None,
) -> CostEstimate:
    """Price a plan from measured mock token counts.

    With ``config``, an Experiment 2 run is priced no higher than its token
    budget, and the worst case at the caps is computed as well.

    Raises when a planned condition has no measurement anywhere in the profile.
    Guessing there would defeat the purpose: the estimate exists to make a real
    batch's cost known before it is authorised, and a fabricated component makes
    the total unusable for that.
    """
    if profile.runs_measured == 0:
        raise CostError(
            "no measured runs: build a mock profile before pricing a campaign",
            key="error.cost.no_profile",
        )

    input_tokens = output_tokens = 0
    calls = 0
    worst_input = worst_output = 0
    # The upper bound's tokens, run by run, so each run's is capped by its own
    # budget ceiling (see ``_upper_run``).
    upper_input = upper_output = 0
    extrapolated: list[str] = []
    per_scenario: dict[str, dict[str, Any]] = {}

    for trio in plan.trios:
        for condition in trio.conditions:
            measurement = profile.median_for(
                trio.scenario, trio.experiment, condition, horizon=horizon
            )
            if measurement is None:
                measurement = profile.fallback(condition, horizon=horizon)
                if measurement is None:
                    raise CostError(
                        f"no measured runs for condition {condition.value}; cannot price "
                        f"{trio.scenario.value}/{trio.experiment.value}",
                        key="error.cost.unmeasured",
                        condition=condition.value,
                        scenario=trio.scenario.value,
                        experiment=trio.experiment.value,
                    )
                cell = f"{trio.scenario.value}/{trio.experiment.value}/{condition.value}"
                if cell not in extrapolated:
                    extrapolated.append(cell)

            run_input, run_output = measurement
            run_calls = profile.calls_for(
                condition, scenario=trio.scenario, experiment=trio.experiment, horizon=horizon
            )
            # Per call, before any cap below: the cap shortens a run, never a call.
            input_per_call = run_input / run_calls if run_calls else 0.0
            # Mock output is not a forecast of a reasoning model's output; use
            # the measured real rate against the mock's call count. A profile
            # of real runs already measured it.
            if not profile.real:
                run_output = run_calls * MEASURED_OUTPUT_PER_CALL if run_calls else run_output
            cap = None
            if config is not None and trio.experiment is ExperimentRegime.EQUAL_COST:
                cap = config.budget.experiment_2_total_tokens
                if cap is not None and run_input + run_output > cap:
                    # A run stops calling once its budget is spent.
                    share = Decimal(cap) / Decimal(run_input + run_output)
                    run_input = int(run_input * share)
                    run_output = cap - run_input
            calls += run_calls
            input_tokens += run_input
            output_tokens += run_output
            u_in, u_out = _upper_run(
                run_input,
                run_output,
                ceiling=(
                    None
                    if cap is None or config is None
                    else cap + int(input_per_call) + config.llm.max_output_tokens
                ),
            )
            upper_input += u_in
            upper_output += u_out
            if config is not None:
                w_in, w_out = _worst_case_run(
                    config, condition, input_per_call, horizon, budget=cap
                )
                worst_input += w_in
                worst_output += w_out

            bucket = per_scenario.setdefault(
                trio.scenario.value, {"runs": 0, "input_tokens": 0, "output_tokens": 0}
            )
            bucket["runs"] += 1
            bucket["input_tokens"] += run_input
            bucket["output_tokens"] += run_output

    million = Decimal(1_000_000)
    input_cost = (Decimal(input_tokens) / million * pricing.input_per_1m_tokens).quantize(
        CENT, rounding=ROUND_HALF_UP
    )
    output_cost = (Decimal(output_tokens) / million * pricing.output_per_1m_tokens).quantize(
        CENT, rounding=ROUND_HALF_UP
    )
    upper_bound = (
        Decimal(upper_input) / million * pricing.input_per_1m_tokens
        + Decimal(upper_output) / million * pricing.output_per_1m_tokens
    ).quantize(CENT, rounding=ROUND_HALF_UP)

    worst_case = None
    if config is not None:
        worst_case = (
            Decimal(worst_input) / million * pricing.input_per_1m_tokens
            + Decimal(worst_output) / million * pricing.output_per_1m_tokens
        ).quantize(CENT, rounding=ROUND_HALF_UP)

    for bucket in per_scenario.values():
        bucket["cost"] = str(
            (
                Decimal(bucket["input_tokens"]) / million * pricing.input_per_1m_tokens
                + Decimal(bucket["output_tokens"]) / million * pricing.output_per_1m_tokens
            ).quantize(CENT, rounding=ROUND_HALF_UP)
        )

    warnings = list(plan.warnings)
    if extrapolated:
        warnings.append(
            Notice(
                f"{len(extrapolated)} planned cell(s) priced by extrapolation from other "
                f"scenarios; no pilot measurement exists for them",
                "campaign.warning.extrapolated",
                count=len(extrapolated),
            )
        )

    return CostEstimate(
        currency=pricing.currency,
        quoted_on=pricing.quoted_on,
        model=model,
        total_runs=plan.total_runs,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        upper_bound=upper_bound,
        worst_case=worst_case,
        retry_attempts=MAX_PROVIDER_ATTEMPTS,
        extrapolated_cells=extrapolated,
        per_scenario=per_scenario,
        warnings=warnings,
        plan_warnings=list(plan.warnings),
        real=profile.real,
    )
