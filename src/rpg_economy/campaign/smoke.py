"""Real-provider smoke verification (authorized, budget-capped).

The mock exercises every code path the provider path uses -- prompt assembly,
parsing, validation, retries, token accounting -- but it supplies the usage
figures itself. Two things can therefore only be checked against the real API:
that the provider's usage fields are read into the ledger correctly, and that
budget enforcement binds on the provider's total rather than on our estimate
(design v1.0 s8.4).

These checks are assertions about a completed run's log, so they can be re-run
on a stored log without spending anything again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.logging_ import EventType, read_events, read_metadata

#: The smallest shape the config validators accept: two ticks per phase, with
#: the volatility window and recovery hold shrunk to match. Chosen so a real
#: provider run costs cents while still crossing both phase boundaries.
SMOKE_PHASE_TICKS = 2
SMOKE_WINDOW = 2

#: Measured on the equivalent mock trio at this size, plus condition A again for
#: SMOKE-2. Used only to price the runs before they are authorised; the ledger,
#: not this, is what the budget and the report use.
SMOKE_CALLS = 214 + 33
SMOKE_INPUT_TOKENS = 220_485 + 34_790

# Output tokens per call live in campaign.cost.MEASURED_OUTPUT_PER_CALL, taken
# from a whole real run rather than a handful of isolated probes. There is one
# constant so two literals cannot drift apart.


def smoke_config(base, *, experiment: str, budget: int | None = None):
    """The smallest real-provider configuration, derived from the base configuration.

    Everything except size comes from the configuration, so the smoke test
    exercises the same prompts, traits, recipes and thresholds the real runs
    will use -- a bespoke miniature economy would verify a different system.
    """
    from rpg_economy.config.models import SimulationConfig

    payload = base.model_dump(mode="python")
    payload["llm"] = {**payload["llm"], "mock": False}
    payload["phases"] = {
        "base_ticks": SMOKE_PHASE_TICKS,
        "shock_ticks": SMOKE_PHASE_TICKS,
        "recovery_ticks": SMOKE_PHASE_TICKS,
    }
    payload["time"] = {**payload["time"], "horizon_ticks": SMOKE_PHASE_TICKS * 3}
    payload["metrics"] = {
        **payload["metrics"],
        "volatility_window_w": SMOKE_WINDOW,
        "recovery_hold_ticks_m": SMOKE_WINDOW,
    }
    payload["run"] = {**payload["run"], "experiment": experiment}
    if budget is not None:
        payload["budget"] = {**payload["budget"], "experiment_2_total_tokens": budget}
    return SimulationConfig.model_validate(payload)


@dataclass
class Check:
    """One named verification and what it found."""

    name: str
    passed: bool
    detail: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class SmokeReport:
    """Everything the two smoke runs are meant to establish."""

    label: str
    log_path: str
    model: str
    experiment: str
    llm_calls: int = 0
    director_calls: int = 0
    provider_input: int = 0
    provider_output: int = 0
    ledger_total: int = 0
    billed_total: int = 0
    billed_calls: int = 0
    #: False when the log carries no billing provenance at all
    #: (IMPLEMENTATION.md section 4.7). Absent is not zero: a log that never recorded the field
    #: cannot be retro-verified, and reporting it as "billed 0" would read as
    #: "this run was free" rather than "this run cannot say".
    billing_recorded: bool = True
    mock_calls: int = 0
    models_seen: list[str] = field(default_factory=list)
    exhaustion_tick: int | None = None
    horizon_reached: int = 0
    planned_horizon: int = 0
    calls_after_exhaustion: int = 0
    ticks_after_exhaustion: int = 0
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def provider_total(self) -> int:
        return self.provider_input + self.provider_output

    def to_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "log_path": self.log_path,
            "model": self.model,
            "experiment": self.experiment,
            "llm_calls": self.llm_calls,
            "director_calls": self.director_calls,
            "provider_input": self.provider_input,
            "provider_output": self.provider_output,
            "provider_total": self.provider_total,
            "ledger_total": self.ledger_total,
            "billed_total": self.billed_total,
            "billed_calls": self.billed_calls,
            "billing_recorded": self.billing_recorded,
            "mock_calls": self.mock_calls,
            "models_seen": sorted(self.models_seen),
            "exhaustion_tick": self.exhaustion_tick,
            "horizon_reached": self.horizon_reached,
            "planned_horizon": self.planned_horizon,
            "calls_after_exhaustion": self.calls_after_exhaustion,
            "ticks_after_exhaustion": self.ticks_after_exhaustion,
            "passed": self.passed,
            "checks": [c.to_payload() for c in self.checks],
        }


def verify_smoke(
    log_path: Path | str, *, label: str, planned_horizon: int, expect_exhaustion: bool
) -> SmokeReport:
    """Check one completed real-provider run against the invariants it targets."""
    path = Path(log_path)
    metadata = read_metadata(path)
    events = list(read_events(path))

    report = SmokeReport(
        label=label,
        log_path=str(path),
        model=metadata.llm_model,
        experiment=metadata.experiment.value,
        planned_horizon=planned_horizon,
    )

    exhausted_seq: int | None = None
    per_call_mismatches: list[str] = []
    zero_usage_calls = 0

    for event in events:
        payload = event.payload
        match event.type:
            case EventType.DIRECTOR_CALL:
                # The Director's calls are logged under their own event type, so
                # a total summed from llm_call alone would omit every one of them.
                report.director_calls += 1
                report.provider_input += int(payload.get("input_tokens", 0))
                report.provider_output += int(payload.get("output_tokens", 0))
                if exhausted_seq is not None and event.seq > exhausted_seq:
                    report.calls_after_exhaustion += 1
            case EventType.LLM_CALL:
                report.llm_calls += 1
                provider_in = int(payload.get("input_tokens", 0))
                provider_out = int(payload.get("output_tokens", 0))
                report.provider_input += provider_in
                report.provider_output += provider_out
                if provider_in == 0:
                    zero_usage_calls += 1
                model = str(payload.get("model", ""))
                if model and model not in report.models_seen:
                    report.models_seen.append(model)
                if model.startswith("mock"):
                    report.mock_calls += 1
                buckets = payload.get("input_by_bucket") or {}
                # Buckets are analytical attribution of the provider's figure
                # (design v1.0 s8.4). If they do not sum to it, every per-bucket
                # number is wrong.
                if sum(int(v) for v in buckets.values()) != provider_in:
                    per_call_mismatches.append(
                        f"seq {event.seq}: buckets {sum(int(v) for v in buckets.values())} "
                        f"vs provider input {provider_in}"
                    )
                if exhausted_seq is not None and event.seq > exhausted_seq:
                    report.calls_after_exhaustion += 1
            case EventType.BUDGET_EXHAUSTED:
                exhausted_seq = event.seq
                report.exhaustion_tick = event.tick
            case EventType.TOKENS_CONSUMED:
                report.ledger_total = int(payload.get("authoritative_total", 0))
                report.billing_recorded = "billed_total" in payload
                report.billed_total = int(payload.get("billed_total", 0))
                report.billed_calls = int(payload.get("billed_calls", 0))
            case EventType.TICK:
                report.horizon_reached = max(report.horizon_reached, event.tick)
                if exhausted_seq is not None and event.seq > exhausted_seq:
                    report.ticks_after_exhaustion += 1
            case _:
                pass

    add = report.checks.append
    add(
        Check(
            "llm_calls_made",
            report.llm_calls > 0,
            f"{report.llm_calls} agent calls, {report.director_calls} Director calls",
        )
    )
    add(
        Check(
            "every_call_was_real",
            report.mock_calls == 0,
            f"{report.mock_calls} mock call(s); models seen: {sorted(report.models_seen)}"
            if report.mock_calls
            else f"models seen: {sorted(report.models_seen)}",
        )
    )
    if not report.billing_recorded:
        add(
            Check(
                "billed_matches_ledger",
                True,
                "not applicable: this log records no billing provenance, so it "
                "cannot contribute to a verified spend total",
            )
        )
    else:
        add(
            Check(
                "billed_matches_ledger",
                report.billed_total == report.ledger_total,
                f"billed {report.billed_total} of ledger {report.ledger_total}"
                + (
                    " -- a run that mixed real and mock spend"
                    if 0 < report.billed_total < report.ledger_total
                    else ""
                ),
            )
        )
    add(
        Check(
            "provider_usage_parsed",
            zero_usage_calls == 0,
            f"{zero_usage_calls} call(s) reported zero input tokens"
            if zero_usage_calls
            else "every call carried provider usage",
        )
    )
    add(
        Check(
            "buckets_reconcile_per_call",
            not per_call_mismatches,
            "; ".join(per_call_mismatches[:3]) if per_call_mismatches else "exact on every call",
        )
    )
    add(
        Check(
            "ledger_matches_provider_total",
            report.ledger_total == report.provider_total,
            f"ledger {report.ledger_total} vs summed provider {report.provider_total}",
        )
    )
    add(
        Check(
            "horizon_reached",
            report.horizon_reached >= planned_horizon - 1,
            f"last tick {report.horizon_reached} of planned {planned_horizon}",
        )
    )

    if expect_exhaustion:
        add(
            Check(
                "budget_exhausted",
                report.exhaustion_tick is not None,
                f"exhausted at tick {report.exhaustion_tick}"
                if report.exhaustion_tick is not None
                else "never exhausted -- lower the budget",
            )
        )
        add(
            Check(
                "exhausted_mid_run",
                report.exhaustion_tick is not None and report.exhaustion_tick < planned_horizon - 1,
                f"tick {report.exhaustion_tick} of {planned_horizon}",
            )
        )
        add(
            Check(
                "no_calls_after_exhaustion",
                report.calls_after_exhaustion == 0,
                f"{report.calls_after_exhaustion} call(s) after exhaustion",
            )
        )
        add(
            Check(
                "mechanical_ticks_continue",
                report.ticks_after_exhaustion > 0,
                f"{report.ticks_after_exhaustion} tick(s) after exhaustion",
            )
        )
    else:
        add(
            Check(
                "no_exhaustion",
                report.exhaustion_tick is None,
                "unconstrained run did not exhaust"
                if report.exhaustion_tick is None
                else f"unexpectedly exhausted at {report.exhaustion_tick}",
            )
        )

    add(
        Check(
            "log_reconstructs",
            _reconstructs(events),
            "run_metadata first, gapless sequence, run_end present",
        )
    )
    return report


def _reconstructs(events: list[Any]) -> bool:
    """The s15 requirement, checked structurally rather than asserted."""
    if not events:
        return False
    if events[0].type is not EventType.RUN_METADATA:
        return False
    if [e.seq for e in events] != list(range(len(events))):
        return False
    required = {
        EventType.OBSERVATION_DELIVERED,
        EventType.AGENT_OUTPUT,
        EventType.ACTION_EXECUTED,
        EventType.LLM_CALL,
        EventType.TOKENS_CONSUMED,
        EventType.RUN_END,
    }
    return required <= {e.type for e in events}
