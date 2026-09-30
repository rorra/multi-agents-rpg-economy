"""The two-stage pilot (design v1.0 s11.4, s13).

Calibration answers whether the configuration behaves with real LLM
agents -- the diagnostic list, not a result. It runs at a compressed horizon
because wall-clock and budget both bind, and because none of its questions need
340 ticks to answer.

Budget derivation produces the Experiment 2 budget the experiment then runs
with. It runs at the full horizon exactly, because s8.4 defines the budget as the
pooled median of runs at the horizon the finals will use; a budget derived from
shorter runs would be a third of what a final run needs and would exhaust
immediately.

The horizon difference between the stages is therefore not an inconsistency: one
stage may be compressed because its output is a judgement, and the other may not
because its output is a number the finals consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpg_economy.campaign.plan import CampaignPlan, PlanError, build_plan
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Scenario
from rpg_economy.logging_ import run_logs


@dataclass(frozen=True)
class PilotStage:
    """One stage of the pilot, and how it derives its config from the base one."""

    name: str
    horizon: int | None  # None means "the full horizon, unchanged"
    scenarios: tuple[Scenario, ...] | None  # None means "as configured"
    purpose: str
    #: The catalog key the CLI prints ``purpose`` with.
    purpose_key: str
    #: Run at campaign.pilot_calibration_horizon instead (``horizon`` is None).
    compressed: bool = False

    @property
    def uses_full_horizon(self) -> bool:
        return self.horizon is None and not self.compressed


CALIBRATION = PilotStage(
    name="calibration",
    horizon=None,
    compressed=True,
    scenarios=None,
    purpose="verify the configuration behaves with real agents",
    purpose_key="pilot.purpose.calibration",
)

BUDGET_DERIVATION = PilotStage(
    name="budget-derivation",
    horizon=None,
    scenarios=(Scenario.RESOURCE_SCARCITY,),
    purpose="derive the Experiment 2 budget at the horizon the finals will use",
    purpose_key="pilot.purpose.budget_derivation",
)

STAGES = {stage.name: stage for stage in (CALIBRATION, BUDGET_DERIVATION)}


def stage_config(base: SimulationConfig, stage: PilotStage) -> SimulationConfig:
    """Derive a stage's configuration from the base one.

    The only permitted derivation is the horizon, and the phase lengths that
    must follow it. Everything else -- K, population, recipes, thresholds, traits,
    prompts, model, inference parameters -- comes from the base configuration
    untouched, so a stage config differs from it in exactly one recorded
    dimension. Phases scale proportionally rather than being split evenly, so the
    compressed run keeps the full run's shape.
    """
    if stage.uses_full_horizon:
        return base

    horizon = (
        stage.horizon if stage.horizon is not None else base.campaign.pilot_calibration_horizon
    )
    full = base.time.horizon_ticks
    cycle = base.time.ticks_per_cycle
    if horizon % cycle or horizon >= full:
        raise PlanError(
            f"stage {stage.name}: horizon {horizon} must be a whole number of cycles "
            f"({cycle} ticks each) shorter than the full horizon {full}",
            key="error.pilot.bad_horizon",
            stage=stage.name,
            ticks=horizon,
            cycle=cycle,
            full=full,
        )
    base_ticks = round(base.phases.base_ticks * horizon / full)
    shock_ticks = round(base.phases.shock_ticks * horizon / full)
    recovery_ticks = horizon - base_ticks - shock_ticks

    if base_ticks < base.metrics.volatility_window_w:
        raise PlanError(
            f"stage {stage.name}: base phase {base_ticks} is shorter than the "
            f"volatility window {base.metrics.volatility_window_w}; the recovery "
            f"reference could not be computed",
            key="error.pilot.base_too_short",
            stage=stage.name,
            ticks=base_ticks,
            window=base.metrics.volatility_window_w,
        )
    if shock_ticks < base.metrics.volatility_window_w:
        raise PlanError(
            f"stage {stage.name}: shock phase {shock_ticks} is shorter than the "
            f"volatility window {base.metrics.volatility_window_w}; its volatility "
            f"could not be measured",
            key="error.pilot.shock_too_short",
            stage=stage.name,
            ticks=shock_ticks,
            window=base.metrics.volatility_window_w,
        )
    if recovery_ticks < base.metrics.recovery_hold_ticks_m:
        raise PlanError(
            f"stage {stage.name}: recovery phase {recovery_ticks} is shorter than "
            f"the hold requirement {base.metrics.recovery_hold_ticks_m}; recovery "
            f"could never be observed and every run would censor",
            key="error.pilot.recovery_too_short",
            stage=stage.name,
            ticks=recovery_ticks,
            hold=base.metrics.recovery_hold_ticks_m,
        )

    payload = base.model_dump(mode="python")
    payload["phases"] = {
        "base_ticks": base_ticks,
        "shock_ticks": shock_ticks,
        "recovery_ticks": recovery_ticks,
    }
    payload["time"] = {**payload["time"], "horizon_ticks": horizon}
    return SimulationConfig.model_validate(payload)


def stage_plan(base: SimulationConfig, stage: PilotStage) -> CampaignPlan:
    """The trios one stage will run, from the configured pilot seeds."""
    config = stage_config(base, stage)
    return build_plan(config, phase="pilot", scenarios=stage.scenarios)


def stage_summary(base: SimulationConfig, stage: PilotStage) -> dict[str, Any]:
    """Everything a launch review needs, with no side effects."""
    config = stage_config(base, stage)
    plan = stage_plan(base, stage)
    return {
        "stage": stage.name,
        "purpose": stage.purpose,
        "horizon": config.time.horizon_ticks,
        "full_horizon": base.time.horizon_ticks,
        "derived_from_base": not stage.uses_full_horizon,
        "phases": {
            "base": config.phases.base_ticks,
            "shock": config.phases.shock_ticks,
            "recovery": config.phases.recovery_ticks,
        },
        "k_activations": config.time.k_activations,
        "scenarios": [s.value for s in plan.by_scenario()],
        "seeds": list(config.seeds.pilot),
        "trios": len(plan.trios),
        "runs": plan.total_runs,
        "experiments": sorted({t.experiment.value for t in plan.trios}),
        "warnings": list(plan.warnings),
    }


#: Events that record one provider (or mock) call: agents' and the Director's.
_CALL_EVENTS = ("llm_call", "director_call")


def _billing_record(path: Path, root: Path) -> dict[str, Any] | None:
    """One log's billing facts; None for a log that never recorded its spend."""
    import json

    models: set[str] = set()
    calls = mock_calls = 0
    consumed: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:  # a log cut mid-line by a crash
            continue
        if event.get("type") in _CALL_EVENTS:
            model = (event.get("payload") or {}).get("model")
            if model is None:  # a Director call that never reached the provider
                continue
            models.add(str(model))
            calls += 1
            mock_calls += str(model).startswith("mock")
        elif event.get("type") == "tokens_consumed":
            consumed = event["payload"]
    record: dict[str, Any] = {
        "log": str(path.relative_to(root)),
        "models": sorted(models),
        "llm_calls": calls,
        # Calls answered by a mock model, counted per call.
        "mock_calls": mock_calls,
    }
    if not consumed:
        return None
    authoritative = int(consumed.get("authoritative_total", 0))
    billed = int(consumed.get("billed_total", 0))
    return {
        **record,
        "authoritative_total": authoritative,
        "billed_total": billed,
        "fully_billed": billed == authoritative and authoritative > 0,
    }


def _call_billed(payload: dict[str, Any]) -> bool:
    """Whether one call event was billed.

    The call's own ``billed`` flag, which the client sets from the provider's
    response. Logs written before it was recorded fall back to the model name:
    a mock model is never billed and a real one always is.
    """
    billed = payload.get("billed")
    if isinstance(billed, bool):
        return billed
    return not str(payload.get("model")).startswith("mock")


def log_spend(path: Path) -> dict[str, Any]:
    """What one log recorded as spent, finished or not.

    From the best record the log holds: the ``tokens_consumed`` ledger of a
    finished run, else the ledger snapshot a ``run_aborted`` event carries,
    else the sum of the billed ``llm_call`` and ``director_call`` events (a
    crashed run has neither ledger). ``source`` says which.
    """
    import json

    ledger: dict[str, Any] | None = None
    aborted: dict[str, Any] | None = None
    from_calls = 0
    calls = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:  # a log cut mid-line by a crash
            continue
        payload = event.get("payload") or {}
        kind = event.get("type")
        if kind in _CALL_EVENTS and payload.get("model") is not None:
            calls += 1
            if _call_billed(payload):
                from_calls += int(payload.get("input_tokens", 0)) + int(
                    payload.get("output_tokens", 0)
                )
        elif kind == "tokens_consumed":
            ledger = payload
        elif kind == "run_aborted" and "billed_total" in payload:
            aborted = payload
    if ledger is not None:
        return {
            "llm_calls": calls,
            "billed_total": int(ledger.get("billed_total", 0)),
            "source": "tokens_consumed",
        }
    if aborted is not None:
        return {
            "llm_calls": calls,
            "billed_total": int(aborted.get("billed_total", 0)),
            "source": "run_aborted",
        }
    return {"llm_calls": calls, "billed_total": from_calls, "source": "calls"}


def _summed_spend(root: Path, logs: list[Path]) -> dict[str, Any]:
    """The spend of ``logs``, totalled, with each log's own figure."""
    detail = []
    for path in logs:
        spend = log_spend(path)
        detail.append({"log": str(path.relative_to(root)), **spend})
    return {
        "logs": len(logs),
        "llm_calls": sum(d["llm_calls"] for d in detail),
        "billed_total": sum(d["billed_total"] for d in detail),
        "detail": detail,
    }


def _set_aside_billed(root: Path) -> dict[str, Any]:
    """Spend recorded by logs set aside before a re-run (``previous-N/``).

    They are never results, but they were paid for: a partial real run is
    billed for every call it made (``log_spend``).
    """
    from rpg_economy.logging_.writer import SET_ASIDE_PREFIX

    if not root.exists():
        return {"logs": 0, "llm_calls": 0, "billed_total": 0}
    logs = sorted(
        path
        for path in root.rglob("*.jsonl")
        if any(part.startswith(SET_ASIDE_PREFIX) for part in path.relative_to(root).parts)
    )
    summed = _summed_spend(root, logs)
    summed.pop("detail")
    return summed


def verify_stage_billing(
    directory: Path | str, *, expected_runs: int | None = None
) -> dict[str, Any]:
    """Billing verification (IMPLEMENTATION.md section 4.7) for a whole stage.

    Every run must be real and fully billed. A stage that mixed real and mock
    spend, or that silently fell back to a mock, measures nothing. With
    ``expected_runs`` (the stage plan's run count) the stage must also be
    whole: an unfinished or missing run fails it, however well the finished
    ones were billed.

    Only finished logs are runs. Unfinished ones (an abort, a crash, a third
    Ctrl-C) and logs set aside before a re-run are reported apart
    (``unfinished_spend``, ``set_aside``): they are spend, never results.
    ``billed_total_all`` is everything the directory records as billed.
    """
    from rpg_economy.logging_ import log_is_finished

    root = Path(directory)
    runs: list[dict[str, Any]] = []
    unfinished: list[Path] = []
    for path in run_logs(root) if root.exists() else []:
        if not log_is_finished(path):
            unfinished.append(path)
            continue
        record = _billing_record(path, root)
        if record is not None:
            runs.append(record)

    finished = len(runs)
    complete = not unfinished and (expected_runs is None or finished == expected_runs)
    set_aside = _set_aside_billed(root)
    unfinished_spend = _summed_spend(root, unfinished)
    billed_total = sum(r["billed_total"] for r in runs)
    return {
        "runs": finished,
        "expected_runs": expected_runs,
        "unfinished": [str(path.relative_to(root)) for path in unfinished],
        "unfinished_spend": unfinished_spend,
        "missing": max(0, (expected_runs or 0) - finished),
        "complete": complete,
        "all_real": all(r["mock_calls"] == 0 for r in runs) if runs else False,
        "all_fully_billed": all(r["fully_billed"] for r in runs) if runs else False,
        "billed_total": billed_total,
        "set_aside": set_aside,
        "billed_total_with_set_aside": billed_total + set_aside["billed_total"],
        "billed_total_all": billed_total
        + unfinished_spend["billed_total"]
        + set_aside["billed_total"],
        "detail": runs,
    }


def billing_passed(report: dict[str, Any]) -> bool:
    """Whether a stage's billing report is evidence: whole, real, fully billed."""
    return bool(report["complete"] and report["all_real"] and report["all_fully_billed"])
