"""The model de-risk comparison (design v1.0 s13).

The design fixes ``gpt-5-nano`` for the whole project on budget grounds and
accepts a known risk: if the model is too weak to use the Director's TASK, Condition A's
advantage shrinks for a reason unrelated to centralization, and a floor effect
would read as "centralization does not help".

This module measures that risk so it can be documented. It does not choose a
model, and nothing here may be wired into a run: it compares two models on the
same seed and reports the three rates that would reveal a model failing to work
the protocol at all.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition
from rpg_economy.logging_ import EventType, read_events

#: What a model unable to work the protocol would show: malformed replies, an
#: activation spent achieving nothing, and a Director whose output will not
#: validate. All three are already logged; none needs new instrumentation.
DERISK_METRICS = ("invalid_output_rate", "wasted_activation_rate", "director_error_rate")

#: Actions that read as a response to each TASK type. A heuristic indicator,
#: not a compliance requirement: design v1.0 s3.1.3 has the Director assign a
#: task and the agent decide within its role, so a mismatch is a legitimate
#: choice, not a violation. What the rate measures is whether the model can
#: use the Director's guidance at all, which is the floor-effect question above.
TASK_ALIGNED_ACTIONS: dict[str, frozenset[str]] = {
    "ANALYZE_MARKET": frozenset({"buy", "sell", "send_message", "wait"}),
    "EVALUATE_EXPEDITION": frozenset({"start_expedition", "equip", "wait"}),
    "EVALUATE_PRODUCTION": frozenset({"produce", "buy", "wait"}),
    "MANAGE_PROCUREMENT": frozenset({"buy", "send_message", "wait"}),
    "MANAGE_INVENTORY": frozenset({"sell", "equip", "produce", "wait"}),
    "ADDRESS_IMBALANCE": frozenset({"buy", "sell", "send_message", "wait"}),
}


@dataclass
class ModelRates:
    """The three de-risk rates for one run."""

    model: str
    condition: str
    activations: int
    llm_calls: int
    invalid_outputs: int
    wasted_activations: int
    director_calls: int
    director_errors: int
    tasks_assigned: int = 0
    tasks_aligned: int = 0
    messages_sent: int = 0
    result_status: dict[str, int] = field(default_factory=dict)

    @property
    def invalid_output_rate(self) -> float:
        return self.invalid_outputs / self.llm_calls if self.llm_calls else 0.0

    @property
    def wasted_activation_rate(self) -> float:
        return self.wasted_activations / self.activations if self.activations else 0.0

    @property
    def director_error_rate(self) -> float:
        return self.director_errors / self.director_calls if self.director_calls else 0.0

    @property
    def task_alignment_rate(self) -> float:
        """Share of TASKs the agent answered with a plausibly responsive action."""
        return self.tasks_aligned / self.tasks_assigned if self.tasks_assigned else 0.0

    @property
    def messages_per_activation(self) -> float:
        return self.messages_sent / self.activations if self.activations else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "condition": self.condition,
            "activations": self.activations,
            "llm_calls": self.llm_calls,
            "invalid_outputs": self.invalid_outputs,
            "wasted_activations": self.wasted_activations,
            "director_calls": self.director_calls,
            "director_errors": self.director_errors,
            "tasks_assigned": self.tasks_assigned,
            "tasks_aligned": self.tasks_aligned,
            "task_alignment_rate": self.task_alignment_rate,
            "messages_sent": self.messages_sent,
            "messages_per_activation": self.messages_per_activation,
            "result_status": dict(self.result_status),
            **{name: getattr(self, name) for name in DERISK_METRICS},
        }


def rates_from_log(log_path: Path | str, model: str) -> ModelRates:
    """Compute the de-risk rates from one completed run's log."""
    counts: collections.Counter[EventType] = collections.Counter()
    statuses: collections.Counter[str] = collections.Counter()
    condition = ""
    pending_task: dict[str, str] = {}
    tasks_assigned = tasks_aligned = 0

    for event in read_events(log_path):
        counts[event.type] += 1
        if event.type is EventType.RUN_METADATA:
            condition = str(event.payload.get("condition", ""))
        elif event.type is EventType.DIRECTOR_TASK_ASSIGNED:
            agent = event.payload.get("agent_id") or event.agent_id or ""
            pending_task[agent] = str(event.payload.get("type", ""))
            tasks_assigned += 1
        elif event.type is EventType.ACTION_EXECUTED and event.agent_id in pending_task:
            task_type = pending_task.pop(event.agent_id)
            expected = TASK_ALIGNED_ACTIONS.get(task_type)
            if expected and str(event.payload.get("action", "")) in expected:
                tasks_aligned += 1
        elif event.type is EventType.RESULT_RETURNED:
            statuses[str(event.payload.get("status", ""))] += 1

    return ModelRates(
        model=model,
        condition=condition,
        activations=counts[EventType.ACTIVATION_SERVED] or counts[EventType.AGENT_OUTPUT],
        llm_calls=counts[EventType.LLM_CALL],
        invalid_outputs=counts[EventType.AGENT_OUTPUT_INVALID],
        wasted_activations=counts[EventType.ACTIVATION_WASTED],
        director_calls=counts[EventType.DIRECTOR_CALL],
        director_errors=counts[EventType.DIRECTOR_ERROR],
        tasks_assigned=tasks_assigned,
        tasks_aligned=tasks_aligned,
        messages_sent=counts[EventType.MESSAGE_SENT],
        result_status=dict(statuses),
    )


@dataclass
class DeriskComparison:
    """Both models' rates, for the report."""

    seed: int
    horizon: int
    rates: list[ModelRates] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "horizon": self.horizon,
            "rates": [r.to_payload() for r in self.rates],
        }

    def by_model(self) -> dict[str, list[ModelRates]]:
        grouped: dict[str, list[ModelRates]] = {}
        for rate in self.rates:
            grouped.setdefault(rate.model, []).append(rate)
        return grouped


def derisk_config(base: SimulationConfig, *, model: str, horizon: int) -> SimulationConfig:
    """A short real-provider configuration for one model.

    Phases are thirds of the horizon so base, shock and recovery all occur; the
    comparison is about whether a model can work the protocol, which every phase
    exercises equally.
    """
    third = horizon // 3
    payload = base.model_dump(mode="python")
    # The snapshot is cleared because it is a static config value: overriding
    # `model` without it would record the wrong snapshot in the metadata. The
    # authoritative record is the model each llm_call event reports.
    payload["llm"] = {**payload["llm"], "model": model, "model_snapshot": None, "mock": False}
    payload["phases"] = {
        "base_ticks": third,
        "shock_ticks": third,
        "recovery_ticks": horizon - 2 * third,
    }
    payload["time"] = {**payload["time"], "horizon_ticks": horizon}
    payload["metrics"] = {
        **payload["metrics"],
        "volatility_window_w": min(payload["metrics"]["volatility_window_w"], third),
        "recovery_hold_ticks_m": min(payload["metrics"]["recovery_hold_ticks_m"], third),
    }
    return SimulationConfig.model_validate(payload)


def collect_comparison(
    directory: Path | str, models: tuple[str, ...], *, seed: int, horizon: int
) -> DeriskComparison:
    """Read an already-executed comparison off disk.

    Separate from execution so the comparison can be re-read and re-reported
    without spending anything a second time.
    """
    root = Path(directory)
    comparison = DeriskComparison(seed=seed, horizon=horizon)
    for model in models:
        for condition in (Condition.A, Condition.B, Condition.C):
            path = root / model / f"{condition.value}.jsonl"
            if path.exists():
                comparison.rates.append(rates_from_log(path, model))
    return comparison
