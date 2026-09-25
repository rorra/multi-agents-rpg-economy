"""Small reads of a run log shared by the report pipeline and the budget derivation."""

from __future__ import annotations

from pathlib import Path

from rpg_economy.config.models import SimulationConfig
from rpg_economy.logging_ import last_event


def run_end_ticks(path: Path | str) -> int | None:
    """The ticks a finished run reports in its closing ``run_end`` event."""
    ticks = ((last_event(path) or {}).get("payload") or {}).get("ticks")
    return None if ticks is None else int(ticks)


def with_run_phase(config: SimulationConfig, phase: str | None) -> SimulationConfig:
    """``config`` with the run phase a log recorded, where the config has one.

    The phase says which campaign stage made a run, not how it behaves; it is
    carried over so an identity check compares the run with itself.
    """
    if phase is None or "phase" not in type(config.run).model_fields:
        return config
    if getattr(config.run, "phase", None) == phase:
        return config
    payload = config.model_dump(mode="python")
    payload["run"] = {**payload["run"], "phase": phase}
    return SimulationConfig.model_validate(payload)
