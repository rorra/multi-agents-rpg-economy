"""Activation requests -- the currency of Conditions B and C (design v1.0 s8)."""

from __future__ import annotations

from dataclasses import dataclass, field

from rpg_economy.domain import Trigger


@dataclass
class ActivationRequest:
    """One agent's standing request to act.

    An agent has at most one request at a time; further triggers accumulate onto
    it. ``first_requested_tick`` is kept for the latency metric (design s11.2),
    which measures ticks from the trigger to a traceable action.
    """

    agent_id: str
    first_requested_tick: int
    triggers: set[Trigger] = field(default_factory=set)

    def add(self, trigger: Trigger) -> None:
        self.triggers.add(trigger)

    def trigger_values(self) -> list[str]:
        """Fired triggers in a fixed order, for logging."""
        return sorted(t.value for t in self.triggers)
