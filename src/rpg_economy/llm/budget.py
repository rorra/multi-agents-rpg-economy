"""The Experiment 2 token budget (design v1.0 s8.4).

Experiment 2 gives every condition the same maximum number of LLM tokens per
run. **The budget consumes the provider-reported authoritative total, never the
sum of bucket estimates**: buckets are analytical attribution, the
budget is contractual. Enforcing against a locally derived number would make the
exhaustion tick -- itself a reported outcome of Experiment 2 (s11.2) -- an
artifact of how prompts happen to apportion.

On exhaustion, all LLM calls cease. The engine does not: it runs mechanical ticks
to the planned horizon, orders execute or expire, durability decays, and
scheduled shocks still fire (s8.4). Exhausting early is itself a result.

A call is authorised *before* it is made and charged *after*, because its cost is
not knowable in advance. A call that crosses the cap therefore completes, and the
cap binds from the next one on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenBudget:
    """A run's token allowance. ``cap`` of None means unlimited (Experiment 1)."""

    cap: int | None = None
    consumed: int = 0
    exhausted_tick: int | None = None
    exhausted_cycle: int | None = None

    @property
    def enforced(self) -> bool:
        return self.cap is not None

    @property
    def exhausted(self) -> bool:
        return self.cap is not None and self.consumed >= self.cap

    @property
    def remaining(self) -> int | None:
        return None if self.cap is None else max(0, self.cap - self.consumed)

    @property
    def fraction_consumed(self) -> float | None:
        """Share of the budget used. Reported when a run does not exhaust (s11.2).

        ``None`` without a cap: there is no budget to have used a share of.
        """
        if not self.cap:
            return None
        return min(1.0, self.consumed / self.cap)

    def may_call(self) -> bool:
        """Whether another LLM call is authorised."""
        return not self.exhausted

    def charge(self, provider_total: int, *, tick: int, cycle: int) -> bool:
        """Charge one call's authoritative cost. Returns True if this exhausted it.

        ``provider_total`` is what the provider reported for the call, input plus
        output. Bucket attribution plays no part here.
        """
        if provider_total < 0:
            raise ValueError(f"a call cannot cost {provider_total} tokens")
        was_exhausted = self.exhausted
        self.consumed += provider_total
        if self.exhausted and not was_exhausted:
            self.exhausted_tick = tick
            self.exhausted_cycle = cycle
            return True
        return False

    def snapshot(self) -> dict:
        """Language-neutral summary for the run log."""
        return {
            "cap": self.cap,
            "consumed": self.consumed,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
            "exhausted_tick": self.exhausted_tick,
            "exhausted_cycle": self.exhausted_cycle,
            "fraction_consumed": (
                None if self.fraction_consumed is None else round(self.fraction_consumed, 6)
            ),
        }


def common_active_horizon(exhaustion_ticks: list[int | None], planned_horizon: int) -> int:
    """The trio's common active horizon (design v1.0 s8.4).

    Ends at the **earliest** exhaustion tick observed across A, B and C; if none
    exhausted, at the planned horizon. Metrics recomputed over this interval
    compare the three architectures while all of them still had decision capacity
    -- the secondary analysis s8.4 prescribes, alongside primary metrics over the
    full horizon.
    """
    observed = [tick for tick in exhaustion_ticks if tick is not None]
    return min(observed) if observed else planned_horizon
