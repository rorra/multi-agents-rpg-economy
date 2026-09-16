"""The neutral scheduler (design v1.0 s8.3). Conditions B and C only.

Serves up to K activation requests per macro-cycle. Priority goes to whichever
agent has accumulated the fewest activations so far in the run; ties break on a
fixed permutation of agent ids derived from the run seed.

**Neutrality is structural, not a promise.** ``serve`` accepts agent *ids* and
nothing else -- no world state, no accounts, no messages, no trigger payloads. It
therefore cannot inspect content even by mistake, which is the property design
s8.3 requires and that a test can assert by signature rather than by hope. The
scheduler is a queue discipline, not a decision maker: it never chooses *what* an
agent does, only whether its turn comes up.

No LLM is reachable from here, by construction.
"""

from __future__ import annotations

from collections.abc import Iterable

from rpg_economy.engine.rng import STREAM_SCHEDULER, RunRandom


class NeutralScheduler:
    """Fewest-activations-first queue discipline with a seeded tie-break."""

    def __init__(self, agent_ids: Iterable[str], random: RunRandom, k: int) -> None:
        if k < 1:
            raise ValueError(f"K must be at least 1, got {k}")
        self._k = k
        self._activations: dict[str, int] = dict.fromkeys(agent_ids, 0)
        #: Fixed permutation derived from the run seed (design v1.0 s8.3). Drawn
        #: once at construction so it cannot drift as the run proceeds.
        self._tie_break = random.permutation(sorted(self._activations), STREAM_SCHEDULER)
        self._rank = {agent_id: i for i, agent_id in enumerate(self._tie_break)}

    @property
    def k(self) -> int:
        return self._k

    @property
    def tie_break_order(self) -> list[str]:
        return list(self._tie_break)

    def activations_of(self, agent_id: str) -> int:
        return self._activations.get(agent_id, 0)

    @property
    def total_activations(self) -> int:
        return sum(self._activations.values())

    def serve(self, requesting: Iterable[str]) -> list[str]:
        """Choose up to K agents to activate, in the order they will act.

        Takes ids only. Nothing about state, messages or triggers reaches this
        method, so the choice cannot depend on them.
        """
        candidates = sorted(set(requesting))
        unknown = [a for a in candidates if a not in self._activations]
        if unknown:
            raise KeyError(f"scheduler does not know agents {unknown}")

        ordered = sorted(candidates, key=lambda a: (self._activations[a], self._rank[a]))
        return ordered[: self._k]

    def record_activation(self, agent_id: str) -> None:
        """Count an activation actually carried out."""
        self._activations[agent_id] = self._activations.get(agent_id, 0) + 1
