"""Deterministic random streams derived from a single per-run seed.

Design v1.0 section 15 requires that all engine and scheduler randomness derive
from the run's seeds, and IMPLEMENTATION.md section 4.1 makes determinism binding: the
same seed and the same decisions must produce an identical run.

Randomness is partitioned into named independent streams rather than drawn
from one shared generator. This matters for the seed-paired trio design: A, B and
C draw from the world differently -- a Director-led run activates agents in a
different order than a trigger-led one -- and with a single shared generator the
n-th expedition of condition A would consume a different draw than the n-th of
condition B. Named streams make each subsystem's sequence depend only on the run
seed and on that subsystem's own history, so a trio genuinely shares its world
randomness (IMPLEMENTATION.md section 6, paired-trio harness).

It also means adding a new stream later cannot perturb existing ones.
"""

from __future__ import annotations

import hashlib
import random
from typing import Final

#: Stream names used by the engine. Declared so that typos surface as an error
#: rather than as a silently divergent stream.
STREAM_EXPEDITION: Final = "expedition"
STREAM_SPAWN: Final = "spawn"
STREAM_SHOCK: Final = "shock"
STREAM_SCHEDULER: Final = "scheduler"
STREAM_TEST: Final = "test"
#: Per-tick order in which expeditions due together draw from the zone pools.
STREAM_EXPEDITION_ORDER: Final = "expedition_order"

KNOWN_STREAMS: Final[frozenset[str]] = frozenset(
    {
        STREAM_EXPEDITION,
        STREAM_EXPEDITION_ORDER,
        STREAM_SPAWN,
        STREAM_SHOCK,
        STREAM_SCHEDULER,
        STREAM_TEST,
    }
)

#: Prefix for per-agent streams, e.g. ``agent:smithing_1``.
AGENT_STREAM_PREFIX: Final = "agent:"


def agent_stream_name(agent_id: str) -> str:
    """The stream name reserved for one agent's own decisions."""
    return f"{AGENT_STREAM_PREFIX}{agent_id}"


def derive_seed(run_seed: int, stream: str) -> int:
    """Derive a stable sub-seed for ``stream`` from ``run_seed``.

    Uses SHA-256 rather than Python's ``hash``, whose string hashing is salted
    per process and would silently break reproducibility across runs.
    """
    digest = hashlib.sha256(f"{run_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class RunRandom:
    """The run's single source of randomness, partitioned into named streams."""

    def __init__(self, run_seed: int) -> None:
        self._run_seed = run_seed
        self._streams: dict[str, random.Random] = {}

    @property
    def run_seed(self) -> int:
        return self._run_seed

    def stream(self, name: str) -> random.Random:
        """Return the generator for ``name``, creating it on first use."""
        if name not in KNOWN_STREAMS and not name.startswith(AGENT_STREAM_PREFIX):
            raise ValueError(
                f"unknown random stream {name!r}; declare it in rng.KNOWN_STREAMS so "
                f"that reproducibility stays auditable"
            )
        if name not in self._streams:
            self._streams[name] = random.Random(derive_seed(self._run_seed, name))
        return self._streams[name]

    def agent(self, agent_id: str) -> random.Random:
        """Return one agent's private stream.

        Each agent draws from its own generator so that its behaviour depends on
        the run seed and on its own history alone -- not on how many other agents
        happened to act before it. Without this, activating agents in a different
        order (which is precisely what distinguishes conditions A, B and C) would
        hand the same agent different draws, and a seed-trio would no longer be
        comparing the same world.
        """
        return self.stream(agent_stream_name(agent_id))

    def expedition(self, agent_id: str, zone_id: str, started_tick: int) -> random.Random:
        """A fresh generator for one expedition, keyed by who, where and when.

        Not drawn from the agent's stream: that stream also serves the agent's own
        decisions, so how many draws came before would decide the outcome. Keyed
        this way, the same adventurer leaving for the same zone on the same tick
        meets the same luck in A, B and C, whatever else happened in between.
        """
        return random.Random(
            derive_seed(self._run_seed, f"{STREAM_EXPEDITION}:{agent_id}:{zone_id}:{started_tick}")
        )

    def permutation(self, items: list[str], stream: str) -> list[str]:
        """Return a deterministic permutation of ``items``.

        Used for the neutral scheduler's fixed tie-break order (design v1.0
        s8.3), which must be derived from the run seed.
        """
        shuffled = list(items)
        random.Random(derive_seed(self._run_seed, stream)).shuffle(shuffled)
        return shuffled
