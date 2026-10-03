"""Replay timelines built from a recorded run log (IMPLEMENTATION.md phase 7).

The replay is a presentation of a finished run, not a re-execution of one.
Nothing here simulates anything: it walks a log once, folds each event into a
per-tick frame, and hands the result to a renderer. That is what keeps the
observation tool outside the experiment -- there is no code path from this module
to the engine, and the import-graph test enforces it.

Frames are cumulative-by-construction: every tick carries the state a viewer
should see at that tick, so a scrub to tick 200 needs no replay of ticks 0-199
and seeking never re-folds from the start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from rpg_economy.logging_ import EventType

#: Events shown in the activity feed, in the order a viewer cares about them.
FEED_EVENTS = (
    EventType.SHOCK_APPLIED,
    EventType.SHOCK_REVERTED,
    EventType.BUDGET_EXHAUSTED,
    EventType.TRADE_EXECUTED,
    EventType.EXPEDITION_RESOLVED,
    EventType.PRODUCTION_COMPLETED,
    EventType.EQUIPMENT_BROKEN,
    EventType.DIRECTOR_ERROR,
    EventType.ORDER_EXPIRED,
)

#: How many feed lines one tick may contribute. A tick that placed forty orders
#: should not be able to push everything else off the screen.
FEED_PER_TICK = 8

#: Feed events never dropped by the per-tick cap: the ones a viewer must not miss.
FEED_ALWAYS = frozenset(
    {
        EventType.SHOCK_APPLIED,
        EventType.SHOCK_REVERTED,
        EventType.BUDGET_EXHAUSTED,
        EventType.DIRECTOR_ERROR,
    }
)


@dataclass
class AgentFrame:
    """One agent's visible state at one tick.

    ``observed`` distinguishes "this agent holds nothing" from "this agent has
    not been activated yet, so the log has not said what it holds". Before its
    first activation an agent's gold is unknown, not zero; rendering it as 0
    would show every not-yet-activated agent as bankrupt. Same rule as design
    v1.0 s11.1/s11.3 (missing is never zero), applied to the replay.
    """

    gold: float = 0.0
    items: int = 0
    last_action: str = ""
    last_target: str | None = None
    #: Counted from AGENT_OUTPUT, so it is activations, not executed actions.
    #: Wasted is not subtracted from executed actions: an activation can execute
    #: a WAIT and still be wasted (design v1.0 s11.2), and a rejected action
    #: executes nothing, so the two counts are not paired.
    activations: int = 0
    wasted: int = 0
    away: bool = False
    observed: bool = False
    #: Taxes the agent has paid so far (sales and wealth). Known from the log
    #: for every agent, activated or not.
    taxes: float = 0.0
    #: Wealth tax the agent still owes (arrears), the ``owed`` of its latest
    #: wealth TAX_PAID event: an assessment or an arrears payment.
    owed: float = 0.0

    @property
    def effective(self) -> int:
        return max(0, self.activations - self.wasted)

    def to_payload(self) -> list[Any]:
        # A list, not an object: this is the single largest thing in the file,
        # and repeating the key names per agent per tick would triple its size.
        return [
            round(self.gold, 2),
            self.items,
            self.last_action,
            self.last_target or "",
            self.effective,
            self.activations,
            1 if self.away else 0,
            1 if self.observed else 0,
            round(self.taxes, 2),
            round(self.owed, 2),
        ]


AGENT_FIELDS = (
    "gold",
    "items",
    "action",
    "target",
    "effective",
    "activations",
    "away",
    "observed",
    "taxes",
    "owed",
)


@dataclass
class TickFrame:
    """Everything the UI shows for one tick."""

    tick: int
    cycle: int
    prices: dict[str, float] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    director: list[dict[str, Any]] = field(default_factory=list)
    feed: list[dict[str, Any]] = field(default_factory=list)
    agents: dict[str, list[Any]] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    tokens_total: int = 0
    director_calls: int = 0
    director_errors: int = 0
    cap_remaining: int | None = None
    trades_total: int = 0
    exhausted: bool = False
    #: The activations of this tick, in the order they happened.
    steps: list[dict[str, Any]] = field(default_factory=list)
    #: How the cycle was coordinated: the scheduler's queue in B/C, the
    #: Director's calls in A.
    sched: dict[str, Any] = field(default_factory=dict)
    #: The bulletin published this cycle (Condition C only).
    bulletin: list[list[Any]] | None = None
    #: Expeditions that came back this tick, with what they yielded.
    returns: list[dict[str, Any]] = field(default_factory=list)
    #: End-of-tick stock per good in all agents' inventories, reconstructed
    #: backwards from the final holdings. Observer information: no agent sees it.
    world_stock: dict[str, int] = field(default_factory=dict)
    #: Gold held by each role at the end of the tick, reconstructed from the log.
    role_gold: dict[str, float] = field(default_factory=dict)
    #: Gold held by each agent at the end of the tick, reconstructed from the log,
    #: in the track's ``agent_ids`` order. A list rather than a mapping: it is
    #: repeated every tick, and the ids would dominate its size.
    agent_gold: list[float | None] = field(default_factory=list)
    #: Units each agent holds at the end of the tick (equipped items excluded),
    #: reconstructed the same way and in the same order as ``agent_gold``.
    agent_items: list[int | None] = field(default_factory=list)
    #: Units per good resting in sell orders at the end of the tick.
    on_sale: dict[str, int] = field(default_factory=dict)
    #: Best bid and best ask per good at the end of the tick ([bid, ask], either
    #: None when that side of the book is empty).
    quotes: dict[str, list[float | None]] = field(default_factory=dict)
    #: Items in use per product.
    equipped: dict[str, int] = field(default_factory=dict)
    #: Resources waiting in each zone at the end of the tick, when logged.
    zone_stock: dict[str, dict[str, int]] | None = None
    #: Gold collected so far, per charge: sales tax, wealth tax and storage.
    charges: dict[str, float] = field(default_factory=dict)
    #: The wealth tax collected this tick, one row per payment:
    #: [agent, taxable gold or None, paid, 1 if it settles arrears else 0,
    #: gold still owed or None].
    wealth_tax: list[list[Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "t": self.tick,
            "c": self.cycle,
            "p": self.prices,
            "a": self.agents,
            "tok": self.tokens,
            "tokTotal": self.tokens_total,
            "tradesTotal": self.trades_total,
        }
        # Omitted rather than sent empty: over a full horizon the empty lists
        # would be most of the payload.
        if self.trades:
            payload["tr"] = self.trades
        if self.messages:
            payload["m"] = self.messages
        if self.director:
            payload["d"] = self.director
        if self.feed:
            payload["f"] = self.feed
        if self.director_calls:
            payload["dc"] = self.director_calls
        if self.director_errors:
            payload["de"] = self.director_errors
        if self.cap_remaining is not None:
            payload["cap"] = self.cap_remaining
        if self.exhausted:
            payload["x"] = 1
        if self.steps:
            payload["st"] = self.steps
        if self.sched:
            payload["sc"] = self.sched
        if self.bulletin is not None:
            payload["bu"] = self.bulletin
        if self.returns:
            payload["ex"] = self.returns
        payload["ws"] = self.world_stock
        payload["rg"] = self.role_gold
        if self.agent_gold:
            payload["ag"] = self.agent_gold
        if self.agent_items:
            payload["ai"] = self.agent_items
        payload["os"] = self.on_sale
        if self.quotes:
            payload["q"] = self.quotes
        if self.equipped:
            payload["eqc"] = self.equipped
        if self.zone_stock is not None:
            payload["zs"] = self.zone_stock
        if self.charges:
            payload["tx"] = self.charges
        if self.wealth_tax:
            payload["wt"] = self.wealth_tax
        return payload


@dataclass
class ReplayTrack:
    """One condition's full timeline."""

    run_id: str
    condition: str
    scenario: str
    experiment: str
    seed: int
    model: str
    #: Taken from the llm_call events, not from run metadata: metadata records
    #: what was configured, the events record what answered.
    model_observed: str
    prompt_version: str
    config_hash: str
    horizon: int
    shock_tick: int | None
    reversion_tick: int | None
    exhaustion_tick: int | None
    budget_total: int | None
    agent_ids: list[str]
    agent_roles: dict[str, str]
    goods: list[str]
    frames: list[TickFrame]
    #: Specialization (subclass for adventurers) and fixed traits per agent.
    agent_specs: dict[str, str] = field(default_factory=dict)
    agent_traits: dict[str, dict[str, str]] = field(default_factory=dict)
    #: What the run's shock did, as logged when it was applied.
    shock: dict[str, Any] | None = None
    #: Director calls allowed per macro-cycle (K + 1), Condition A only.
    director_cap: int | None = None
    #: Ticks at which somebody paid the wealth tax. The collection schedule,
    #: which also covers collections nobody paid, comes from the world rules.
    #: Arrears settled between collections are not collections and are left out.
    wealth_tax_ticks: list[int] = field(default_factory=list)
    #: Prompts and replies of every model call, referenced by index from the
    #: steps and Director calls.
    prompts: PromptTexts = field(default_factory=lambda: PromptTexts())

    @property
    def mock(self) -> bool:
        """Whether any model that answered in this run was the offline mock."""
        return any(name.strip().startswith("mock:") for name in self.model_observed.split(","))

    def to_payload(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "condition": self.condition,
            "scenario": self.scenario,
            "experiment": self.experiment,
            "seed": self.seed,
            "model": self.model,
            "modelObserved": self.model_observed,
            "mock": self.mock,
            "promptVersion": self.prompt_version,
            "configHash": self.config_hash,
            "horizon": self.horizon,
            "shockTick": self.shock_tick,
            "reversionTick": self.reversion_tick,
            "exhaustionTick": self.exhaustion_tick,
            "budgetTotal": self.budget_total,
            "directorCap": self.director_cap,
            "agentIds": self.agent_ids,
            "agentRoles": self.agent_roles,
            "agentSpecs": self.agent_specs,
            "agentTraits": self.agent_traits,
            "shock": self.shock,
            "agentFields": list(AGENT_FIELDS),
            "wealthTaxTicks": self.wealth_tax_ticks,
            "promptTexts": self.prompts.packed(),
            "goods": self.goods,
            "frames": [frame.to_payload() for frame in self.frames],
        }


def _decimal(value: Any) -> float:
    return float(Decimal(str(value)))


class PromptTexts:
    """The prompts and replies of one run, each text stored once.

    Steps and Director calls refer to texts by index. The system prompt, the
    same for every call of a role, is logged once per run and named by digest;
    it is stored once here too. The page receives the list gzip-compressed and
    decompresses it only when a prompt is opened.
    """

    def __init__(self) -> None:
        self.texts: list[str] = []
        self._ids: dict[str, int] = {}
        #: System prompt digest -> text index, from prompt_system events.
        self.systems: dict[str, int] = {}

    def add(self, text: Any) -> int | None:
        if not isinstance(text, str):
            return None
        found = self._ids.get(text)
        if found is None:
            found = self._ids[text] = len(self.texts)
            self.texts.append(text)
        return found

    def system(self, sha: Any) -> int | None:
        return self.systems.get(sha) if isinstance(sha, str) else None

    def packed(self) -> str | None:
        """The texts as base64 of gzip of JSON, or None when there are none."""
        if not self.texts:
            return None
        import base64
        import gzip
        import json

        raw = json.dumps(self.texts, ensure_ascii=False, separators=(",", ":"))
        # mtime=0 keeps the page byte-reproducible.
        return base64.b64encode(gzip.compress(raw.encode("utf-8"), mtime=0)).decode("ascii")
