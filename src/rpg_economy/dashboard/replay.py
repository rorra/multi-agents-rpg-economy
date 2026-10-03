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

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from rpg_economy.logging_ import Event, EventType

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


def _on_sale(book: dict[Any, tuple[str, str, float, int]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for good, side, _, remaining in book.values():
        if side == "sell":
            totals[good] += remaining
    return {good: units for good, units in sorted(totals.items()) if units}


def _quotes(book: dict[Any, tuple[str, str, float, int]]) -> dict[str, list[float | None]]:
    """Best bid and best ask per good resting on the book at the end of the tick."""
    bids: dict[str, float] = {}
    asks: dict[str, float] = {}
    for good, side, limit, remaining in book.values():
        if remaining <= 0:
            continue
        if side == "buy":
            bids[good] = max(bids.get(good, limit), limit)
        else:
            asks[good] = min(asks.get(good, limit), limit)
    return {good: [bids.get(good), asks.get(good)] for good in sorted(bids.keys() | asks.keys())}


def _reconstruct_world_stock(
    frames: list[TickFrame], deltas: list[Counter[str]], final: dict[str, Any]
) -> None:
    """End-of-tick stock per good, walked backwards from the final holdings.

    Trades only move goods between agents, so the world total changes only
    through expedition yields, production, consumption and equipping, all of
    which are logged. A log without final holdings (a run cut short) has no
    anchor, and the stock is left empty rather than guessed.
    """
    if not final:
        return
    running: Counter[str] = Counter()
    for holding in final.values():
        for good, units in (holding.get("inventory") or {}).items():
            running[good] += int(units)
    for frame, delta in zip(reversed(frames), reversed(deltas), strict=True):
        frame.world_stock = dict(sorted(running.items()))
        running.subtract(delta)


def _reconstruct_role_gold(
    frames: list[TickFrame],
    deltas: list[dict[str, Decimal]],
    final: dict[str, Any],
    agent_ids: list[str],
) -> None:
    """End-of-tick gold per role and per agent, walked backwards from the final holdings.

    The same for every condition and every tick, unlike the gold an observation
    shows, which only covers agents activated so far and goes stale between
    activations. Left empty without final holdings rather than guessed.
    """
    if not final:
        return
    gold = {agent: Decimal(str(h.get("gold", "0"))) for agent, h in final.items()}
    role = {agent: str(h.get("role", "")) for agent, h in final.items()}
    cent = Decimal("0.01")
    for frame, delta in zip(reversed(frames), reversed(deltas), strict=True):
        totals: dict[str, Decimal] = defaultdict(Decimal)
        for agent, amount in gold.items():
            totals[role[agent]] += amount
        frame.role_gold = {r: float(v.quantize(cent)) for r, v in sorted(totals.items())}
        frame.agent_gold = [
            float(gold[agent].quantize(cent)) if agent in gold else None for agent in agent_ids
        ]
        for agent, change in delta.items():
            if agent in gold:
                gold[agent] -= change


def _reconstruct_agent_items(
    frames: list[TickFrame],
    deltas: list[Counter[str]],
    final: dict[str, Any],
    agent_ids: list[str],
) -> None:
    """End-of-tick units held per agent, walked backwards from the final holdings.

    Every change to an inventory is logged with its agent: trades, expedition
    yields, production, consumption and equipping. Like the world stock, items
    in use are not counted. Left empty without final holdings rather than guessed.
    """
    if not final:
        return
    running = {
        agent: sum(int(u) for u in (h.get("inventory") or {}).values())
        for agent, h in final.items()
    }
    for frame, delta in zip(reversed(frames), reversed(deltas), strict=True):
        frame.agent_items = [running.get(agent) for agent in agent_ids]
        for agent, change in delta.items():
            if agent in running:
                running[agent] -= change


def _plain(payload: dict[str, Any]) -> dict[str, Any]:
    """A payload with Decimal-like values as strings, safe to embed as JSON."""
    return {k: (str(v) if isinstance(v, Decimal) else v) for k, v in payload.items()}


def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """What the agent was shown, without the parts that never change.

    The directory is static and the traits are fixed per agent, so both are
    carried once per track instead of once per activation. The bulletin is
    carried once per tick; here only its presence is recorded.
    """
    market = []
    for row in observation.get("market") or []:
        recent = row.get("recent_prices") or []
        market.append(
            [
                row.get("good"),
                row.get("best_bid"),
                row.get("best_ask"),
                recent[-1] if recent else None,
                len(recent),
            ]
        )
    orders = [
        [
            o.get("side"),
            o.get("good"),
            o.get("remaining"),
            o.get("limit_price"),
            o.get("expires_at_tick"),
            o.get("order_id"),
        ]
        for o in observation.get("open_orders") or []
    ]
    inbox = []
    for message in observation.get("messages") or []:
        body = message.get("payload") or {}
        inbox.append(
            [
                message.get("type"),
                message.get("sender_id"),
                body.get("good"),
                body.get("quantity"),
                body.get("price"),
                message.get("sent_tick"),
            ]
        )
    return {
        "g": observation.get("gold"),
        "av": observation.get("available_gold"),
        "inv": observation.get("inventory") or {},
        "eq": observation.get("equipment") or {},
        "gear": observation.get("expedition_gear") or {},
        "away": 1 if observation.get("is_away") else 0,
        "cap": observation.get("production_capacity", 0),
        "ord": orders,
        "mk": market,
        "in": inbox,
        "bul": 1 if observation.get("bulletin") else 0,
    }


def _compact_bulletin(summary: dict[str, Any]) -> list[list[Any]]:
    """The bulletin's per-good rows: good, mean price, trades, supply, unmet demand."""
    return [
        [
            row.get("good"),
            row.get("mean_price"),
            row.get("trades"),
            row.get("available_supply"),
            row.get("unmet_demand"),
        ]
        for row in summary.get("goods") or []
    ]


def _waste_reason(engine: list[list[Any]], payload: dict[str, Any]) -> Any:
    """Why an activation was wasted, as a code when the log allows one.

    The event itself carries the engine's message or the agent's own reason;
    the code comes from the rejection logged just before it, or from the WAIT
    the activation ended in.
    """
    if payload.get("code"):
        return payload["code"]
    rejected = [entry[1] for entry in engine if entry[0] == "rejected" and entry[1]]
    if rejected:
        return rejected[-1]
    if payload.get("action") == "wait":
        # The fallback WAIT: invalid output after the one retry, or the budget
        # running out mid-activation (its reason is the runner's fixed text).
        reason = str(payload.get("reason") or "")
        if reason == "budget exhausted":
            return "budget_exhausted"
        # Invalid output, whether or not the budget left room for the retry.
        if reason.startswith("invalid output") or int(payload.get("invalid_outputs") or 0) >= 2:
            return "invalid_output"
        return "wait"
    return payload.get("reason")


def _attach_trade(step: dict[str, Any], payload: dict[str, Any]) -> None:
    """A trade the acting agent was party to, filled during its activation."""
    agent = step["a"]
    if agent not in (payload.get("buyer_id"), payload.get("seller_id")):
        return
    buying = payload.get("buyer_id") == agent
    step["e"].append(
        [
            "trade",
            "buy" if buying else "sell",
            payload.get("good"),
            payload.get("quantity"),
            payload.get("price"),
            payload.get("seller_id") if buying else payload.get("buyer_id"),
        ]
    )


def _reconcile(
    final: TickFrame | None, ledger: dict[str, int], authoritative: int, path: Path
) -> None:
    """Refuse to build a replay whose bars would contradict the run's ledger.

    A displayed total that disagrees with the log leaves a viewer no way to tell
    which number is wrong, so the contradiction is made unrenderable.
    """
    if final is None or not ledger:
        return  # a log without a tokens_consumed event has nothing to check against

    displayed = {bucket: final.tokens.get(bucket, 0) for bucket in ledger}
    mismatched = {b: (displayed[b], ledger[b]) for b in ledger if displayed[b] != ledger[b]}
    if mismatched:
        detail = ", ".join(
            f"{b}: shows {d} but ledger says {a}" for b, (d, a) in mismatched.items()
        )
        raise ValueError(f"{path.name}: replay bucket totals disagree with the ledger -- {detail}")

    if authoritative and final.tokens_total != authoritative:
        raise ValueError(
            f"{path.name}: replay total {final.tokens_total} disagrees with the "
            f"ledger's authoritative total {authoritative}"
        )


def _feed_detail(event: Event) -> dict[str, Any]:
    """The one or two fields that make a feed line worth reading."""
    payload = event.payload
    match event.type:
        case EventType.TRADE_EXECUTED:
            return {"g": payload["good"], "v": str(payload["price"]), "q": payload["quantity"]}
        case EventType.EXPEDITION_RESOLVED:
            return {"g": payload.get("outcome", ""), "q": payload.get("total_units", 0)}
        case EventType.PRODUCTION_COMPLETED:
            return {"g": payload.get("product", ""), "q": payload.get("quantity", 0)}
        case EventType.ORDER_EXPIRED:
            return {"g": payload.get("good", ""), "v": str(payload.get("limit_price", ""))}
        case EventType.SHOCK_APPLIED | EventType.SHOCK_REVERTED:
            # The good a shock targets when it has one; otherwise the scenario,
            # which the page labels. A scenario id is not a good.
            good = payload.get("target_resource") or payload.get("target_product")
            return {"g": str(good)} if good else {"s": str(payload.get("scenario", ""))}
        case _:
            return {}


@dataclass
class ReplayBundle:
    """One or more tracks replayed together, synchronized by tick."""

    tracks: list[ReplayTrack]
    title: str = ""
    #: Shown as a banner. Used to mark instrument-validation runs so they are
    #: never mistaken for experimental results: a replay of a pilot or mock run
    #: looks exactly like a replay of a final one.
    caveat: str = ""
    #: The world's crafting and gathering rules, for the reference guide. Set
    #: only from a configuration proven to be the one the runs used.
    rules: dict[str, Any] | None = None

    @property
    def horizon(self) -> int:
        return max((track.horizon for track in self.tracks), default=0)

    @property
    def mock(self) -> bool:
        """Whether any track was answered by the offline mock rather than a model.

        Such a replay shows the instrument working, not how a model behaves, and
        the page says so whether or not a caveat was given.
        """
        return any(track.mock for track in self.tracks)

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "caveat": self.caveat,
            "mock": self.mock,
            "horizon": self.horizon,
            "tracks": [track.to_payload() for track in self.tracks],
            "rules": self.rules,
        }
