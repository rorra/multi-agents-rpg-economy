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

from rpg_economy.config import config_hash
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Condition, ExperimentRegime, Scenario, TokenBucket
from rpg_economy.logging_ import Event, EventType, read_events, read_metadata

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


def build_track(log_path: Path | str) -> ReplayTrack:
    """Fold one run log into a per-tick timeline."""
    path = Path(log_path)
    metadata = read_metadata(path)
    events: list[Event] = list(read_events(path))
    prompts = PromptTexts()

    # The last tick the engine ran. The closing events (final token totals,
    # run_end) are stamped one past it, when the clock has already advanced;
    # they belong to the last tick rather than to a tick in which nothing ran.
    ran = [e.tick for e in events if e.type is EventType.TICK]
    horizon = max(ran) if ran else max((e.tick for e in events), default=0)
    by_tick: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        by_tick[min(event.tick, horizon)].append(event)

    prices: dict[str, float] = {}
    goods: list[str] = []
    observed_models: list[str] = []
    agents: dict[str, AgentFrame] = {}
    roles: dict[str, str] = {}
    tokens: Counter[str] = Counter()
    tokens_total = 0
    trades_total = 0
    director_calls = 0
    director_errors = 0
    cap_remaining: int | None = None
    shock_tick = reversion_tick = exhaustion_tick = None
    ledger_totals: dict[str, int] = {}
    ledger_authoritative = 0
    budget_total: int | None = None
    director_cap: int | None = None
    exhausted = False

    frames: list[TickFrame] = []
    cycle = 0
    specs: dict[str, str] = {}
    traits: dict[str, dict[str, str]] = {}
    #: Why an agent is about to act: the trigger it was served for (B/C) or the
    #: TASK it was assigned (A). Consumed by its next observation.
    pending_why: dict[str, dict[str, Any]] = {}
    last_director_call: dict[str, Any] | None = None
    shock_details: dict[str, Any] | None = None
    #: Net change of each good's world stock per tick, and the final holdings
    #: the stock is reconstructed from.
    deltas: list[Counter[str]] = []
    gold_deltas: list[dict[str, Decimal]] = []
    item_deltas: list[Counter[str]] = []
    final_holdings: dict[str, Any] = {}
    #: Resting orders by id: (good, side, limit price, units remaining).
    book: dict[Any, tuple[str, str, float, int]] = {}
    equipped: Counter[str] = Counter()
    zone_stock: dict[str, dict[str, int]] | None = None
    charges: Counter[str] = Counter()
    wealth_tax_ticks: list[int] = []

    for tick in range(horizon + 1):
        trades: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        director: list[dict[str, Any]] = []
        feed: list[dict[str, Any]] = []
        feed_dropped = 0
        steps: list[dict[str, Any]] = []
        sched: dict[str, Any] = {}
        bulletin: list[list[Any]] | None = None
        returns: list[dict[str, Any]] = []
        step: dict[str, Any] | None = None
        delta: Counter[str] = Counter()
        gold_delta: dict[str, Decimal] = defaultdict(Decimal)
        #: Net change of each agent's unit count this tick.
        item_delta: Counter[str] = Counter()
        #: This tick's trade feed lines by trade id, so the sales tax the seller
        #: paid can be shown on the trade itself.
        trade_lines: dict[str, dict[str, Any]] = {}
        wealth: dict[str, Any] | None = None
        wealth_rows: list[list[Any]] = []

        for event in by_tick.get(tick, ()):
            # Gold moves only through trades, expedition loot, confiscation pay, taxes
            # and storage.
            if event.type is EventType.TRADE_EXECUTED:
                value = Decimal(str(event.payload["value"]))
                gold_delta[event.payload["buyer_id"]] -= value
                gold_delta[event.payload["seller_id"]] += value
                item_delta[event.payload["buyer_id"]] += int(event.payload["quantity"])
                item_delta[event.payload["seller_id"]] -= int(event.payload["quantity"])
            elif event.type is EventType.EXPEDITION_RESOLVED and event.agent_id:
                gold_delta[event.agent_id] += Decimal(str(event.payload.get("gold", "0")))
            elif event.type in (EventType.TAX_PAID, EventType.STORAGE_PAID) and event.agent_id:
                gold_delta[event.agent_id] -= Decimal(str(event.payload["amount"]))
            elif event.type is EventType.GOODS_CONFISCATED and event.agent_id:
                gold_delta[event.agent_id] += Decimal(str(event.payload["gold"]))
                item_delta[event.agent_id] -= int(event.payload["quantity"])
            if event.tick <= horizon:  # closing events keep the last cycle that ran
                cycle = event.cycle
            payload = event.payload
            agent_id = event.agent_id

            if step is not None and agent_id == step["a"]:
                _extend_step(step, event, prompts)
            if step is not None and event.type is EventType.TRADE_EXECUTED:
                _attach_trade(step, payload)

            match event.type:
                case EventType.PROMPT_SYSTEM:
                    index = prompts.add(payload.get("text"))
                    if index is not None:
                        prompts.systems[str(payload.get("sha"))] = index
                case EventType.OBSERVATION_DELIVERED:
                    observation = payload.get("observation") or {}
                    aid = observation.get("agent_id") or agent_id
                    if aid:
                        frame = agents.setdefault(aid, AgentFrame())
                        frame.gold = _decimal(observation.get("gold", 0))
                        frame.items = sum((observation.get("inventory") or {}).values())
                        frame.away = bool(observation.get("is_away"))
                        frame.observed = True
                        roles.setdefault(aid, str(observation.get("role", "")))
                        if observation.get("specialization"):
                            specs.setdefault(aid, str(observation["specialization"]))
                        if observation.get("traits"):
                            traits.setdefault(aid, dict(observation["traits"]))
                        if bulletin is None and observation.get("bulletin"):
                            bulletin = _compact_bulletin(observation["bulletin"])
                        step = {
                            "n": len(steps) + 1,
                            "a": aid,
                            "w": pending_why.pop(aid, None),
                            "o": _compact_observation(observation),
                            "l": [],
                            "x": [],
                            "e": [],
                        }
                        steps.append(step)
                case EventType.ACTIVATION_SERVED if agent_id:
                    pending_why[agent_id] = {
                        "tr": list(payload.get("triggers") or []),
                        "wt": payload.get("waiting_ticks", 0),
                    }
                    sched.setdefault("served", []).append(agent_id)
                case EventType.ACTIVATION_DEFERRED if agent_id:
                    sched.setdefault("deferred", []).append(
                        [agent_id, list(payload.get("triggers") or [])]
                    )
                case EventType.ACTIVATION_REQUESTED:
                    sched["requested"] = sched.get("requested", 0) + 1
                case EventType.TRIGGER_FIRED if not payload.get("pending"):
                    # Repeat firings on a standing request are logged as pending.
                    fired = sched.setdefault("fired", {})
                    name = str(payload.get("trigger", ""))
                    fired[name] = fired.get(name, 0) + 1
                case EventType.DIRECTOR_TASK_ASSIGNED if agent_id:
                    pending_why[agent_id] = {
                        "task": {
                            "id": payload.get("task_id"),
                            "ty": payload.get("type"),
                            "pr": payload.get("priority"),
                            "p": payload.get("params") or {},
                            "ctx": [
                                c.get("result_id", c) if isinstance(c, dict) else c
                                for c in payload.get("context") or []
                            ],
                        },
                        "dc": last_director_call,
                    }
                    last_director_call = None
                case EventType.DIRECTOR_CYCLE_CLOSED:
                    sched["closed"] = {"dc": last_director_call, **_plain(payload)}
                    last_director_call = None
                case EventType.EXPEDITION_RESOLVED if agent_id:
                    for resource, units in (payload.get("yielded") or {}).items():
                        delta[resource] += int(units)
                        item_delta[agent_id] += int(units)
                        if zone_stock is not None:
                            pool = zone_stock.setdefault(str(payload.get("zone_id")), {})
                            pool[resource] = pool.get(resource, 0) - int(units)
                    returns.append(
                        {
                            "a": agent_id,
                            "ok": 1 if payload.get("outcome") == "success" else 0,
                            "z": payload.get("zone_id"),
                            "r": payload.get("risk"),
                            "pr": payload.get("success_probability"),
                            "y": payload.get("yielded") or {},
                            "rec": payload.get("recovering_until_tick") or 0,
                            "gd": str(payload.get("gold") or "0"),
                            "br": [],
                        }
                    )
                case EventType.EQUIPMENT_BROKEN if agent_id:
                    equipped[str(payload.get("product"))] -= 1
                    # Wear from the expedition itself is logged right after the
                    # return; attach it to that return rather than to the feed.
                    if payload.get("cause") == "expedition_wear":
                        for entry in reversed(returns):
                            if entry["a"] == agent_id:
                                entry["br"].append(payload.get("product"))
                                break
                case EventType.RESOURCES_SPAWNED:
                    zone_stock = {
                        zone: {r: int(u) for r, u in pool.items()}
                        for zone, pool in (payload.get("stock") or {}).items()
                    }
                case EventType.PRODUCTION_COMPLETED:
                    for good, units in (payload.get("consumed") or {}).items():
                        delta[good] -= int(units)
                        if agent_id:
                            item_delta[agent_id] -= int(units)
                    delta[str(payload.get("product"))] += int(payload.get("quantity", 0))
                    if agent_id:
                        item_delta[agent_id] += int(payload.get("quantity", 0))
                case EventType.GOODS_CONFISCATED:
                    delta[str(payload.get("good"))] -= int(payload.get("quantity", 0))
                case EventType.CONSUMPTION:
                    delta[str(payload.get("product"))] -= int(payload.get("quantity", 0))
                    if agent_id:
                        item_delta[agent_id] -= int(payload.get("quantity", 0))
                case EventType.EQUIPMENT_EQUIPPED:
                    delta[str(payload.get("product"))] -= 1
                    equipped[str(payload.get("product"))] += 1
                    if agent_id:
                        item_delta[agent_id] -= 1
                case EventType.ORDER_PLACED:
                    book[payload.get("order_id")] = (
                        str(payload.get("good")),
                        str(payload.get("side")),
                        _decimal(payload.get("limit_price", 0)),
                        int(payload.get("remaining", 0)),
                    )
                case EventType.ORDER_PARTIALLY_FILLED if payload.get("order_id") in book:
                    good_id, side, limit, _ = book[payload["order_id"]]
                    book[payload["order_id"]] = (
                        good_id,
                        side,
                        limit,
                        int(payload.get("remaining", 0)),
                    )
                case EventType.ORDER_FILLED | EventType.ORDER_EXPIRED | EventType.ORDER_CANCELLED:
                    book.pop(payload.get("order_id"), None)
                case EventType.RUN_END:
                    final_holdings = payload.get("holdings") or {}
                case EventType.CYCLE_END:
                    step = None
                    sched["end"] = _plain(payload)
                case EventType.AGENT_OUTPUT if agent_id:
                    agents.setdefault(agent_id, AgentFrame()).activations += 1
                case EventType.ACTION_EXECUTED if agent_id:
                    frame = agents.setdefault(agent_id, AgentFrame())
                    frame.last_action = str(payload.get("action", ""))
                    frame.last_target = payload.get("target")
                case EventType.ACTIVATION_WASTED if agent_id:
                    agents.setdefault(agent_id, AgentFrame()).wasted += 1
                case EventType.TRADE_EXECUTED:
                    good = str(payload["good"])
                    price = _decimal(payload["price"])
                    prices[good] = price
                    if good not in goods:
                        goods.append(good)
                    trades_total += 1
                    trades.append(
                        {
                            "g": good,
                            "p": price,
                            "q": payload["quantity"],
                            "b": payload["buyer_id"],
                            "s": payload["seller_id"],
                        }
                    )
                case EventType.MESSAGE_SENT:
                    messages.append(
                        {
                            "ty": str(payload.get("type") or payload.get("message_type", "")),
                            "from": str(agent_id or payload.get("sender_id", "")),
                            "to": str(payload.get("recipient_id", "")),
                            "g": payload.get("good"),
                            "q": payload.get("quantity"),
                            "p": payload.get("price"),
                        }
                    )
                case EventType.DIRECTOR_TASK_ASSIGNED:
                    director.append(
                        {
                            "k": "task",
                            "id": payload.get("task_id"),
                            "ty": payload.get("type"),
                            "to": payload.get("agent_id") or agent_id,
                            "pr": payload.get("priority"),
                        }
                    )
                case EventType.RESULT_RETURNED:
                    director.append(
                        {
                            "k": "result",
                            "id": payload.get("result_id"),
                            "st": payload.get("status"),
                            "to": payload.get("agent_id") or agent_id,
                        }
                    )
                case EventType.DIRECTOR_ERROR:
                    director_errors += 1
                    sched.setdefault("errors", []).append(_plain(payload))
                case EventType.LLM_CALL:
                    name = str(payload.get("model") or "")
                    if name and name not in observed_models:
                        observed_models.append(name)
                    # Input is apportioned across buckets by the ledger; output
                    # belongs wholly to the speaker, which for an llm_call is the
                    # agent.
                    for bucket, amount in (payload.get("input_by_bucket") or {}).items():
                        tokens[bucket] += int(amount)
                        tokens_total += int(amount)
                    output = int(payload.get("output_tokens", 0))
                    tokens[TokenBucket.AGENT.value] += output
                    tokens_total += output
                case EventType.DIRECTOR_CALL:
                    director_cap = payload.get("cap") or director_cap
                    # The Director's calls are logged under their own event type
                    # and carry no input_by_bucket, so they are charged to the
                    # Director bucket here; llm_call alone would miss them.
                    director_calls += 1
                    cap_remaining = payload.get("cap_remaining", cap_remaining)
                    last_director_call = {
                        "n": payload.get("call_in_cycle"),
                        "cap": payload.get("cap"),
                        "rem": payload.get("cap_remaining"),
                        "it": payload.get("input_tokens", 0),
                        "ot": payload.get("output_tokens", 0),
                        "err": payload.get("error"),
                        # The prompt and the reply, as indices into the texts.
                        "ps": prompts.system(payload.get("system_sha")),
                        "pu": prompts.add(payload.get("user")),
                        "pr": prompts.add(payload.get("raw")),
                    }
                    sched.setdefault("calls", []).append(last_director_call)
                    charged = int(payload.get("input_tokens", 0)) + int(
                        payload.get("output_tokens", 0)
                    )
                    tokens[TokenBucket.DIRECTOR.value] += charged
                    tokens_total += charged
                case EventType.SHOCK_APPLIED:
                    shock_tick = tick
                    shock_details = _plain(payload)
                case EventType.SHOCK_REVERTED:
                    reversion_tick = tick
                case EventType.BUDGET_EXHAUSTED:
                    exhausted = True
                    exhaustion_tick = tick
                    budget_total = payload.get("cap") or budget_total
                case EventType.TAX_PAID if agent_id:
                    amount = _decimal(payload["amount"])
                    kind = str(payload.get("tax", ""))
                    charges[kind] += amount
                    agents.setdefault(agent_id, AgentFrame()).taxes += amount
                    line = trade_lines.get(str(payload.get("trade_id")))
                    if kind == "sales" and line is not None:
                        line["tx"] = str(payload["amount"])
                    elif kind == "wealth":
                        arrears = bool(payload.get("arrears"))
                        # One feed line per tick for the wealth tax, however many
                        # pay. The collection (v, q) and the arrears settled (ar,
                        # aq) are kept apart: paying arrears is not a collection,
                        # and "c" marks a tick that had one.
                        if wealth is None:
                            wealth = {
                                "e": EventType.TAX_PAID.value,
                                "a": "",
                                "k": kind,
                                "c": 0,
                                "v": 0.0,
                                "q": 0,
                                "ar": 0.0,
                                "aq": 0,
                            }
                            feed.insert(0, wealth)
                        # Arrears settled between collections are not a collection.
                        if not arrears and wealth_tax_ticks[-1:] != [tick]:
                            wealth_tax_ticks.append(tick)
                        # An assessment that could not be paid now becomes arrears;
                        # it is logged with nothing paid and counts as no payer.
                        if arrears:
                            wealth["ar"] = round(wealth["ar"] + amount, 2)
                            if amount > 0:
                                wealth["aq"] += 1
                        else:
                            wealth["c"] = 1
                            wealth["v"] = round(wealth["v"] + amount, 2)
                            if amount > 0:
                                wealth["q"] += 1
                        taxable = payload.get("taxable")
                        owed = payload.get("owed")
                        if owed is not None:
                            agents.setdefault(agent_id, AgentFrame()).owed = _decimal(owed)
                        wealth_rows.append(
                            [
                                agent_id,
                                _decimal(taxable) if taxable is not None else None,
                                round(amount, 2),
                                1 if arrears else 0,
                                _decimal(owed) if owed is not None else None,
                            ]
                        )
                case EventType.STORAGE_PAID:
                    charges["storage"] += _decimal(payload["amount"])
                case EventType.TOKENS_CONSUMED:
                    budget = payload.get("budget") or {}
                    budget_total = budget.get("cap") or budget_total
                    # Captured for the reconciliation below, which refuses to
                    # build a replay whose bars would contradict this.
                    ledger_totals = {b: t for b, t in (payload.get("total") or {}).items() if t}
                    ledger_authoritative = int(payload.get("authoritative_total", 0))
                case _:
                    pass

            if event.type in FEED_EVENTS:
                # The wealth-tax line is not an event line and takes no slot.
                shown = sum(1 for line in feed if line.get("e") != "tax_paid")
                if shown < FEED_PER_TICK or event.type in FEED_ALWAYS:
                    line = {"e": event.type.value, "a": agent_id or "", **_feed_detail(event)}
                    feed.append(line)
                    if event.type is EventType.TRADE_EXECUTED:
                        trade_lines[str(payload.get("trade_id"))] = line
                else:
                    feed_dropped += 1

        if feed_dropped:
            # Said, not silent: the viewer knows the tick had more.
            feed.append({"e": "more", "a": "", "q": feed_dropped})
        frames.append(
            TickFrame(
                tick=tick,
                cycle=cycle,
                prices=dict(prices),
                trades=trades,
                messages=messages,
                director=director,
                feed=feed,
                agents={aid: frame.to_payload() for aid, frame in sorted(agents.items())},
                tokens=dict(tokens),
                tokens_total=tokens_total,
                director_calls=director_calls,
                director_errors=director_errors,
                cap_remaining=cap_remaining,
                trades_total=trades_total,
                exhausted=exhausted,
                steps=steps,
                sched=sched,
                bulletin=bulletin,
                returns=returns,
                on_sale=_on_sale(book),
                quotes=_quotes(book),
                equipped={k: v for k, v in equipped.items() if v},
                zone_stock=(
                    {zone: dict(pool) for zone, pool in zone_stock.items()}
                    if zone_stock is not None
                    else None
                ),
                charges={k: round(v, 2) for k, v in sorted(charges.items())},
                wealth_tax=sorted(wealth_rows, key=lambda row: -row[2]),
            )
        )
        deltas.append(delta)
        gold_deltas.append(gold_delta)
        item_deltas.append(item_delta)

    _reconstruct_world_stock(frames, deltas, final_holdings)
    # Every agent the run had, including one the log never shows acting.
    agent_ids = sorted(set(agents) | set(final_holdings))
    _reconstruct_role_gold(frames, gold_deltas, final_holdings, agent_ids)
    _reconstruct_agent_items(frames, item_deltas, final_holdings, agent_ids)

    _reconcile(frames[-1] if frames else None, ledger_totals, ledger_authoritative, path)

    return ReplayTrack(
        run_id=metadata.run_id,
        condition=metadata.condition.value,
        scenario=metadata.scenario.value,
        experiment=metadata.experiment.value,
        seed=metadata.seed,
        model=metadata.llm_model,
        model_observed=", ".join(observed_models) or metadata.llm_model,
        prompt_version=str((metadata.prompt_hashes or {}).get("version", "?")),
        config_hash=metadata.config_hash,
        horizon=horizon,
        shock_tick=shock_tick,
        reversion_tick=reversion_tick,
        exhaustion_tick=exhaustion_tick,
        budget_total=budget_total,
        director_cap=director_cap,
        agent_ids=agent_ids,
        agent_roles=roles,
        goods=goods,
        frames=frames,
        agent_specs=specs,
        agent_traits=traits,
        shock=shock_details,
        wealth_tax_ticks=wealth_tax_ticks,
        prompts=prompts,
    )


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


def _extend_step(step: dict[str, Any], event: Event, prompts: PromptTexts) -> None:
    """Fold one of the acting agent's own events into its activation."""
    payload = event.payload
    engine: list[list[Any]] = step["e"]
    match event.type:
        case EventType.LLM_CALL:
            step["l"].append(
                [
                    payload.get("attempt", 1),
                    payload.get("input_tokens", 0),
                    payload.get("output_tokens", 0),
                    payload.get("finish_reason"),
                    payload.get("input_by_bucket") or {},
                    # System prompt, user prompt and reply, as text indices.
                    [
                        prompts.system(payload.get("system_sha")),
                        prompts.add(payload.get("user")),
                        prompts.add(payload.get("raw")),
                    ],
                ]
            )
        case EventType.AGENT_OUTPUT_INVALID:
            step["x"].append(
                [
                    payload.get("attempt"),
                    payload.get("reason"),
                    # A call that failed outright still sent its prompt.
                    [
                        prompts.system(payload.get("system_sha")),
                        prompts.add(payload.get("user")),
                    ]
                    if payload.get("user") is not None
                    else None,
                    _invalid_kind(payload),
                ]
            )
        case EventType.AGENT_OUTPUT:
            step["d"] = {
                "ac": payload.get("action"),
                "tg": payload.get("target"),
                "mt": payload.get("message_type"),
                "q": payload.get("quantity"),
                "p": payload.get("price"),
                "r": payload.get("recipient"),
                "pr": payload.get("priority"),
                "why": payload.get("short_reason"),
                # place_orders: [action, good, quantity, price] per order.
                "os": [
                    [o.get("action"), o.get("target"), o.get("quantity"), o.get("price")]
                    for o in payload.get("orders") or []
                ]
                or None,
            }
        case EventType.ORDER_PLACED:
            engine.append(
                [
                    "order",
                    payload.get("side"),
                    payload.get("good"),
                    payload.get("quantity"),
                    payload.get("limit_price"),
                    payload.get("order_id"),
                    payload.get("expires_at_tick"),
                ]
            )
        case EventType.ORDER_CANCELLED:
            engine.append(["cancel", payload.get("order_id"), payload.get("good")])
        case EventType.MESSAGE_SENT:
            engine.append(["msg", payload.get("type"), payload.get("recipient_id")])
        case EventType.PRODUCTION_COMPLETED:
            engine.append(
                [
                    "produce",
                    payload.get("product"),
                    payload.get("quantity"),
                    payload.get("consumed"),
                ]
            )
        case EventType.PRODUCTION_REJECTED:
            engine.append(["noproduce", payload.get("product"), payload.get("reason")])
        case EventType.EXPEDITION_STARTED:
            engine.append(
                [
                    "expedition",
                    payload.get("zone_id"),
                    payload.get("risk"),
                    payload.get("success_probability"),
                    payload.get("resolves_at_tick"),
                    payload.get("gear_bonus") or {},
                ]
            )
        case EventType.EQUIPMENT_EQUIPPED:
            engine.append(["equip", payload.get("product"), payload.get("durability")])
        case EventType.ACTION_REJECTED:
            # The closed code, which the page labels in its language; the free
            # text is the engine's English message and only a fallback.
            engine.append(
                [
                    "rejected",
                    payload.get("code") or payload.get("reason"),
                    # place_orders: which order of the batch (0-based).
                    payload.get("order_index"),
                ]
            )
        case EventType.ACTIVATION_WASTED:
            engine.append(["wasted", _waste_reason(engine, payload)])
        case EventType.ACTION_EXECUTED:
            step["ok"] = 1
        case EventType.RESULT_RETURNED:
            step["r"] = {
                "id": payload.get("result_id"),
                "st": payload.get("status"),
                "needs": payload.get("needs") or [],
                "offers": payload.get("offers") or [],
                "obs": payload.get("observations") or [],
                "out": payload.get("outcome") or {},
            }
        case _:
            pass


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


def trio_logs(directory: Path | str) -> list[Path]:
    """The three condition logs of a trio directory, in A/B/C order."""
    root = Path(directory)
    paths = [root / f"{condition}.jsonl" for condition in ("A", "B", "C")]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{root} is not a trio directory; missing {missing}")
    return paths


def wealth_tax_schedule(config: SimulationConfig) -> list[int]:
    """The ticks at which the wealth tax is collected over the run's horizon.

    The engine collects it at the start of the macro-cycle whose span holds a
    multiple of ``wealth_every_ticks``; the same rule is applied here, so a
    collection at which nobody held enough to pay still shows on the timeline.
    No collections at all when the rate is zero.
    """
    taxes, time = config.economy.taxes, config.time
    if taxes.wealth_rate <= 0:
        return []
    every, span = taxes.wealth_every_ticks, time.ticks_per_cycle
    ticks = []
    for start in range(0, time.horizon_ticks, span):
        due = -(-max(start, 1) // every) * every
        if due < start + span:
            ticks.append(start)
    return ticks


def world_rules(bundle: ReplayBundle, config: SimulationConfig) -> dict[str, Any] | None:
    """The static rules of the world, if ``config`` is the one every track ran under.

    The logs record the configuration only by hash, so the configuration is
    re-derived for each track (its scenario, experiment, seed and condition) and
    its hash compared with the logged one. Any mismatch returns None: a guide
    built from a different configuration would describe a different game.
    """
    for track in bundle.tracks:
        derived = config.for_cell(
            scenario=Scenario(track.scenario),
            experiment=ExperimentRegime(track.experiment),
            seed=track.seed,
        ).for_condition(Condition(track.condition), seed=track.seed)
        if config_hash(derived) != track.config_hash:
            return None

    economy, expeditions = config.economy, config.expeditions
    taxes = economy.taxes
    return {
        "wealth": {
            "exemption": str(taxes.wealth_exemption),
            "rate": str(taxes.wealth_rate),
            "every": taxes.wealth_every_ticks,
            "ticks": wealth_tax_schedule(config),
        },
        "recipes": {
            product.value: {
                "in": {r.value: q for r, q in recipe.inputs.items()},
                "out": recipe.output_quantity,
                "by": recipe.specialization.value,
            }
            for product, recipe in economy.recipes.items()
        },
        "gear": {
            s.value: {p.value: str(b) for p, b in gear.items()}
            for s, gear in expeditions.gear.items()
        },
        "durability": {p.value: ticks for p, ticks in economy.durability.items()},
        "wear": expeditions.durability_loss_per_expedition,
        "zones": [
            {
                "id": zone.id,
                "risk": zone.risk.value,
                "spawn": {r.value: str(rate) for r, rate in zone.spawn_per_tick.items()},
            }
            for zone in expeditions.zones
        ],
        "affinity": {
            s.value: {r.value: str(m) for r, m in settings.resource_affinity.items()}
            for s, settings in expeditions.subclasses.items()
        },
    }


def _invalid_kind(payload: dict[str, Any]) -> str:
    """Why an output was rejected, as a closed code the page labels in its
    language; the parser's own message stays as detail."""
    reason = str(payload.get("reason") or "")
    if payload.get("truncated"):
        return "truncated"
    if reason.startswith("llm error"):
        return "llm_error"
    if reason.startswith("no JSON object"):
        return "invalid_json"
    if "RESULT" in reason:
        return "result_fields"
    return "schema"
