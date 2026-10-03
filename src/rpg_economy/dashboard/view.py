"""Building an observation view from a run log (IMPLEMENTATION.md phase 7).

Reads logs. Writes nothing. Decides nothing. Every number shown here is derived
from events the run already recorded, so the dashboard can be pointed at a run
that finished months ago and at a run someone else produced.

Language lives at render time only: :class:`DashboardData` holds enums and
numbers, and :func:`render_dashboard` is the only place a locale is consulted.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from rpg_economy.domain import TokenBucket
from rpg_economy.i18n import format_number, set_locale, t
from rpg_economy.logging_ import Event, EventType, read_events, read_metadata

ZERO = Decimal(0)

#: Events the activity feed shows. Deliberately not every event type: a feed
#: that printed durability decay for every agent every tick would bury the
#: things a person is actually watching for.
FEED_EVENTS = (
    EventType.SHOCK_APPLIED,
    EventType.SHOCK_REVERTED,
    EventType.BUDGET_EXHAUSTED,
    EventType.EXPEDITION_RESOLVED,
    EventType.EQUIPMENT_BROKEN,
    EventType.DIRECTOR_ERROR,
    EventType.ACTIVATION_WASTED,
    EventType.AGENT_OUTPUT_INVALID,
)

#: Never dropped from the capped list of notable events.
KEY_FEED_EVENTS = frozenset(
    {
        EventType.SHOCK_APPLIED.value,
        EventType.SHOCK_REVERTED.value,
        EventType.BUDGET_EXHAUSTED.value,
        EventType.DIRECTOR_ERROR.value,
    }
)


@dataclass
class FeedEntry:
    """One notable event, or a run of identical ones collapsed together.

    Collapsing matters more than it sounds: a stalled artisan emits a wasted
    activation every tick, and an uncollapsed feed shows forty identical lines
    and nothing else. The interesting fact is "this agent stalled for 40 ticks",
    which is what a collapsed entry says.
    """

    first_tick: int
    last_tick: int
    event_type: str
    agent_id: str | None
    count: int

    @property
    def repeated(self) -> bool:
        return self.count > 1


@dataclass
class PricePoint:
    tick: int
    good: str
    price: Decimal
    quantity: int


@dataclass
class DashboardData:
    """One run, summarised for observation. Language-neutral throughout."""

    run_id: str
    condition: str
    scenario: str
    experiment: str
    seed: int
    ticks: int

    prices: dict[str, list[PricePoint]] = field(default_factory=dict)
    holdings: dict[str, dict[str, Any]] = field(default_factory=dict)
    event_counts: dict[str, int] = field(default_factory=dict)
    feed: list[FeedEntry] = field(default_factory=list)
    #: Notable-event entries left out by the cap, said rather than silent.
    feed_dropped: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    tokens: dict[str, Any] = field(default_factory=dict)
    trigger_counts: dict[str, int] = field(default_factory=dict)
    exhaustion_tick: int | None = None

    def last_price(self, good: str) -> Decimal | None:
        points = self.prices.get(good)
        return points[-1].price if points else None

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "scenario": self.scenario,
            "experiment": self.experiment,
            "seed": self.seed,
            "ticks": self.ticks,
            "goods_traded": sorted(self.prices),
            "event_counts": self.event_counts,
            "trigger_counts": self.trigger_counts,
            "messages": len(self.messages),
            "tokens": self.tokens,
            "exhaustion_tick": self.exhaustion_tick,
        }


def build_dashboard(log_path: Path | str, *, feed_limit: int = 40) -> DashboardData:
    """Read one run log into an observation view.

    Opened read-only. The dashboard never writes to a log, and a log it cannot
    parse is an error to report, not a file to repair.
    """
    path = Path(log_path)
    metadata = read_metadata(path)
    data = DashboardData(
        run_id=metadata.run_id,
        condition=metadata.condition.value,
        scenario=metadata.scenario.value,
        experiment=metadata.experiment.value,
        seed=metadata.seed,
        ticks=0,
    )

    counts: Counter[str] = Counter()
    triggers: Counter[str] = Counter()
    feed: dict[tuple[str, str | None], FeedEntry] = {}

    for event in read_events(path):
        counts[event.type.value] += 1
        data.ticks = max(data.ticks, event.tick)

        match event.type:
            case EventType.TRADE_EXECUTED:
                good = event.payload["good"]
                data.prices.setdefault(good, []).append(
                    PricePoint(
                        tick=event.tick,
                        good=good,
                        price=Decimal(event.payload["price"]),
                        quantity=event.payload["quantity"],
                    )
                )
            case EventType.TRIGGER_FIRED if not event.payload.get("pending"):
                # A repeat firing of a trigger the request already carries is
                # logged too (pending); count each trigger once per request.
                triggers[event.payload.get("trigger", "unknown")] += 1
            case EventType.MESSAGE_SENT:
                data.messages.append({"tick": event.tick, **event.payload})
            case EventType.TOKENS_CONSUMED:
                data.tokens = dict(event.payload)
            case EventType.BUDGET_EXHAUSTED:
                data.exhaustion_tick = event.tick
            case EventType.RUN_END:
                data.holdings = dict(event.payload.get("holdings") or {})

        if event.type in FEED_EVENTS:
            _append_to_feed(feed, event)

    data.event_counts = dict(counts)
    data.trigger_counts = dict(triggers)
    # Ordered by when each first appeared, capped: a feed nobody can scroll is
    # not an observation tool.
    ordered = sorted(feed.values(), key=lambda e: (e.first_tick, e.event_type))
    # The shock, budget exhaustion and Director errors always make the list;
    # the rest fill what is left, in order.
    key = [e for e in ordered if e.event_type in KEY_FEED_EVENTS]
    rest = [e for e in ordered if e.event_type not in KEY_FEED_EVENTS]
    kept = {id(e) for e in key + rest[: max(0, feed_limit - len(key))]}
    data.feed = [e for e in ordered if id(e) in kept]
    data.feed_dropped = len(ordered) - len(data.feed)
    return data


def _append_to_feed(feed: dict[tuple[str, str | None], FeedEntry], event: Event) -> None:
    """Fold an event into its (type, agent) entry, widening the tick range.

    Grouped by pair rather than by adjacency because agents interleave: six
    stalled artisans produce a round-robin of six event types per tick, so
    collapsing only consecutive repeats collapses nothing at all.
    """
    key = (event.type.value, event.agent_id)
    entry = feed.get(key)
    if entry is None:
        feed[key] = FeedEntry(
            first_tick=event.tick,
            last_tick=event.tick,
            event_type=event.type.value,
            agent_id=event.agent_id,
            count=1,
        )
        return
    entry.last_tick = event.tick
    entry.count += 1


def _bar(value: int, maximum: int, width: int = 24) -> str:
    if maximum <= 0:
        return ""
    return "#" * max(1, round(value / maximum * width)) if value else ""


def _collation_key(label: str) -> str:
    """Order labels as a reader would: ignoring case and accents."""
    decomposed = unicodedata.normalize("NFKD", label.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def render_dashboard(data: DashboardData, *, locale: str) -> str:
    """Render one run's observation view in one language."""
    set_locale(locale)

    lines = [
        t(
            "dashboard.header",
            run_id=data.run_id,
            condition=t(f"condition.{data.condition}"),
            scenario=t(f"scenario.{data.scenario}"),
            seed=data.seed,
        ),
        t(
            "dashboard.subheader",
            experiment=t(f"experiment.{data.experiment}"),
            ticks=data.ticks,
        ),
    ]

    if data.exhaustion_tick is not None:
        lines.append(t("dashboard.exhausted", tick=data.exhaustion_tick))

    # -- prices ------------------------------------------------------------
    if data.prices:
        lines += ["", t("dashboard.section.prices")]
        # Sorted by the label the reader sees, so each language reads in order.
        labels = {
            good: t(f"resource.{good}") if _is_resource(good) else t(f"product.{good}")
            for good in data.prices
        }
        for good in sorted(data.prices, key=lambda g: (_collation_key(labels[g]), g)):
            points = data.prices[good]
            prices = [p.price for p in points]
            lines.append(
                t(
                    "dashboard.price_row",
                    good=labels[good],
                    trades=format_number(len(points)),
                    first=format_number(prices[0], 2),
                    last=format_number(prices[-1], 2),
                    low=format_number(min(prices), 2),
                    high=format_number(max(prices), 2),
                )
            )

    # -- holdings ----------------------------------------------------------
    if data.holdings:
        lines += ["", t("dashboard.section.holdings")]
        for agent_id, holding in sorted(data.holdings.items()):
            inventory = holding.get("inventory") or {}
            lines.append(
                t(
                    "dashboard.holding_row",
                    agent=agent_id,
                    gold=format_number(Decimal(str(holding["gold"])), 2),
                    items=format_number(sum(inventory.values())),
                )
            )

    # -- triggers ----------------------------------------------------------
    if data.trigger_counts:
        lines += ["", t("dashboard.section.triggers")]
        peak = max(data.trigger_counts.values())
        for trigger in sorted(data.trigger_counts):
            count = data.trigger_counts[trigger]
            lines.append(
                t(
                    "dashboard.trigger_row",
                    trigger=t(f"trigger.{trigger}"),
                    count=format_number(count),
                    bar=_bar(count, peak),
                )
            )

    # -- tokens ------------------------------------------------------------
    if data.tokens:
        lines += ["", t("dashboard.section.tokens")]
        # Fixed bucket order (agent, director, p2p_content, bulletin), the same
        # as the replay's bars; an unknown bucket goes last.
        order = {bucket.value: i for i, bucket in enumerate(TokenBucket)}
        totals = data.tokens.get("total") or {}
        for bucket, total in sorted(
            totals.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])
        ):
            lines.append(
                t("dashboard.token_row", bucket=t(f"bucket.{bucket}"), tokens=format_number(total))
            )
        lines.append(
            t(
                "dashboard.token_total",
                total=format_number(data.tokens.get("authoritative_total", 0)),
            )
        )

    # -- feed --------------------------------------------------------------
    if data.event_counts:
        lines += ["", t("dashboard.section.event_counts")]
        for name, count in sorted(data.event_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
            lines.append(
                t("dashboard.event_count_row", event=t(f"event.{name}"), count=format_number(count))
            )

    if data.feed:
        lines += ["", t("dashboard.section.feed")]
        for entry in data.feed:
            key = "dashboard.feed_run" if entry.repeated else "dashboard.feed_row"
            lines.append(
                t(
                    key,
                    tick=entry.first_tick,
                    last_tick=entry.last_tick,
                    count=format_number(entry.count),
                    event=t(f"event.{entry.event_type}"),
                    agent=entry.agent_id or "-",
                )
            )
        if data.feed_dropped:
            lines.append(t("dashboard.feed.more", n=format_number(data.feed_dropped)))

    lines += ["", t("dashboard.note.read_only")]
    return "\n".join(lines)


def _is_resource(good: str) -> bool:
    return good in {"iron", "wood", "leather", "herbs", "arcane_crystal"}


# ---------------------------------------------------------------------------
# Live tail (optional, for monitoring a pilot in progress)
# ---------------------------------------------------------------------------
def tail_events(
    log_path: Path | str, *, from_offset: int = 0, types: tuple[EventType, ...] = FEED_EVENTS
) -> tuple[list[Event], int]:
    """Read events appended since ``from_offset``; return them and the new offset.

    Deliberately a plain re-read from a byte offset rather than a file watcher:
    run logs are append-only, so an offset is all the state a tail needs, and a
    partially written final line is simply not consumed until it is complete.
    The dashboard still only reads.
    """
    path = Path(log_path)
    if not path.exists():
        return [], from_offset

    with path.open("rb") as handle:
        handle.seek(from_offset)
        raw = handle.read()

    # Stop at the last newline: anything after it is a line still being written.
    cut = raw.rfind(b"\n")
    if cut < 0:
        return [], from_offset
    consumed = from_offset + cut + 1

    wanted = set(types)
    events: list[Event] = []
    for line in raw[: cut + 1].decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = Event.model_validate_json(line)
        except Exception:  # a line we cannot parse is skipped, never repaired
            continue
        if event.type in wanted:
            events.append(event)
    return events, consumed
