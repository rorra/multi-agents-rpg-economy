"""Per-run metric computation (design v1.0 s11).

Everything here is derived from a run's event log, which s15 requires be
sufficient to reconstruct the run. Analysis reads logs; it never mutates them.

The censoring rule of s11.1 is the load-bearing part. When recovery does not
occur inside the analysed horizon, the observation is marked censored at the
maximum observed time and never replaced by an artificial recovery time.
Substituting the horizon as though it were a recovery would silently convert
"we do not know how long this took" into "it took exactly this long", biasing
every condition that recovers slowly toward looking better than it is.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import ALL_GOODS, OrderSide, Resource, TokenBucket
from rpg_economy.logging_ import Event, EventType, read_events, read_metadata
from rpg_economy.metrics.targets import PrimaryTarget, TargetKind, propagation_product

ZERO = Decimal(0)

PHASE_BASE = "base"
PHASE_SHOCK = "shock"
PHASE_RECOVERY = "recovery"
PHASES = (PHASE_BASE, PHASE_SHOCK, PHASE_RECOVERY)


# ---------------------------------------------------------------------------
# Censored observations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Duration:
    """A possibly-censored duration (design v1.0 s11.1).

    ``censored`` means the event had not occurred when observation stopped, so
    ``value`` is a lower bound, not a measurement. Consumers must respect that:
    a censored observation is never a recovery time.
    """

    value: int
    censored: bool

    @property
    def observed(self) -> bool:
        return not self.censored

    def to_payload(self) -> dict[str, Any]:
        return {"value": self.value, "censored": self.censored}


# ---------------------------------------------------------------------------
# Reading a run
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    tick: int
    good: str
    quantity: int
    price: Decimal
    value: Decimal


class RunLog:
    """An indexed view of one run's events."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.metadata = read_metadata(self.path)
        self.events: list[Event] = list(read_events(self.path))
        self.trades: list[Trade] = [
            Trade(
                tick=e.tick,
                good=e.payload["good"],
                quantity=e.payload["quantity"],
                price=Decimal(e.payload["price"]),
                value=Decimal(e.payload["value"]),
            )
            for e in self.events
            if e.type is EventType.TRADE_EXECUTED
        ]

    def of_type(self, *types: EventType) -> Iterator[Event]:
        wanted = set(types)
        return (e for e in self.events if e.type in wanted)

    def count(self, event_type: EventType) -> int:
        return sum(1 for _ in self.of_type(event_type))

    @property
    def last_tick(self) -> int:
        return max((e.tick for e in self.events), default=0)

    @property
    def final_holdings(self) -> dict[str, dict[str, Any]]:
        """Every agent's closing gold and inventory, from the run_end event.

        Read from the log rather than from a live WorldState so analysis stays
        log-only (s15). Sourcing them here means a missing argument can never
        yield a Gini of 0.0 from an absence of data (design s11.1/s11.3).
        """
        for event in reversed(self.events):
            if event.type is EventType.RUN_END:
                return dict(event.payload.get("holdings") or {})
        return {}


def phase_of(tick: int, config: SimulationConfig) -> str:
    if tick < config.phases.shock_tick:
        return PHASE_BASE
    if tick < config.phases.reversion_tick:
        return PHASE_SHOCK
    return PHASE_RECOVERY


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------
def price_series(
    trades: Iterable[Trade], good: str, horizon: int, window: int
) -> list[Decimal | None]:
    """A per-tick price series, smoothed over the trailing ``window`` ticks.

    ``None`` where the good did not trade in the window. An undefined tick can
    never count as inside a recovery band: the price is unknown, not recovered.
    """
    by_tick: dict[int, list[Decimal]] = {}
    for trade in trades:
        if trade.good == good:
            by_tick.setdefault(trade.tick, []).append(trade.price)

    series: list[Decimal | None] = []
    for tick in range(horizon):
        prices = [p for t in range(max(0, tick - window + 1), tick + 1) for p in by_tick.get(t, ())]
        series.append(sum(prices, ZERO) / len(prices) if prices else None)
    return series


def value_series(trades: Iterable[Trade], horizon: int, window: int) -> list[Decimal | None]:
    """Total traded value per trailing window -- the liquidity-loss target."""
    by_tick: dict[int, Decimal] = {}
    for trade in trades:
        by_tick[trade.tick] = by_tick.get(trade.tick, ZERO) + trade.value

    series: list[Decimal | None] = []
    for tick in range(horizon):
        total = sum(
            (by_tick.get(t, ZERO) for t in range(max(0, tick - window + 1), tick + 1)),
            ZERO,
        )
        series.append(total)
    return series


def target_series(
    log: RunLog, config: SimulationConfig, target: PrimaryTarget
) -> list[Decimal | None]:
    """The series the scenario's recovery is measured on (design v1.0 s11.3)."""
    horizon = config.time.horizon_ticks
    window = config.metrics.volatility_window_w
    if target.kind is TargetKind.TRADED_VALUE:
        return value_series(log.trades, horizon, window)
    assert target.good is not None
    return price_series(log.trades, target.good.value, horizon, window)


# ---------------------------------------------------------------------------
# Recovery, with censoring
# ---------------------------------------------------------------------------
def recovery_reference(series: list[Decimal | None], config: SimulationConfig) -> Decimal | None:
    """The target over the last W ticks of the base phase (design s11.1).

    The target series is already a trailing W-tick window (the mean trade price,
    or the traded value, over the last W ticks), so its value at the last base
    tick *is* the reference: the mean price of the trades in the last W base
    ticks. Averaging the smoothed series again would widen the window to 2W-1.
    ``None`` when nothing traded in that window.
    """
    last_base = config.phases.shock_tick - 1
    if not 0 <= last_base < len(series):
        return None
    return series[last_base]


def recovery_time(
    series: list[Decimal | None], config: SimulationConfig, *, horizon: int | None = None
) -> Duration | None:
    """Ticks from the reversion tick until the target holds inside its band.

    Recovery requires the target to sit within +/-X% of its base-phase reference
    for M consecutive ticks (design v1.0 s11.1). If that never happens
    before observation stops, the result is censored at the time observed --
    never imputed.

    ``None`` only when recovery cannot be measured at all: a horizon that has
    observed the last base tick, yet no reference (nothing traded at the end of
    the base phase). That is missing, not "did not
    recover" (design s11.3): counting it as a non-recovery would charge thin
    base-phase markets, themselves an effect of the treatment, with a failure
    they were never able to show.

    A horizon that ends at or before the reversion tick (the common active
    horizon of Experiment 2, design s8.4) leaves no recovery phase to observe.
    The censoring rule applies on that horizon too, so the observation is
    censored at the time observed -- zero ticks -- not dropped as missing.

    That holds even for a horizon ending at or before the last base tick, which
    has not observed the reference: recovery was not observed within the
    horizon, so the observation is censored at zero ticks, never missing. The
    answer needs no reference, so no tick past the horizon is read.
    """
    end = horizon if horizon is not None else config.time.horizon_ticks
    end = min(end, len(series))
    start = config.phases.reversion_tick

    if horizon is not None and horizon <= config.phases.shock_tick - 1:
        return Duration(value=0, censored=True)
    reference = recovery_reference(series, config)
    if reference is None or reference <= ZERO:
        return None
    if start >= end:
        return Duration(value=0, censored=True)

    band = config.metrics.recovery_band_pct_x
    low, high = reference * (Decimal(1) - band), reference * (Decimal(1) + band)
    hold = config.metrics.recovery_hold_ticks_m

    consecutive = 0
    for tick in range(start, end):
        value = series[tick]
        # An unknown value breaks the run: it is not evidence of recovery.
        if value is not None and low <= value <= high:
            consecutive += 1
            if consecutive >= hold:
                # Recovery is dated to when the qualifying stretch began.
                return Duration(value=(tick - hold + 1) - start, censored=False)
        else:
            consecutive = 0

    return Duration(value=end - start, censored=True)


# ---------------------------------------------------------------------------
# Economic metrics
# ---------------------------------------------------------------------------
def _phase_bounds(config: SimulationConfig, until: int | None = None) -> dict[str, tuple[int, int]]:
    """Tick bounds of each phase, so every metric slices the run identically.

    ``until`` clips every phase at that tick (the common active horizon of
    Experiment 2, design s8.4). A phase that starts at or after it is empty.
    """
    bounds = {
        PHASE_BASE: (0, config.phases.shock_tick),
        PHASE_SHOCK: (config.phases.shock_tick, config.phases.reversion_tick),
        PHASE_RECOVERY: (config.phases.reversion_tick, config.time.horizon_ticks),
    }
    if until is None:
        return bounds
    return {phase: (lo, max(lo, min(hi, until))) for phase, (lo, hi) in bounds.items()}


#: A window needs at least this many trades before a dispersion can be computed
#: from it. One trade has no dispersion, and zero trades have no price at all
#: (design s11.1/s11.3).
MIN_TRADES_PER_WINDOW = 2


def volatility_by_phase(
    log: RunLog, config: SimulationConfig, good: str, *, until: int | None = None
) -> dict[str, float | None]:
    """Standard deviation of transaction price in windows W, per phase (s11.1).

    Reported as the mean of per-window deviations rather than one deviation over
    the whole phase, so a phase that drifts steadily is not scored as volatile
    merely for having moved.

    Insufficient data is missing, never zero (design s11.1/s11.3). A window
    with fewer than two trades yields no dispersion and is excluded from the
    phase mean; a phase in which no window qualifies returns ``None``. Returning
    0.0 there would state that the price was perfectly stable, which is the
    opposite of what thin trading shows -- a phase with no trades has no price,
    and one with two trades far apart in time can hide a large move. Because
    zero is also the most favourable value this metric can take, defaulting to
    it would bias a primary metric toward whichever condition traded least.

    ``until`` clips the phases at that tick; an empty phase is missing.
    """
    window = config.metrics.volatility_window_w
    result: dict[str, float | None] = {}
    for phase, (lo, hi) in _phase_bounds(config, until).items():
        deviations: list[float] = []
        for start in range(lo, hi, window):
            prices = [
                float(t.price)
                for t in log.trades
                if t.good == good and start <= t.tick < min(start + window, hi)
            ]
            if len(prices) >= MIN_TRADES_PER_WINDOW:
                deviations.append(statistics.stdev(prices))
        result[phase] = statistics.mean(deviations) if deviations else None
    return result


#: A phase needs at least this many trades before a phase-level dispersion can
#: be computed. The robust companion definition of design s11.3: no
#: per-window requirement, so a phase whose trades are scattered thinly across
#: the horizon is still measurable.
MIN_TRADES_PER_PHASE = 2


def volatility_phase_level(
    log: RunLog, config: SimulationConfig, good: str, *, until: int | None = None
) -> dict[str, float | None]:
    """Standard deviation over all of a phase's transaction prices (s11.3).

    The pre-registered robust companion to :func:`volatility_by_phase`. The
    window-mean definition of s11.1 is the more faithful measure of instability
    -- it does not score a steady drift as volatile -- but its per-window
    minimum makes measurability depend on trading density, which is itself an
    effect of the treatment. Requiring only two trades in the whole phase makes
    the metric measurable wherever a price exists at all, so a contrast on it is
    not conditioned on a condition having happened to trade.

    Still ``None`` below two trades in the phase (missing, never zero): a condition that
    traded once or not at all across the entire shock window is reported as
    unmeasurable by treatment, which is a finding, not a gap to be filled.
    ``until`` clips the phases at that tick, as in :func:`volatility_by_phase`.
    """
    result: dict[str, float | None] = {}
    for phase, (lo, hi) in _phase_bounds(config, until).items():
        prices = [float(t.price) for t in log.trades if t.good == good and lo <= t.tick < hi]
        result[phase] = statistics.stdev(prices) if len(prices) >= MIN_TRADES_PER_PHASE else None
    return result


def volatility_windows(
    log: RunLog, config: SimulationConfig, good: str | None
) -> dict[str, dict[str, int]]:
    """Per-phase window census behind :func:`volatility_by_phase`.

    Reported so a missing volatility can be told from a measured one, and so the
    reason for it is visible: how many windows carried enough trades to yield a
    dispersion, and how many were too thin.
    """
    window = config.metrics.volatility_window_w
    census: dict[str, dict[str, int]] = {}
    for phase, (lo, hi) in _phase_bounds(config).items():
        counts = {"windows": 0, "measurable": 0, "thin": 0, "trades": 0}
        for start in range(lo, hi, window):
            if good is None:
                break
            n = sum(
                1
                for t in log.trades
                if t.good == good and start <= t.tick < min(start + window, hi)
            )
            counts["windows"] += 1
            counts["trades"] += n
            if n >= MIN_TRADES_PER_WINDOW:
                counts["measurable"] += 1
            else:
                counts["thin"] += 1
        census[phase] = counts
    return census


def _ask_stockout_samples(log: RunLog, good: str) -> Iterator[tuple[int, bool]]:
    """``(tick, stocked_out)`` at every tick sample: no resting ask for the good.

    The order book is replayed from its logged transitions, which also confirms
    the log carries enough to reconstruct it (s15).
    """
    open_asks: set[int] = set()
    for event in log.events:
        payload = event.payload
        if event.type is EventType.ORDER_PLACED:
            if payload["good"] == good and payload["side"] == OrderSide.SELL.value:
                open_asks.add(payload["order_id"])
        elif event.type in {
            EventType.ORDER_FILLED,
            EventType.ORDER_EXPIRED,
            EventType.ORDER_CANCELLED,
        }:
            open_asks.discard(payload["order_id"])
        elif event.type is EventType.TICK:
            yield event.tick, not open_asks


def _count_stockouts(
    samples: Iterable[tuple[int, bool]], config: SimulationConfig, until: int | None
) -> dict[str, Any]:
    by_phase = dict.fromkeys(PHASES, 0)
    episodes = 0
    longest = 0
    current = 0
    for tick, stocked_out in samples:
        if until is not None and tick >= until:
            break
        if stocked_out:
            by_phase[phase_of(tick, config)] += 1
            current += 1
            if current == 1:
                episodes += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "episodes": episodes,
        "total_ticks": sum(by_phase.values()),
        "longest": longest,
        "by_phase": by_phase,
    }


def stockouts(
    log: RunLog, config: SimulationConfig, good: str, *, until: int | None = None
) -> dict[str, Any]:
    """Ticks with no resting ask for the good (design s11.1, primary definition).

    ``until`` stops the count at that tick (the common active horizon of
    Experiment 2).
    """
    return {"good": good, **_count_stockouts(_ask_stockout_samples(log, good), config, until)}


def stockout_ticks_between(log: RunLog, good: str, lo: int, hi: int) -> int:
    """Ticks in ``[lo, hi)`` with no resting ask for the good."""
    return sum(
        1
        for tick, stocked_out in _ask_stockout_samples(log, good)
        if stocked_out and lo <= tick < hi
    )


def _inventory_delta(event: Event, good: str) -> int:
    """How one event changes the world's total holdings of ``good``.

    Trades only move units between agents, so they never change the total.
    Units enter through expeditions and production, and leave through
    production inputs, consumption and equipping.
    """
    payload = event.payload
    match event.type:
        case EventType.EXPEDITION_RESOLVED:
            return int((payload.get("yielded") or {}).get(good, 0))
        case EventType.PRODUCTION_COMPLETED:
            made = int(payload.get("quantity", 0)) if payload.get("product") == good else 0
            return made - int((payload.get("consumed") or {}).get(good, 0))
        case EventType.CONSUMPTION if payload.get("product") == good:
            return -int(payload.get("quantity", 0))
        case EventType.EQUIPMENT_EQUIPPED if payload.get("product") == good:
            return -1
    return 0


def inventory_stockouts(
    log: RunLog, config: SimulationConfig, good: str, *, until: int | None = None
) -> dict[str, Any] | None:
    """Ticks with no unit of the good held anywhere in the world (s11.1, secondary).

    The secondary stockout definition: the world's total inventory of the good
    is zero. Reconstructed from the log alone (s15), backwards from the final
    holdings in ``run_end`` through every event that creates or destroys units,
    then sampled at each tick like the primary definition.

    ``None`` when the log carries no final holdings, or when the replay goes
    negative -- the log would then be missing an inventory event, and a count
    built on it would be wrong rather than merely thin. ``until`` stops the
    count at that tick (the common active horizon of Experiment 2).
    """
    holdings = log.final_holdings
    if not holdings:
        return None
    final = sum(int((h.get("inventory") or {}).get(good, 0)) for h in holdings.values())
    held = final - sum(_inventory_delta(event, good) for event in log.events)
    if held < 0:
        return None
    samples: list[tuple[int, bool]] = []
    for event in log.events:
        held += _inventory_delta(event, good)
        if held < 0:
            return None
        if event.type is EventType.TICK:
            samples.append((event.tick, held == 0))
    return {"good": good, **_count_stockouts(samples, config, until)}


def useful_production(
    log: RunLog, config: SimulationConfig, *, until: int | None = None
) -> dict[str, Any]:
    """Units placed within T ticks of being produced (s11.1).

    A unit counts as placed when its own producer sells it within T ticks.
    Strictly first-in first-out per producer and product: every sale consumes
    the oldest lot that still holds units, whether or not that lot's window has
    already closed. A late sale of an old unit is late, and is never credited
    to a newer lot that happens to still be inside its window. Trades by anyone
    else (a buyer reselling, an adventurer's starting potions) say nothing about
    whether this producer's output found a buyer. The complement -- units still
    unplaced after T ticks -- is the design's proxy for overproduction.

    ``useful_fraction`` is ``None`` when nothing was produced: no output is
    missing, not a 0% success (design s11.3).

    ``until`` ends observation at that tick (the common active horizon of
    Experiment 2, design s8.4): nothing at or after it is seen, and the
    right-censoring of the last T ticks applies at that horizon instead.
    """
    horizon_t = config.metrics.useful_production_ticks_t
    end = config.time.horizon_ticks if until is None else min(until, config.time.horizon_ticks)
    lots: dict[tuple[str, str], deque[list[int]]] = {}
    total = 0
    useful = 0
    for event in log.events:
        if event.tick >= end:
            continue
        if event.type is EventType.PRODUCTION_COMPLETED and event.agent_id:
            # Right-censored: units made in the last T ticks never had their
            # full window to sell in, so they count neither way.
            if event.tick + horizon_t >= end:
                continue
            quantity = int(event.payload["quantity"])
            total += quantity
            lots.setdefault((event.agent_id, event.payload["product"]), deque()).append(
                [event.tick, quantity]
            )
        elif event.type is EventType.TRADE_EXECUTED:
            queue = lots.get((event.payload.get("seller_id"), event.payload["good"]))
            left = int(event.payload["quantity"])
            while left > 0 and queue:
                lot = queue[0]
                produced_tick, remaining = lot
                take = min(left, remaining)
                lot[1] -= take
                left -= take
                if event.tick <= produced_tick + horizon_t:
                    useful += take
                if lot[1] <= 0:
                    queue.popleft()

    return {
        "produced": total,
        "useful": useful,
        "overproduction": total - useful,
        "useful_fraction": (useful / total) if total else None,
        "window_t": horizon_t,
    }


def net_gold(holding: dict[str, Any]) -> Decimal:
    """An agent's gold less the wealth tax it still owes.

    Arrears are a lien on the agent's gold: owed, and collected as soon as gold
    becomes available, so they are not the agent's wealth. A log written before
    arrears were recorded carries no field, read as nothing owed.
    """
    return Decimal(str(holding.get("gold", 0))) - Decimal(str(holding.get("tax_arrears") or 0))


def gini(
    log: RunLog,
    config: SimulationConfig,
    holdings: dict[str, dict[str, Any]],
    *,
    until: int | None = None,
) -> float | None:
    """Gini over net gold plus inventory at fixed reference prices (design s11.1).

    Gold is net of tax arrears (:func:`net_gold`). Reserved and frozen gold stay
    in: both are still the agent's.

    Inventory is valued at each good's mean base-phase transaction price, or its
    configured reference where it did not trade then. Valuing at concurrent
    prices would make Gini partly an artifact of volatility, which is itself a
    dependent variable of the study.

    An agent whose arrears exceed everything it holds counts as holding
    nothing: Gini is defined on non-negative wealth, and the lien cannot take
    more than the agent has.

    ``None`` when there are no holdings to measure. Zero would assert perfect
    equality, which is the same fabrication s11.3 forbids for volatility.
    ``until`` limits the valuation basis to the trades seen before that tick
    (the common active horizon; pass the holdings at that tick).
    """
    reference = base_phase_reference_prices(log, config, until=until)
    wealth: list[float] = []
    for holding in holdings.values():
        total = net_gold(holding)
        for good, units in (holding.get("inventory") or {}).items():
            total += reference.get(good, ZERO) * units
        wealth.append(max(0.0, float(total)))

    if not wealth or sum(wealth) <= 0:
        # No measurable distribution: missing, not equal.
        return None
    wealth.sort()
    n = len(wealth)
    cumulative = sum((i + 1) * value for i, value in enumerate(wealth))
    return (2 * cumulative) / (n * sum(wealth)) - (n + 1) / n


def _holding_deltas(event: Event) -> Iterator[tuple[str, str | None, Decimal]]:
    """``(agent, good or None for gold, change)`` for every holding an event moves.

    The complete list of what changes an agent's gold or inventory, the same
    flows the economy check reconciles gold against.
    """
    payload = event.payload
    match event.type:
        case EventType.TRADE_EXECUTED:
            buyer = payload.get("buyer_id") or payload.get("buyer")
            seller = payload.get("seller_id") or payload.get("seller")
            value = Decimal(str(payload["value"]))
            quantity = Decimal(int(payload["quantity"]))
            if buyer:
                yield buyer, None, -value
                yield buyer, payload["good"], quantity
            if seller:
                yield seller, None, value
                yield seller, payload["good"], -quantity
        case EventType.EXPEDITION_RESOLVED if event.agent_id:
            yield event.agent_id, None, Decimal(str(payload.get("gold", "0")))
            for good, units in (payload.get("yielded") or {}).items():
                yield event.agent_id, good, Decimal(int(units))
        case EventType.TAX_PAID | EventType.STORAGE_PAID if event.agent_id:
            yield event.agent_id, None, -Decimal(str(payload["amount"]))
        case EventType.GOODS_CONFISCATED if event.agent_id:
            yield event.agent_id, None, Decimal(str(payload["gold"]))
            yield event.agent_id, payload["good"], -Decimal(int(payload["quantity"]))
        case EventType.PRODUCTION_COMPLETED if event.agent_id:
            yield event.agent_id, payload["product"], Decimal(int(payload.get("quantity", 0)))
            for good, units in (payload.get("consumed") or {}).items():
                yield event.agent_id, good, -Decimal(int(units))
        case EventType.CONSUMPTION if event.agent_id:
            yield event.agent_id, payload["product"], -Decimal(int(payload.get("quantity", 0)))
        case EventType.EQUIPMENT_EQUIPPED if event.agent_id:
            yield event.agent_id, payload["product"], Decimal(-1)


def holdings_at(
    log: RunLog, until: int, final: dict[str, dict[str, Any]] | None = None
) -> dict[str, dict[str, Any]] | None:
    """Every agent's gold, arrears and inventory at the start of tick ``until``.

    Reconstructed from the log alone (s15): backwards from the final holdings
    through every event at or after ``until`` that moves gold or goods. Arrears
    are the ``owed`` recorded by the last wealth-tax event before ``until``
    (none recorded: nothing owed).

    ``None`` with no final holdings, or when the replay goes negative -- the
    log would then be missing an event, and a Gini built on it would be wrong
    rather than merely thin.
    """
    final = final if final is not None else log.final_holdings
    if not final:
        return None
    gold = {agent: Decimal(str(h.get("gold", 0))) for agent, h in final.items()}
    goods = {
        agent: {good: Decimal(int(n)) for good, n in (h.get("inventory") or {}).items()}
        for agent, h in final.items()
    }
    arrears = dict.fromkeys(final, ZERO)
    for event in log.events:
        if event.tick >= until:
            for agent, good, change in _holding_deltas(event):
                if good is None:
                    gold[agent] = gold.get(agent, ZERO) - change
                else:
                    held = goods.setdefault(agent, {})
                    held[good] = held.get(good, ZERO) - change
        elif event.type is EventType.TAX_PAID and event.agent_id and "owed" in event.payload:
            arrears[event.agent_id] = Decimal(str(event.payload["owed"]))
    if any(v < ZERO for v in gold.values()) or any(
        n < ZERO for held in goods.values() for n in held.values()
    ):
        return None
    return {
        agent: {
            "gold": str(gold.get(agent, ZERO)),
            "tax_arrears": str(arrears.get(agent, ZERO)),
            "inventory": {good: int(n) for good, n in sorted(goods.get(agent, {}).items()) if n},
        }
        for agent in sorted(set(gold) | set(goods))
    }


def base_phase_reference_prices(
    log: RunLog, config: SimulationConfig, *, until: int | None = None
) -> dict[str, Decimal]:
    """Per-good valuation basis, fixed for the run (design s11.1).

    ``until`` hides the trades at or after that tick, so an analysis cut at the
    common active horizon never values holdings with prices it did not observe.
    """
    cutoff = config.phases.shock_tick if until is None else min(config.phases.shock_tick, until)
    prices: dict[str, list[Decimal]] = {}
    for trade in log.trades:
        if trade.tick < cutoff:
            prices.setdefault(trade.good, []).append(trade.price)

    reference: dict[str, Decimal] = {}
    for good in ALL_GOODS:
        traded = prices.get(good.value)
        reference[good.value] = (
            sum(traded, ZERO) / len(traded)
            if traded
            else config.economy.initial_reference_price.get(good.value, ZERO)
        )
    return reference


# ---------------------------------------------------------------------------
# Pre-registered secondary analyses (design s11.1, s11.3)
# ---------------------------------------------------------------------------
def volume_by_good(
    log: RunLog, config: SimulationConfig, *, until: int | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """Units and value traded per good and phase (design s11.1, "Volumen").

    ``until`` leaves out the trades at or after that tick.
    """
    table: dict[str, dict[str, dict[str, Any]]] = {
        good.value: {phase: {"units": 0, "value": 0.0} for phase in PHASES} for good in ALL_GOODS
    }
    for trade in log.trades:
        if until is not None and trade.tick >= until:
            continue
        cell = table.setdefault(
            trade.good, {phase: {"units": 0, "value": 0.0} for phase in PHASES}
        )[phase_of(trade.tick, config)]
        cell["units"] += trade.quantity
        cell["value"] += float(trade.value)
    return table


def _mean_price_by_phase(
    log: RunLog, config: SimulationConfig, good: str, *, until: int | None = None
) -> dict[str, float | None]:
    prices: dict[str, list[float]] = {phase: [] for phase in PHASES}
    for trade in log.trades:
        if trade.good == good and (until is None or trade.tick < until):
            prices[phase_of(trade.tick, config)].append(float(trade.price))
    # No trade in a phase: no price, missing rather than zero.
    return {phase: (statistics.fmean(v) if v else None) for phase, v in prices.items()}


def propagation(
    log: RunLog, config: SimulationConfig, resource: str, *, until: int | None = None
) -> tuple[dict[str, Any], Duration | None] | None:
    """How a shock to a raw material reaches its main finished product (s11.3).

    The pre-registered secondary propagation analysis: the product comes from
    the fixed resource -> product mapping. Reported on that product: its mean
    transaction price per phase, its stockout ticks per phase (primary
    definition), and its recovery time, censored exactly like the target's.
    ``None`` when the target is not a raw material (there is nothing upstream
    to propagate from). ``until`` cuts every part of it at that tick, the
    recovery included (censored there, as the target's is).
    """
    try:
        product = propagation_product(Resource(resource))
    except ValueError:
        return None
    if product is None:
        return None
    series = price_series(
        log.trades, product.value, config.time.horizon_ticks, config.metrics.volatility_window_w
    )
    counted = stockouts(log, config, product.value, until=until)
    payload = {
        "source": resource,
        "good": product.value,
        "mean_price": _mean_price_by_phase(log, config, product.value, until=until),
        "stockout_ticks": counted["by_phase"],
    }
    return payload, recovery_time(series, config, horizon=until)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def multi_good_aggregate(
    log: RunLog,
    config: SimulationConfig,
    volumes: dict[str, dict[str, dict[str, Any]]],
    *,
    until: int | None = None,
) -> dict[str, Any]:
    """Secondary metrics over every good, one value per run (design s11.3).

    The pre-registered rule for metrics that involve several equivalent goods
    and no directed shock: each good is normalized by its own base-phase value,
    then the median across goods is the run's value. Normalizing first keeps a
    good with high prices or volumes from dominating the aggregate.

    * ``volatility``: phase-level shock volatility over base volatility;
    * ``volume``: units traded per tick in the shock phase over the base phase
      (per tick, so phases of different lengths compare);
    * ``stockouts``: the share of shock-phase ticks without a resting ask
      minus the share of base-phase ticks, a difference rather than a ratio:
      a good that never stocked out in the base phase is exactly the one a
      scarcity shock can empty, and a ratio would drop it. Where the scenario
      has no target good (liquidity_loss) this is how stockouts enter as a
      secondary metric (design s11.3).

    For volatility and volume, a good with no base-phase value (no dispersion,
    nothing traded) has no ratio and is left out of the median; with no value
    at all the aggregate is missing.

    ``until`` clips both phases at that tick (the common active horizon); the
    rates are per tick of the clipped phases, and ``volumes`` must be clipped
    the same way. A horizon at or before the shock tick leaves no shock phase,
    so every ratio is missing.
    """
    bounds = _phase_bounds(config, until)
    base_ticks = bounds[PHASE_BASE][1] - bounds[PHASE_BASE][0]
    shock_ticks = bounds[PHASE_SHOCK][1] - bounds[PHASE_SHOCK][0]
    per_good: dict[str, dict[str, float | None]] = {}
    for good in ALL_GOODS:
        volatility = volatility_phase_level(log, config, good.value, until=until)
        units = volumes.get(good.value, {})
        base_rate = units.get(PHASE_BASE, {}).get("units", 0) / base_ticks if base_ticks else None
        shock_rate = (
            units.get(PHASE_SHOCK, {}).get("units", 0) / shock_ticks if shock_ticks else None
        )
        out_ticks = stockouts(log, config, good.value, until=until)["by_phase"]
        per_good[good.value] = {
            "volatility": _ratio(volatility[PHASE_SHOCK], volatility[PHASE_BASE]),
            "volume": _ratio(shock_rate, base_rate),
            "stockouts": (
                out_ticks[PHASE_SHOCK] / shock_ticks - out_ticks[PHASE_BASE] / base_ticks
                if shock_ticks and base_ticks
                else None
            ),
        }

    def median_of(key: str) -> float | None:
        values = [ratios[key] for ratios in per_good.values() if ratios[key] is not None]
        return statistics.median(values) if values else None

    return {
        "volatility": median_of("volatility"),
        "volume": median_of("volume"),
        "stockouts": median_of("stockouts"),
        "per_good": per_good,
    }


# ---------------------------------------------------------------------------
# Coordination and cost
# ---------------------------------------------------------------------------
def coordination(log: RunLog) -> dict[str, Any]:
    """Activation and coordination metrics (design v1.0 s11.2)."""
    activations = log.count(EventType.AGENT_OUTPUT)
    wasted = log.count(EventType.ACTIVATION_WASTED)

    per_agent: dict[str, int] = {}
    for event in log.of_type(EventType.AGENT_OUTPUT):
        per_agent[event.agent_id or ""] = per_agent.get(event.agent_id or "", 0) + 1
    by_type: dict[str, int] = {}
    for event in log.of_type(EventType.MESSAGE_SENT):
        kind = str(event.payload.get("type", ""))
        by_type[kind] = by_type.get(kind, 0) + 1
    waits = [
        int(e.payload.get("waiting_ticks", 0)) for e in log.of_type(EventType.ACTIVATION_SERVED)
    ]
    tasks = log.count(EventType.DIRECTOR_TASK_ASSIGNED)
    cycle_ends = list(log.of_type(EventType.CYCLE_END))
    director_calls = log.count(EventType.DIRECTOR_CALL)
    director_errors = log.count(EventType.DIRECTOR_ERROR)

    return {
        "activations": activations,
        "wasted_activations": wasted,
        # None with no activations: no coordination to rate, not a 0% one.
        "coordination_efficiency": ((activations - wasted) / activations) if activations else None,
        # Herfindahl index of activations across agents: 1/n when spread evenly,
        # 1 when one agent took every activation (s11.2, coordinator dependence).
        "activation_concentration": (
            sum((n / activations) ** 2 for n in per_agent.values()) if activations else None
        ),
        # Ticks between the trigger and the action it led to (s11.2). In B/C
        # that is how long the request waited in the scheduler's queue; in A a
        # TASK is carried out in the tick it is assigned.
        "latency_mean": (statistics.mean(waits) if waits else (0.0 if tasks else None)),
        "messages_by_type": dict(sorted(by_type.items())),
        "director_retries": sum(int(e.payload.get("director_retries", 0)) for e in cycle_ends),
        "cycles_closed_by_cap": sum(1 for e in cycle_ends if e.payload.get("closed_by_cap")),
        "director_error_rate": (director_errors / director_calls) if director_calls else None,
        "activation_requests": log.count(EventType.ACTIVATION_REQUESTED),
        # Scheduler deferrals only: requests left over once K is used up. Those
        # skipped because the token budget ran out are logged too, with their
        # own reason, and are not a coordination outcome.
        "activations_deferred": sum(
            1
            for e in log.of_type(EventType.ACTIVATION_DEFERRED)
            if e.payload.get("reason", "k_limit") == "k_limit"
        ),
        "messages_sent": log.count(EventType.MESSAGE_SENT),
        "director_calls": log.count(EventType.DIRECTOR_CALL),
        "director_errors": log.count(EventType.DIRECTOR_ERROR),
        "invalid_outputs": log.count(EventType.AGENT_OUTPUT_INVALID),
        # Replies cut off by the output cap: the model ran out of room, which is
        # not the same failure as a malformed answer.
        "truncated_outputs": sum(
            1 for e in log.of_type(EventType.AGENT_OUTPUT_INVALID) if e.payload.get("truncated")
        )
        + sum(
            1
            for e in log.of_type(EventType.DIRECTOR_CALL)
            if e.payload.get("finish_reason") == "length"
        ),
        "llm_calls": log.count(EventType.LLM_CALL) + log.count(EventType.DIRECTOR_CALL),
        # Share of cycles that used every one of the K activations: whether K
        # actually limited this condition (design s8.1; K is calibrated so it
        # binds in B and C too, not only under the Director).
        "k_binding_rate": _k_binding_rate(cycle_ends),
        "task_types": dict(
            sorted(
                Counter(
                    str(e.payload.get("type"))
                    for e in log.of_type(EventType.DIRECTOR_TASK_ASSIGNED)
                ).items()
            )
        ),
    }


def _k_binding_rate(cycle_ends: list) -> float | None:
    counted = [
        e.payload
        for e in cycle_ends
        if e.payload.get("k")
        and "effective_activations" in e.payload
        and not e.payload.get("post_exhaustion")
    ]
    if not counted:
        return None
    return sum(1 for p in counted if p["effective_activations"] >= p["k"]) / len(counted)


#: Messages that answer a REQUEST (design s11.2).
ANSWER_TYPES = frozenset({"OFFER", "ACK", "REJECT"})


def unanswered_requests(
    log: RunLog, config: SimulationConfig, *, exhaustion_tick: int | None = None
) -> int:
    """REQUESTs with no compatible reply within D ticks (design v1.0 s11.2)."""
    window = config.metrics.unanswered_request_ticks_d
    sent = [
        (
            e.seq,
            e.tick,
            e.agent_id,
            e.payload["recipient_id"],
            e.payload["type"],
            e.payload.get("good"),
        )
        for e in log.of_type(EventType.MESSAGE_SENT)
    ]
    requests = [(q, t, s, r, g) for q, t, s, r, kind, g in sent if kind == "REQUEST"]
    # Replies can come only while agents are activated: up to the end of the
    # run, or up to the budget's exhaustion (Experiment 2), whichever is first.
    last_tick = config.time.horizon_ticks - 1
    if exhaustion_tick is not None:
        last_tick = min(last_tick, exhaustion_tick - 1)
    unanswered = 0
    for seq, tick, sender, recipient, good in requests:
        # A compatible reply is an answer from the recipient about the same
        # good (design s11.2): an OFFER, an ACK or a REJECT, never a REQUEST of
        # its own. It may come in the same tick: delivery is immediate, so a
        # reply can come from an agent activated later that cycle.
        replied = any(
            reply_seq > seq
            and reply_tick <= tick + window
            and reply_sender == recipient
            and reply_recipient == sender
            and reply_good == good
            and reply_kind in ANSWER_TYPES
            for reply_seq, reply_tick, reply_sender, reply_recipient, reply_kind, reply_good in sent
        )
        # Right-censored: a request whose reply window runs past the end of the
        # run was never given its D ticks, so it is not counted as unanswered.
        if not replied and tick + window <= last_tick:
            unanswered += 1
    return unanswered


def token_totals(log: RunLog) -> dict[str, Any]:
    """Token consumption by bucket, plus the authoritative total (design s8.4)."""
    for event in log.of_type(EventType.TOKENS_CONSUMED):
        payload = event.payload
        return {
            "by_bucket": payload["total"],
            # Kept separate because input and output are priced differently, so
            # a cost projection cannot work from the combined total.
            "input_by_bucket": payload.get("input", {}),
            "output_by_bucket": payload.get("output", {}),
            "calls_by_bucket": payload.get("calls", {}),
            "authoritative_total": payload.get("authoritative_total", payload["grand_total"]),
            "budget": payload.get("budget", {}),
        }
    empty = dict.fromkeys((b.value for b in TokenBucket), 0)
    return {
        "by_bucket": dict(empty),
        "input_by_bucket": dict(empty),
        "output_by_bucket": dict(empty),
        "calls_by_bucket": dict(empty),
        "authoritative_total": 0,
        "budget": {},
    }


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    """One run's metrics. Language-neutral; rendering happens elsewhere."""

    run_id: str
    condition: str
    scenario: str
    experiment: str
    seed: int
    target_good: str | None
    target_kind: str

    volatility: dict[str, float | None]
    volatility_phase: dict[str, float | None]
    volatility_windows: dict[str, dict[str, int]]
    stockouts: dict[str, Any]
    volume_units: dict[str, int]
    volume_value: dict[str, float]
    recovery: Duration | None
    recovery_common_horizon: Duration | None
    useful_production: dict[str, Any]
    gini: float | None
    coordination: dict[str, Any]
    unanswered_requests: int
    tokens: dict[str, Any]
    exhaustion_tick: int | None
    primary_metrics: tuple[str, ...] = field(default_factory=tuple)
    #: True for a run made with the offline mock: never evidence about a model.
    mock: bool = False
    #: Economic metrics cut at the trio's common active horizon (Experiment 2
    #: secondary analysis, design s8.4). ``None`` when no horizon was given.
    common_horizon: dict[str, Any] | None = None
    #: Units and value traded per good and phase (design s11.1).
    volume_by_good: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    #: Secondary stockout definition: no unit of the target held in the world.
    #: ``None`` with no target good, or when the log cannot support the replay.
    stockouts_inventory: dict[str, Any] | None = None
    #: The shock's propagation to the resource's main product (secondary, s11.3).
    #: ``None`` when the target is not a raw material.
    propagation: dict[str, Any] | None = None
    propagation_recovery: Duration | None = None
    #: The propagated product's recovery at the common active horizon (s8.4).
    propagation_recovery_common_horizon: Duration | None = None
    #: Secondary multi-good aggregates, normalized by base phase (s11.3).
    multi_good: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "scenario": self.scenario,
            "experiment": self.experiment,
            "seed": self.seed,
            "target_good": self.target_good,
            "target_kind": self.target_kind,
            "primary_metrics": list(self.primary_metrics),
            "volatility": self.volatility,
            "volatility_phase": self.volatility_phase,
            "volatility_windows": self.volatility_windows,
            "stockouts": self.stockouts,
            "volume_units": self.volume_units,
            "volume_value": self.volume_value,
            "recovery": None if self.recovery is None else self.recovery.to_payload(),
            "recovery_common_horizon": (
                None
                if self.recovery_common_horizon is None
                else self.recovery_common_horizon.to_payload()
            ),
            "useful_production": self.useful_production,
            "gini": self.gini,
            "coordination": self.coordination,
            "unanswered_requests": self.unanswered_requests,
            "tokens": self.tokens,
            "exhaustion_tick": self.exhaustion_tick,
            "common_horizon": self.common_horizon,
            "volume_by_good": self.volume_by_good,
            "stockouts_inventory": self.stockouts_inventory,
            "propagation": self.propagation,
            "propagation_recovery": (
                None
                if self.propagation_recovery is None
                else self.propagation_recovery.to_payload()
            ),
            "propagation_recovery_common_horizon": (
                None
                if self.propagation_recovery_common_horizon is None
                else self.propagation_recovery_common_horizon.to_payload()
            ),
            "multi_good": self.multi_good,
            "mock": self.mock,
        }
