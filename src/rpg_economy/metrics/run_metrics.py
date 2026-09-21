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
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import OrderSide
from rpg_economy.logging_ import Event, EventType, read_events, read_metadata
from rpg_economy.metrics.targets import PrimaryTarget, TargetKind

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
