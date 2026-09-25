"""Economic sanity check for a completed run.

A diagnostic, not part of the experiment's metric set (see ``rpg_economy.metrics``).
It checks whether the economy behaves sensibly: do prices respond to scarcity,
does the market stay active in every phase, and does anyone go bankrupt or
accumulate without bound?

Everything except the final holdings is reconstructed from the event log,
which doubles as a check on design v1.0 s15: a run must be fully reconstructable
from its log. The order book is replayed from its recorded transitions to derive
stockouts, so if the log were missing an order event the stockout series would be
wrong.

This module computes; it renders nothing. All strings are applied at render time
(IMPLEMENTATION.md section 2).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Good, OrderSide, Role
from rpg_economy.engine.state import WorldState
from rpg_economy.logging_ import Event, EventType, read_events
from rpg_economy.metrics.run_metrics import net_gold

ZERO = Decimal(0)

#: The three phases of a run, as tick half-open intervals.
PHASE_BASE = "base"
PHASE_SHOCK = "shock"
PHASE_RECOVERY = "recovery"
PHASES = (PHASE_BASE, PHASE_SHOCK, PHASE_RECOVERY)


@dataclass
class PhaseStats:
    """Market activity within one phase."""

    phase: str
    first_tick: int
    last_tick: int
    trades: int = 0
    target_trades: int = 0
    target_units: int = 0
    target_prices: list[Decimal] = field(default_factory=list)
    traded_value: Decimal = ZERO

    @property
    def ticks(self) -> int:
        return self.last_tick - self.first_tick

    @property
    def mean_target_price(self) -> Decimal | None:
        if not self.target_prices:
            return None
        return sum(self.target_prices, ZERO) / len(self.target_prices)

    @property
    def min_target_price(self) -> Decimal | None:
        return min(self.target_prices) if self.target_prices else None

    @property
    def max_target_price(self) -> Decimal | None:
        return max(self.target_prices) if self.target_prices else None

    @property
    def trades_per_tick(self) -> Decimal:
        return Decimal(self.trades) / self.ticks if self.ticks else ZERO


@dataclass
class StockoutStats:
    """Ticks during which a good had no resting ask (design s11.1)."""

    good: str
    episodes: int = 0
    total_ticks: int = 0
    longest: int = 0
    by_phase: dict[str, int] = field(default_factory=dict)


@dataclass
class RoleHoldings:
    """Aggregate holdings of one role, at the start and at the end of a run."""

    role: str
    agents: int
    gold_start: Decimal
    gold_end: Decimal
    units_start: int
    units_end: int

    @property
    def gold_change(self) -> Decimal:
        return self.gold_end - self.gold_start

    @property
    def bankrupt(self) -> int:
        return 0  # populated by the caller when per-agent detail is needed


@dataclass
class EconomyCheck:
    """The full diagnostic."""

    #: ``None`` when the scenario's target is total traded value, not a good.
    target_good: str | None
    phases: list[PhaseStats]
    stockouts: list[StockoutStats]
    roles: list[RoleHoldings]
    total_gold_start: Decimal
    total_gold_end: Decimal
    #: Gold that entered as expedition loot and confiscation pay and left as
    #: taxes and storage: the only places gold is created or destroyed.
    gold_looted: Decimal
    gold_confiscation_paid: Decimal
    gold_taxed: Decimal
    gold_storage: Decimal
    gold_reconciles: bool
    #: Agents whose gold net of tax arrears cannot buy the cheapest good.
    bankrupt_agents: list[str]
    #: The richest agent's share of the economy's gold net of tax arrears.
    richest_share: Decimal
    expeditions: int
    production: int
    target_gathered: dict[str, int]

    def phase(self, name: str) -> PhaseStats:
        return next(p for p in self.phases if p.phase == name)


def phase_of(tick: int, config: SimulationConfig) -> str:
    if tick < config.phases.shock_tick:
        return PHASE_BASE
    if tick < config.phases.reversion_tick:
        return PHASE_SHOCK
    return PHASE_RECOVERY


def _phase_bounds(config: SimulationConfig) -> dict[str, tuple[int, int]]:
    return {
        PHASE_BASE: (0, config.phases.shock_tick),
        PHASE_SHOCK: (config.phases.shock_tick, config.phases.reversion_tick),
        PHASE_RECOVERY: (config.phases.reversion_tick, config.time.horizon_ticks),
    }


def analyse(
    log_path: Path | str,
    config: SimulationConfig,
    final_state: WorldState,
    initial_state: WorldState,
    target: Good | None,
) -> EconomyCheck:
    """Build the diagnostic for a completed run.

    ``target`` is ``None`` when the scenario has no target good
    (liquidity_loss): the per-good sections are then empty and the traded
    value per phase carries the diagnostic.
    """
    events = list(read_events(log_path))
    bounds = _phase_bounds(config)

    phases = {
        name: PhaseStats(phase=name, first_tick=lo, last_tick=hi)
        for name, (lo, hi) in bounds.items()
    }
    stockouts = _replay_stockouts(events, config, target) if target is not None else []
    expeditions = 0
    production = 0
    gathered: dict[str, int] = {}
    gold_flow: dict[str, Decimal] = {}
    looted = ZERO
    confiscation_paid = ZERO
    taxed = ZERO
    storage = ZERO

    for event in events:
        name = phase_of(event.tick, config)
        match event.type:
            case EventType.TRADE_EXECUTED:
                stats = phases[name]
                stats.trades += 1
                value = Decimal(event.payload["value"])
                stats.traded_value += value
                gold_flow[event.payload["buyer_id"]] = (
                    gold_flow.get(event.payload["buyer_id"], ZERO) - value
                )
                gold_flow[event.payload["seller_id"]] = (
                    gold_flow.get(event.payload["seller_id"], ZERO) + value
                )
                if target is not None and event.payload["good"] == target.value:
                    stats.target_trades += 1
                    stats.target_units += event.payload["quantity"]
                    stats.target_prices.append(Decimal(event.payload["price"]))
            case EventType.EXPEDITION_RESOLVED:
                expeditions += 1
                loot = Decimal(event.payload.get("gold", "0"))
                looted += loot
                gold_flow[event.agent_id] = gold_flow.get(event.agent_id, ZERO) + loot
                for good, quantity in event.payload["yielded"].items():
                    gathered[good] = gathered.get(good, 0) + quantity
            case EventType.TAX_PAID | EventType.STORAGE_PAID:
                paid = Decimal(event.payload["amount"])
                if event.type is EventType.TAX_PAID:
                    taxed += paid
                else:
                    storage += paid
                gold_flow[event.agent_id] = gold_flow.get(event.agent_id, ZERO) - paid
            case EventType.GOODS_CONFISCATED:
                paid = Decimal(event.payload["gold"])
                confiscation_paid += paid
                gold_flow[event.agent_id] = gold_flow.get(event.agent_id, ZERO) + paid
            case EventType.PRODUCTION_COMPLETED:
                production += 1
            case _:
                pass

    roles = _role_holdings(initial_state, final_state)
    # Broke rather than literally at zero: an agent that cannot afford the
    # cheapest good in the economy can no longer take part in it. Judged on
    # gold net of tax arrears: arrears are a lien, collected from the first
    # gold that becomes available, so that gold is not the agent's to spend.
    cheapest = min(Decimal(p) for p in config.economy.initial_reference_price.values())
    net = {
        agent_id: net_gold(
            {
                "gold": final_state.account(agent_id).gold,
                "tax_arrears": final_state.account(agent_id).tax_arrears,
            }
        )
        for agent_id in final_state.agent_ids
    }
    bankrupt = [agent_id for agent_id in final_state.agent_ids if net[agent_id] < cheapest]

    total_end = final_state.total_gold()
    reconciled = all(
        initial_state.account(agent_id).gold + gold_flow.get(agent_id, ZERO)
        == final_state.account(agent_id).gold
        for agent_id in final_state.agent_ids
    )

    # The richest agent's share of the economy's net gold; the lien of an agent
    # owing more than it holds takes nothing from anyone else's share.
    holdings = sorted((max(ZERO, value) for value in net.values()), reverse=True)
    total_net = sum(holdings, ZERO)
    richest_share = (holdings[0] / total_net) if total_net > ZERO else ZERO

    return EconomyCheck(
        target_good=None if target is None else target.value,
        phases=[phases[name] for name in PHASES],
        stockouts=stockouts,
        roles=roles,
        total_gold_start=initial_state.total_gold(),
        total_gold_end=total_end,
        gold_looted=looted,
        gold_confiscation_paid=confiscation_paid,
        gold_taxed=taxed,
        gold_storage=storage,
        gold_reconciles=reconciled,
        bankrupt_agents=bankrupt,
        richest_share=richest_share,
        expeditions=expeditions,
        production=production,
        target_gathered={name: _gathered_in_phase(events, config, name, target) for name in PHASES},
    )


def _gathered_in_phase(
    events: Iterable[Event], config: SimulationConfig, phase: str, target: Good | None
) -> int:
    if target is None:
        return 0
    return sum(
        event.payload["yielded"].get(target.value, 0)
        for event in events
        if event.type is EventType.EXPEDITION_RESOLVED and phase_of(event.tick, config) == phase
    )


def _replay_stockouts(
    events: Iterable[Event], config: SimulationConfig, target: Good
) -> list[StockoutStats]:
    """Replay the order book's ask side to find ticks with no offer.

    A stockout holds at a tick when no ask rests on the book (design s11.1).
    The book is sampled at each ``tick`` event, which is the end of that
    tick's mechanical phase -- a fixed point in every cycle, before the cycle's
    activations place new orders.
    """
    open_asks: set[int] = set()
    stats = StockoutStats(good=target.value)
    in_episode = False
    current = 0

    for event in events:
        payload = event.payload
        if event.type is EventType.ORDER_PLACED:
            if payload["good"] == target.value and payload["side"] == OrderSide.SELL.value:
                open_asks.add(payload["order_id"])
        elif event.type in {
            EventType.ORDER_FILLED,
            EventType.ORDER_EXPIRED,
            EventType.ORDER_CANCELLED,
        }:
            open_asks.discard(payload["order_id"])
        elif event.type is EventType.TICK:
            phase = phase_of(event.tick, config)
            if not open_asks:
                stats.total_ticks += 1
                stats.by_phase[phase] = stats.by_phase.get(phase, 0) + 1
                current += 1
                if not in_episode:
                    stats.episodes += 1
                    in_episode = True
            else:
                stats.longest = max(stats.longest, current)
                current = 0
                in_episode = False

    stats.longest = max(stats.longest, current)
    return [stats]


def _role_holdings(initial: WorldState, final: WorldState) -> list[RoleHoldings]:
    rows: list[RoleHoldings] = []
    for role in Role:
        agents = final.agents_with_role(role)
        if not agents:
            continue
        rows.append(
            RoleHoldings(
                role=role.value,
                agents=len(agents),
                gold_start=sum((initial.account(a).gold for a in agents), ZERO),
                gold_end=sum((final.account(a).gold for a in agents), ZERO),
                units_start=sum(sum(initial.account(a).inventory.values()) for a in agents),
                units_end=sum(sum(final.account(a).inventory.values()) for a in agents),
            )
        )
    return rows


def target_good_for(config: SimulationConfig) -> Good | None:
    """The scenario's pre-registered primary target good (design v1.0 s11.3).

    The same table the metrics use, so the diagnostic can never check a
    different good than the analysis measures. ``None`` for liquidity_loss,
    whose target is total traded value rather than a good.
    """
    from rpg_economy.metrics.targets import primary_target

    return primary_target(config).good
