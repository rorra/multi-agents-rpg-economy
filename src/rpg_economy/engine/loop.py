"""The macro-cycle loop -- the engine's clock.

Implements the macro-cycle structure of design v1.0 s8.1. One macro-cycle is:

1. Mechanical phase -- run ``ticks_per_cycle`` ticks. Each tick expires due
   orders and advances resource spawns, expeditions, durability and shocks.
   Triggers are evaluated after each tick, so activation requests accumulate
   across the whole phase.
2. Decision phase -- at the cycle boundary. Condition A runs the Director
   loop; B and C run the neutral scheduler. Activations execute sequentially,
   each seeing the previous one's effects, identically in all three
   conditions.
3. Close -- report effective activations.

The decision phase is a seam here, not an implementation: this module is part of
the engine and must never import the agent, LLM or orchestration layers. A run
is mechanical only unless a caller supplies a decision phase.

Everything the loop does is deterministic given the run seed. The only randomness
available is ``RunRandom``, and it is partitioned into named streams so that
conditions A, B and C draw the same world from the same seed.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Protocol

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Good, OrderSide, Resource, Role
from rpg_economy.engine.expeditions import ExpeditionSystem
from rpg_economy.engine.market import Market, MarketEvents, Order, OrderRejected, Trade
from rpg_economy.engine.production import ProductionSystem
from rpg_economy.engine.rng import RunRandom
from rpg_economy.engine.shocks import ShockSystem
from rpg_economy.engine.state import WorldState
from rpg_economy.engine.zones import ZoneSystem
from rpg_economy.logging_ import EventLogWriter, EventType

ZERO = Decimal(0)


class DecisionPhase(Protocol):
    """What runs at a macro-cycle boundary.

    Implemented by the orchestration layer: the Director loop for
    condition A, the neutral scheduler for B and C. Declared as a protocol so the
    engine depends on the shape of the collaborator, never on its module.
    """

    def __call__(self, engine: Engine) -> None: ...


class Engine:
    """Deterministic simulation of the economy."""

    def __init__(
        self,
        config: SimulationConfig,
        state: WorldState,
        log: EventLogWriter,
        *,
        random: RunRandom | None = None,
        shock_enabled: bool = True,
    ) -> None:
        """``shock_enabled=False`` runs the same world without its shock.

        Diagnostics only: the shock-efficacy check compares a seed with and
        without its shock to see whether the shock moves the economy at all.
        Experimental runs always apply their shock (design v1.0 s10).
        """
        self._config = config
        self._shock_enabled = shock_enabled
        self._state = state
        self._log = log
        self._random = random or RunRandom(config.run.seed)
        self._market = Market(
            state,
            order_expiry_ticks=config.market.order_expiry_ticks,
            visible_last_n_trades=config.market.visible_last_n_trades,
            allow_self_trade=config.market.allow_self_trade,
            events=self._market_events(),
        )
        self._tick_observers: list[Callable[[Engine], None]] = []
        self._order_observers: list[Callable[[EventType, str], None]] = []
        self._cycle_summaries: list[Callable[[], dict]] = []
        self._bulletin: dict | None = None
        self._zones = ZoneSystem(config.expeditions)
        self._expeditions = ExpeditionSystem(
            config.expeditions, self._zones, frozenset(config.economy.durability)
        )
        self._production = ProductionSystem(config.economy)
        self._shocks = ShockSystem(config, state, self._zones, self._production, self._market)

    # -- accessors ---------------------------------------------------------
    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def state(self) -> WorldState:
        return self._state

    @property
    def market(self) -> Market:
        return self._market

    @property
    def random(self) -> RunRandom:
        return self._random

    @property
    def zones(self) -> ZoneSystem:
        return self._zones

    @property
    def expeditions(self) -> ExpeditionSystem:
        return self._expeditions

    @property
    def production(self) -> ProductionSystem:
        return self._production

    @property
    def shocks(self) -> ShockSystem:
        return self._shocks

    @property
    def log(self) -> EventLogWriter:
        return self._log

    @property
    def bulletin(self) -> dict | None:
        """The most recently published global bulletin, if any."""
        return dict(self._bulletin) if self._bulletin is not None else None

    def publish_bulletin(self, summary: dict) -> None:
        """Publish the global bulletin for Condition C (design v1.0 s3.3).

        Deterministic engine output, not agent traffic. It fires no trigger,
        consumes no activation, and counts as no peer-to-peer message. That is
        structural rather than checked: the bulletin never enters the message
        bus, which is the only thing T1 reads.

        The content comes from the same generator that builds the
        Director's global view in Condition A, so the two are byte-identical.
        Its tokens are counted only when it is actually included in a prompt.
        """
        self._bulletin = dict(summary)
        self._log.write(
            EventType.BULLETIN_PUBLISHED,
            tick=self.tick,
            cycle=self.cycle,
            payload={"summary": self._bulletin},
        )

    @property
    def tick(self) -> int:
        return self._state.tick

    @property
    def cycle(self) -> int:
        return self._state.cycle

    # -- market logging ----------------------------------------------------
    def _market_events(self) -> MarketEvents:
        """Wire the market's transitions to the run log.

        Design v1.0 s5.2 requires every order and trade to be recorded with its
        counterparties, roles, quantity, price, tick and status.
        """
        return MarketEvents(
            on_order_placed=lambda o: self._log_order(EventType.ORDER_PLACED, o),
            on_order_filled=lambda o: self._log_order(EventType.ORDER_FILLED, o),
            on_order_partially_filled=lambda o: self._log_order(
                EventType.ORDER_PARTIALLY_FILLED, o
            ),
            on_order_expired=lambda o: self._log_order(EventType.ORDER_EXPIRED, o),
            on_order_cancelled=lambda o: self._log_order(EventType.ORDER_CANCELLED, o),
            on_trade=self._log_trade,
        )

    def add_tick_observer(self, observer: Callable[[Engine], None]) -> None:
        """Register a callback to run at the end of every mechanical tick.

        This is how the orchestration layer evaluates triggers per tick
        (design v1.0 s8.1) without the engine importing it. The engine calls
        back; it never reaches upward.
        """
        self._tick_observers.append(observer)

    def add_order_observer(self, observer: Callable[[EventType, str], None]) -> None:
        """Register a callback for each order status change, by agent id."""
        self._order_observers.append(observer)

    def add_cycle_summary_provider(self, provider: Callable[[], dict]) -> None:
        """Contribute fields to the cycle-end record.

        The orchestration layer reports effective activations here (design v1.0
        s8.1) rather than writing its own event, so a macro-cycle closes with
        exactly one cycle_end in the log.
        """
        self._cycle_summaries.append(provider)

    def _log_order(self, event: EventType, order: Order) -> None:
        self._log.write(
            event,
            tick=self.tick,
            cycle=self.cycle,
            agent_id=order.agent_id,
            payload={
                "order_id": order.order_id,
                "side": order.side.value,
                "good": order.good.value,
                "quantity": order.quantity,
                "remaining": order.remaining,
                "filled": order.filled,
                "limit_price": str(order.limit_price),
                "status": order.status.value,
                "placed_tick": order.placed_tick,
                "expires_at_tick": order.expires_at_tick,
            },
        )
        for observer in self._order_observers:
            observer(event, order.agent_id)

    def _log_trade(self, trade: Trade) -> None:
        self._log.write(
            EventType.TRADE_EXECUTED,
            tick=self.tick,
            cycle=self.cycle,
            payload={
                "trade_id": trade.trade_id,
                "good": trade.good.value,
                "quantity": trade.quantity,
                "price": str(trade.price),
                "value": str(trade.value),
                "buyer_id": trade.buyer_id,
                "seller_id": trade.seller_id,
                "buyer_role": trade.buyer_role.value,
                "seller_role": trade.seller_role.value,
                "buy_order_id": trade.buy_order_id,
                "sell_order_id": trade.sell_order_id,
            },
        )
        self._collect_sales_tax(trade)

    # -- convenience for callers -------------------------------------------
    def submit_order(
        self, agent_id: str, side: OrderSide, good: Good, quantity: int, price: object
    ) -> Order:
        """Place an order at the current tick.

        An adventurer cannot bid for raw materials: it cannot craft, so a raw
        material is worth to it only what it sells for. Real agents bought
        herbs and hoarded them, outbidding the alchemists who use them. Nor can
        it offer products: they are what its trips use. Real agents
        sold the potions their next trip needed, undercutting the alchemists,
        then bought them back; told in the prompt to keep them, they still sold
        nine times in ten.
        """
        account = self._state.account(agent_id)
        if account.role is Role.ADVENTURER:
            if side is OrderSide.BUY and isinstance(good, Resource):
                raise OrderRejected(
                    f"{agent_id} is an adventurer and cannot buy raw materials; "
                    "it can only sell them"
                )
            if side is OrderSide.SELL and not isinstance(good, Resource):
                raise OrderRejected(
                    f"{agent_id} is an adventurer and may not offer products; it only buys them"
                )
        return self._market.place(
            agent_id=agent_id,
            side=side,
            good=good,
            quantity=quantity,
            price=price if isinstance(price, Decimal) else Decimal(str(price)),
            tick=self.tick,
        )

    # -- agent-facing operations -------------------------------------------
    def log_event(
        self, event: EventType, *, agent_id: str | None = None, payload: dict | None = None
    ) -> None:
        """Record an event at the current tick and cycle.

        Exposed so the agent and orchestration layers can log against the run's
        clock without holding the writer or tracking time themselves.
        """
        self._log.write(
            event, tick=self.tick, cycle=self.cycle, agent_id=agent_id, payload=payload or {}
        )

    def expedition_odds(self, agent_id: str) -> dict[str, Decimal]:
        """An adventurer's success probability per zone if it set out now; empty for others."""
        account = self._state.account(agent_id)
        if account.role is not Role.ADVENTURER:
            return {}
        needed = self._expeditions.supplies_for(account, self._shocks.consumption_multiplier)
        return self._expeditions.odds_now(account, needed, self.tick)

    def start_expedition(self, agent_id: str, zone_id: str) -> None:
        """Send an adventurer to a zone. Resolution happens on a later tick."""
        account = self._state.account(agent_id)
        self._expeditions.check_start(account, zone_id, self.tick)
        for product in self._expeditions.to_equip(account):
            self.equip(agent_id, product)
        needed = self._expeditions.supplies_for(account, self._shocks.consumption_multiplier)
        state, consumed = self._expeditions.start(account, zone_id, self.tick, needed)
        for product, quantity in consumed.items():
            self._log.write(
                EventType.CONSUMPTION,
                tick=self.tick,
                cycle=self.cycle,
                agent_id=agent_id,
                payload={"product": product.value, "quantity": quantity, "use": "expedition"},
            )
        self._log.write(
            EventType.EXPEDITION_STARTED,
            tick=self.tick,
            cycle=self.cycle,
            agent_id=agent_id,
            payload={
                "zone_id": state.zone_id,
                "risk": state.risk.value,
                "resolves_at_tick": state.resolves_at_tick,
                "supplies_needed": {p.value: q for p, q in needed.items()},
                # What each carried item adds, fixed now for the whole trip.
                "gear_bonus": {
                    p.value: str(b.quantize(Decimal("0.0001"))) for p, b in state.gear_bonus.items()
                },
                "success_probability": str(
                    self._expeditions.success_probability(
                        account, state.risk, gear_bonus=state.gear_bonus, crowding=state.crowding
                    )
                ),
                "crowding": str(state.crowding),
            },
        )
