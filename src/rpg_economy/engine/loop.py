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

import random
from collections.abc import Callable
from decimal import Decimal
from typing import Protocol

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import MONEY_QUANTUM, Good, OrderSide, Product, Resource, Role
from rpg_economy.engine.expeditions import ExpeditionResult, ExpeditionSystem
from rpg_economy.engine.market import Market, MarketEvents, Order, OrderRejected, Trade
from rpg_economy.engine.production import ProductionResult, ProductionSystem, unit_cost
from rpg_economy.engine.rng import STREAM_EXPEDITION_ORDER, RunRandom, derive_seed
from rpg_economy.engine.shocks import ShockState, ShockSystem
from rpg_economy.engine.state import AgentAccount, WorldState
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

    def zone_traffic(self) -> dict[str, int]:
        """Expeditions set out for each zone over the crowding window: public."""
        return self._expeditions.traffic(self.tick)

    def zone_crowding(self) -> dict[str, Decimal]:
        """The success each zone's crowding adds (negative: takes) for a trip set out now."""
        return {z: self._expeditions.crowding(z, self.tick) for z in self._zones.zone_ids}

    def produce(self, agent_id: str, product: Product, quantity: int) -> ProductionResult:
        """Craft, consuming recipe inputs against this cycle's capacity."""
        account = self._state.account(agent_id)
        result = self._production.produce(account, product, quantity)
        self._log.write(
            EventType.PRODUCTION_COMPLETED,
            tick=self.tick,
            cycle=self.cycle,
            agent_id=agent_id,
            payload={
                "product": result.product.value,
                "quantity": result.quantity,
                "consumed": {r.value: q for r, q in sorted(result.consumed.items())},
                "capacity_remaining": self._production.capacity_of(account),
            },
        )
        return result

    def equip(self, agent_id: str, product: Product) -> None:
        """Move one unit of ``product`` from inventory into the equipment slot.

        Only an adventurer equips, and only in town: gear carried on an
        expedition is fixed when it sets out. Trading while away stays allowed.
        A product that is not equipment is refused as such before its stock is
        checked. Any subclass may equip any equipment: gear outside its own
        subclass's kit is allowed, it just does not serve its expeditions.
        """
        account = self._state.account(agent_id)
        if account.role is not Role.ADVENTURER:
            raise ValueError(f"{agent_id} is a {account.role}, not an adventurer")
        if account.expedition is not None:
            raise ValueError(f"{agent_id} is already away on an expedition and cannot equip")
        durability = self._config.economy.durability.get(product)
        if durability is None:
            raise ValueError(f"{product} is not equipment")
        if account.available_quantity(product) < 1:
            raise ValueError(f"{agent_id} has no unreserved {product} to equip")
        account.consume(product, 1)
        account.equipment[product] = account.durability_of(product) + durability
        self._log.write(
            EventType.EQUIPMENT_EQUIPPED,
            tick=self.tick,
            cycle=self.cycle,
            agent_id=agent_id,
            payload={"product": product.value, "durability": account.durability_of(product)},
        )

    # -- the loop ----------------------------------------------------------
    def mechanical_tick(self) -> None:
        """Do one tick's mechanical work, without advancing the clock.

        The order is fixed and matters. Shocks are checked first so that a shock
        applied at tick t governs that tick's appearance rather than the next
        one's; resources then appear; expeditions due this tick resolve against
        the pools just replenished (equipment wears on them); and finally orders
        expire.

        No LLM and no agent decision is reachable from here: everything in the
        mechanical phase is engine-determined, which is what lets a seed-trio
        share the same world across conditions A, B and C.
        """
        self._apply_due_shocks()
        self._spawn_resources()
        self._resolve_due_expeditions()
        self._market.expire_due(self.tick)
        self._log.write(EventType.TICK, tick=self.tick, cycle=self.cycle)
        for observer in self._tick_observers:
            observer(self)

    def _apply_due_shocks(self) -> None:
        if not self._shock_enabled:
            return
        if self._shocks.is_due(self.tick):
            shock = self._shocks.apply(self.tick)
            self._log_shock(EventType.SHOCK_APPLIED, shock)
            good = shock.details.get("target_resource")
            for agent_id, (quantity, paid) in shock.confiscated.items():
                self._log.write(
                    EventType.GOODS_CONFISCATED,
                    tick=self.tick,
                    cycle=self.cycle,
                    agent_id=agent_id,
                    payload={"good": good, "quantity": quantity, "gold": str(paid)},
                )
        elif self._shocks.is_reversion_due(self.tick):
            self._log_shock(EventType.SHOCK_REVERTED, self._shocks.revert(self.tick))

    def _log_shock(self, event: EventType, shock: ShockState) -> None:
        self._log.write(
            event,
            tick=self.tick,
            cycle=self.cycle,
            payload={
                "scenario": shock.scenario.value,
                "applied_tick": shock.applied_tick,
                "reverted_tick": shock.reverted_tick,
                **shock.details,
            },
        )

    def _spawn_resources(self) -> None:
        """Replenish the zone pools, and log what appeared and what is there now.

        Logging only: nothing here reads or changes randomness, so the record
        makes the zone stock reconstructable from the log without altering the
        run (design v1.0 s15). Agents never see it; it is not part of any
        observation, the Director's global view or the bulletin.
        """
        before = self._zone_stock()
        self._zones.accrue()
        after = self._zone_stock()
        spawned = {
            zone: {
                resource: units - before[zone].get(resource, 0)
                for resource, units in stock.items()
                if units != before[zone].get(resource, 0)
            }
            for zone, stock in after.items()
        }
        self._log.write(
            EventType.RESOURCES_SPAWNED,
            tick=self.tick,
            cycle=self.cycle,
            payload={
                "spawned": {zone: added for zone, added in spawned.items() if added},
                "stock": after,
                **(
                    {"gold": {z.zone_id: str(z.gold) for z in self._zones.zones()}}
                    if any(z.gold_rate is not None for z in self._zones.zones())
                    else {}
                ),
            },
        )

    def _zone_stock(self) -> dict[str, dict[str, int]]:
        return {
            zone.zone_id: {
                resource.value: zone.stock(resource)
                for resource in sorted(zone.spawn_per_tick, key=lambda r: r.value)
            }
            for zone in self._zones.zones()
        }

    def _resolve_due_expeditions(self) -> None:
        """Resolve every expedition due this tick.

        Expeditions due together draw from shared zone pools, so the order
        matters: it is drawn from the run seed afresh each tick, not the agent
        ids' alphabetical order, which would always serve the same subclass
        first. Each agent gets its own key from the seed, tick and id, so any
        two agents due together draw in the same order in A, B and C, whoever
        else is due.
        """
        due = sorted(
            agent_id
            for agent_id in self._state.agent_ids
            if (expedition := self._state.account(agent_id).expedition) is not None
            and expedition.resolves_at_tick <= self.tick
        )
        order = sorted(
            due,
            key=lambda agent_id: derive_seed(
                self._random.run_seed, f"{STREAM_EXPEDITION_ORDER}:{self.tick}:{agent_id}"
            ),
        )
        # Grouped by zone, each group in that order: those due together in a
        # zone take turns drawing from its pool, so a small pool is split
        # among them rather than emptied by the first to resolve.
        groups: dict[str, list[tuple[AgentAccount, random.Random]]] = {}
        for agent_id in order:
            account = self._state.account(agent_id)
            expedition = account.expedition
            assert expedition is not None
            rng = self._random.expedition(agent_id, expedition.zone_id, expedition.started_tick)
            groups.setdefault(expedition.zone_id, []).append((account, rng))
        for group in groups.values():
            for result in self._expeditions.resolve_group(group, self.tick):
                self._log_expedition(result)

    def _log_expedition(self, result: ExpeditionResult) -> None:
        self._log.write(
            EventType.EXPEDITION_RESOLVED,
            tick=self.tick,
            cycle=self.cycle,
            agent_id=result.agent_id,
            payload={
                "zone_id": result.zone_id,
                "risk": result.risk.value,
                "outcome": result.outcome.value,
                "success_probability": str(result.success_probability),
                "yielded": {r.value: q for r, q in sorted(result.yielded.items())},
                "total_units": result.total_units,
                "gold": str(result.gold),
                "gold_due": str(result.gold_due),
                "recovering_until_tick": result.recovering_until_tick,
            },
        )
        for product in result.equipment_broken:
            self._log.write(
                EventType.EQUIPMENT_BROKEN,
                tick=self.tick,
                cycle=self.cycle,
                agent_id=result.agent_id,
                payload={"product": product.value, "cause": "expedition_wear"},
            )

    def _charge_storage(self) -> None:
        """Artisans pay storage per unit of goods held (a gold sink).

        Storage is per unit held, offered on the book or not, so holding stock at a
        price nobody pays costs something. Paid from available gold, never into
        debt, and never below the charge floor: gold backing a resting bid counts
        toward the floor, since it is already on its way to goods, while gold owed
        as wealth-tax arrears does not, since it is not the agent's to keep, and
        neither does gold frozen by a liquidity shock, since the agent cannot use
        it: during a freeze the floor protects gold the agent can still use. That
        excludes the part of a pre-freeze reservation a withdrawal would freeze
        (``_frozen_on_release``), as the wealth-tax base does.
        """
        economy = self._config.economy
        if not economy.storage_cost_per_unit:
            return
        for agent_id in self._state.agent_ids:
            account = self._state.account(agent_id)
            floor = self._charge_floor(account)
            units = sum(account.inventory.values())
            due = economy.storage_cost_per_unit.get(account.role, ZERO) * units
            # The floor is measured on gold the agent can use, net of what a
            # pre-freeze bid would leave frozen on release: otherwise storage
            # could take gold the freeze is still owed, as the wealth tax does not.
            usable = (
                account.gold
                - account.tax_arrears
                - account.frozen_gold
                - self._frozen_on_release(account)
            )
            paid = min(due, max(ZERO, usable - floor), account.available_gold)
            if paid <= ZERO:
                continue
            account.gold -= paid
            self._log.write(
                EventType.STORAGE_PAID,
                tick=self.tick,
                cycle=self.cycle,
                agent_id=agent_id,
                payload={"amount": str(paid), "units": units},
            )

    def _charge_floor(self, account: AgentAccount) -> Decimal:
        """Gold that storage and the profit tax never take an agent below.

        With ``floor_at_initial_gold``, the role's initial gold, so taxes on thin
        margins never leave an artisan with a fraction of what it started with.
        """
        economy = self._config.economy
        floor = economy.charge_floor
        if economy.floor_at_initial_gold:
            floor = max(floor, economy.initial_gold.get(account.role.value, ZERO))
        return floor

    def _collect_wealth_tax(self) -> None:
        """Every ``wealth_every_ticks``, each agent pays on its gold above the exemption.

        The base is the agent's gold less its frozen gold (not the agent's to
        use, so not taxed), less the arrears it already owes (taxed once, when
        assessed), less the exempt minimum, so an agent at or below it pays
        nothing and one holding more pays more. Gold reserved behind a bid is
        still the agent's and stays in the base -- except, while a liquidity
        freeze is active, the part of a pre-freeze reservation that would be
        frozen if its bid were withdrawn (``freezable_reserved`` times the
        freeze rate), so the tax does not depend on whether a bid happened to
        be resting at the shock tick.

        The tax due is paid from available gold -- neither reserved, frozen nor
        already owed -- and never into debt. What cannot be paid now is not
        forgiven: it is added to the account's ``tax_arrears``. Otherwise an
        agent could avoid the tax by parking its gold in a bid over the
        collection tick.

        Arrears are a lien: they are left out of available gold, so gold freed
        later (a bid expired or was cancelled, a sale came in) cannot be re-bid
        or spent ahead of them. They are collected first, at the start of every
        cycle, before any new tax is assessed.

        Called at the start of a cycle, it collects once for every multiple of
        the period that falls inside the cycle's span (tick 0 excluded), so the
        number of collections over a run does not depend on the cycle length.
        """
        # With ticks_per_cycle > 1, every collection due within the cycle runs
        # here, at its start -- before the cycle's mechanical ticks, and so
        # before a shock that lands mid-cycle. A liquidity freeze applied at a
        # later tick of the cycle does not shrink these collections' base.
        self._collect_tax_arrears()
        taxes = self._config.economy.taxes
        if taxes.wealth_rate <= ZERO:
            return
        period = taxes.wealth_every_ticks
        first = max(self.tick, 1)
        last = self.tick + self._config.time.ticks_per_cycle - 1
        for _ in range(max(0, last // period - (first - 1) // period)):
            self._collect_wealth_tax_once()

    def _collect_tax_arrears(self) -> None:
        """Pay what each agent owes from earlier collections, out of its free gold.

        Free gold is gold neither reserved nor frozen: the lien never takes
        either, and never makes gold negative. Paying lowers gold and arrears
        together, so available gold is unchanged. Logged as a wealth TAX_PAID
        flagged ``arrears``; ``amount`` is the gold paid and ``owed`` what is
        still outstanding afterwards.
        """
        for agent_id in self._state.agent_ids:
            account = self._state.account(agent_id)
            if account.tax_arrears <= ZERO:
                continue
            paid = min(account.tax_arrears, account.free_gold)
            if paid <= ZERO:
                continue
            account.gold -= paid
            account.tax_arrears -= paid
            self._log.write(
                EventType.TAX_PAID,
                tick=self.tick,
                cycle=self.cycle,
                agent_id=agent_id,
                payload={
                    "tax": "wealth",
                    "amount": str(paid),
                    "arrears": True,
                    "owed": str(account.tax_arrears),
                },
            )

    def _collect_wealth_tax_once(self) -> None:
        """Assess one collection; log it for every agent that owes something.

        ``amount`` is the gold actually paid, ``due`` what was assessed, and
        ``owed`` the account's total arrears after the collection.
        """
        taxes = self._config.economy.taxes
        for agent_id in self._state.agent_ids:
            account = self._state.account(agent_id)
            taxable = (
                account.gold
                - account.frozen_gold
                - account.tax_arrears
                - self._frozen_on_release(account)
                - taxes.wealth_exemption
            )
            if taxable <= ZERO:
                continue
            due = (taxable * taxes.wealth_rate).quantize(MONEY_QUANTUM)
            if due <= ZERO:
                continue
            paid = min(due, account.available_gold)
            account.gold -= paid
            account.tax_arrears += due - paid
            self._log.write(
                EventType.TAX_PAID,
                tick=self.tick,
                cycle=self.cycle,
                agent_id=agent_id,
                payload={
                    "tax": "wealth",
                    "amount": str(paid),
                    "taxable": str(taxable),
                    "due": str(due),
                    "owed": str(account.tax_arrears),
                },
            )

    @staticmethod
    def _frozen_on_release(account: AgentAccount) -> Decimal:
        """Gold a liquidity freeze would take if every pre-freeze bid were withdrawn.

        Zero outside a freeze. Left out of the wealth-tax base and of the gold
        storage may charge above the floor, as frozen gold is.
        """
        if account.freeze_rate <= ZERO or account.freezable_reserved <= ZERO:
            return ZERO
        return (account.freezable_reserved * account.freeze_rate).quantize(MONEY_QUANTUM)

    def _collect_sales_tax(self, trade: Trade) -> None:
        """The seller of a finished product pays the profit tax from the proceeds.

        The profit is the sale's value less the product's input cost at the last
        traded prices (``unit_cost``, the figure the artisan's prompt shows) per
        unit sold; a sale at or below that cost pays nothing. Paid from free
        gold -- neither reserved nor frozen -- even when it is under the arrears
        lien: the tax falls due with the sale, and capping it by available gold
        would forgive it to agents in arrears. The arrears stay owed.
        """
        if not isinstance(trade.good, Product):
            return
        rate = self._config.economy.taxes.profit_rate
        if rate <= ZERO:
            return
        cost = unit_cost(self._config.economy, self._last_price, trade.good) * trade.quantity
        profit = trade.value - cost
        if profit <= ZERO:
            return
        seller = self._state.account(trade.seller_id)
        paid = min((profit * rate).quantize(MONEY_QUANTUM), seller.free_gold)
        if self._config.economy.floor_at_initial_gold:
            # Measured as storage measures it, after the sale's proceeds are in:
            # a sale that only brings the seller back toward its initial gold
            # pays nothing.
            usable = (
                seller.gold
                - seller.tax_arrears
                - seller.frozen_gold
                - self._frozen_on_release(seller)
            )
            paid = min(paid, max(ZERO, usable - self._charge_floor(seller)))
        if paid <= ZERO:
            return
        seller.gold -= paid
        self._log.write(
            EventType.TAX_PAID,
            tick=self.tick,
            cycle=self.cycle,
            agent_id=trade.seller_id,
            payload={
                "tax": "sales",
                "amount": str(paid),
                "good": trade.good.value,
                "trade_id": trade.trade_id,
                "profit": str(profit),
            },
        )

    def _last_price(self, good: Good) -> Decimal | None:
        last = self._market.last_trades(good, 1)
        return last[-1].price if last else None

    def run_cycle(self, decision_phase: DecisionPhase | None = None) -> None:
        """Run one complete macro-cycle.

        Macro-cycle ``c`` spans ticks ``[c * ticks_per_cycle, (c+1) *
        ticks_per_cycle - 1]``, and its decision phase happens at the last
        tick of that span. Orders an activation places are therefore stamped
        with a tick inside the cycle that placed them, and the next cycle's
        expiry sweep runs at a strictly later tick -- so an order's lifetime is
        the configured number of ticks, not one less.
        """
        ticks_per_cycle = self._config.time.ticks_per_cycle
        self._log.write(
            EventType.CYCLE_START,
            tick=self.tick,
            cycle=self.cycle,
            payload={"ticks_per_cycle": ticks_per_cycle},
        )

        self._charge_storage()
        self._collect_wealth_tax()
        for index in range(ticks_per_cycle):
            self.mechanical_tick()
            if index < ticks_per_cycle - 1:
                self._state.tick += 1

        if decision_phase is not None:
            decision_phase(self)

        summary: dict = {"trades_so_far": len(self._market.trades)}
        for provider in self._cycle_summaries:
            summary.update(provider())
        self._log.write(EventType.CYCLE_END, tick=self.tick, cycle=self.cycle, payload=summary)
        self._reset_cycle_capacity()
        self._state.tick += 1
        self._state.cycle += 1

    def _reset_cycle_capacity(self) -> None:
        """Production capacity accrues at the close of each macro-cycle."""
        for account in self._state.accounts.values():
            self._production.refresh(account)

    def run(self, decision_phase: DecisionPhase | None = None) -> None:
        """Run to the planned horizon.

        The horizon is always reached. In Experiment 2 that still holds after the
        token budget is exhausted: LLM calls cease, mechanical ticks do not
        (design v1.0 s8.4).
        """
        for _ in range(self._config.time.total_cycles):
            self.run_cycle(decision_phase)
        self.log_run_end()

    def log_run_end(self) -> None:
        """Close the log with the run summary and every agent's final holdings.

        Exposed separately because the runner drives macro-cycles itself rather
        than through :meth:`run`; both paths must close the log the same way, or
        a metric sourced from run_end silently reads nothing.
        """
        self._log.write(
            EventType.RUN_END,
            tick=self.tick,
            cycle=self.cycle,
            payload={
                "ticks": self.tick,
                "cycles": self.cycle,
                "trades": len(self._market.trades),
                # Final holdings belong in the log, not only in memory: s15
                # requires a run be reconstructable from its log alone, and
                # Gini is computed from these (design v1.0 s11.1, s11.3).
                "holdings": {
                    agent_id: {
                        "gold": str(account.gold),
                        # Gold is gross: arrears are owed out of it, and the
                        # reserved and frozen parts are still the agent's.
                        "tax_arrears": str(account.tax_arrears),
                        "reserved_gold": str(account.reserved_gold),
                        "frozen_gold": str(account.frozen_gold),
                        "role": account.role.value,
                        "inventory": {
                            good.value: units
                            for good, units in sorted(account.inventory.items())
                            if units
                        },
                    }
                    for agent_id, account in sorted(self._state.accounts.items())
                },
            },
        )
