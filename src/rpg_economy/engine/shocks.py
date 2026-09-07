"""Controlled perturbations (design v1.0 s10).

One run contains a base phase, exactly one main shock, and a recovery phase.
Every shock has an explicit finite duration: it is applied at
``phases.shock_tick`` and reverted at ``phases.reversion_tick``, and that
reversion tick is the origin from which recovery time is measured
(design v1.0 s10, s11.1).

Timing and magnitude derive entirely from configuration, so for a given seed and
scenario conditions A, B and C necessarily receive the same shock at the same
tick. The Director cannot prevent or modify a shock (s10): nothing in this
module is reachable from any decision path.

``liquidity_loss`` freezes gold rather than destroying it. The design calls for
reducing purchasing capacity, which is a liquidity constraint: the shock models
illiquidity, not a loss of wealth, so the gold stays the agent's and comes back
at reversion. (Gold does leave the economy elsewhere, through taxes and storage;
that is a sink by design, not what this shock represents.) At the shock tick,
each affected agent has the configured fraction of its available gold (free
gold net of tax arrears owed) frozen, and the same fraction of the gold then
reserved behind its bids, net of the arrears free gold cannot cover, is frozen
if that reservation is later released (the bid expires or is cancelled). Gold
reserved by bids placed after the shock is never frozen on release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from rpg_economy.config.models import LiquidityLossConfig, SimulationConfig
from rpg_economy.domain import OrderSide, Resource, Role, Scenario
from rpg_economy.engine.market import Market
from rpg_economy.engine.production import ProductionSystem
from rpg_economy.engine.state import WorldState
from rpg_economy.engine.zones import ZoneSystem

ZERO = Decimal(0)


class ShockNotImplemented(Exception):
    """Raised for a scenario with no implementation.

    ``noisy_info`` is admissible only if defined before the final
    runs (design v1.0 s10); until then it must fail loudly rather than run as a
    silently shock-free control.
    """


@dataclass
class ShockState:
    """What a shock did, so that reversion can undo exactly that."""

    scenario: Scenario
    applied_tick: int
    reverted_tick: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    #: Gold frozen at the shock tick, per affected agent (zero included).
    frozen_by_agent: dict[str, Decimal] = field(default_factory=dict)
    #: Every agent the shock applies to, so reversion lifts the freeze on all
    #: of them -- including one with no available gold at the shock, whose frozen
    #: gold can only come later, from a pre-shock bid released during it.
    affected_agents: list[str] = field(default_factory=list)
    #: Agent -> (units taken, gold paid) by a resource_scarcity shock's held_keep.
    confiscated: dict[str, tuple[int, Decimal]] = field(default_factory=dict)
    #: Agent -> units its sell orders of the target offered before and after the
    #: confiscation cut them, for the agents whose orders it touched.
    orders_cut: dict[str, tuple[int, int]] = field(default_factory=dict)


class ShockSystem:
    """Applies and reverts the run's single main shock."""

    def __init__(
        self,
        config: SimulationConfig,
        state: WorldState,
        zones: ZoneSystem,
        production: ProductionSystem,
        market: Market | None = None,
    ) -> None:
        # Refused when the engine is built, not when the shock tick arrives: a
        # run that fails halfway has already spent its LLM calls on nothing.
        if config.run.scenario is Scenario.NOISY_INFO:
            raise ShockNotImplemented(
                "noisy_info is admissible only if defined before the "
                "final runs (design v1.0 s10); it is not implemented"
            )
        self._config = config
        self._state = state
        self._zones = zones
        self._production = production
        self._market = market
        self._active: ShockState | None = None
        self._history: list[ShockState] = []

    @property
    def active(self) -> ShockState | None:
        return self._active

    @property
    def history(self) -> list[ShockState]:
        return list(self._history)

    @property
    def scenario(self) -> Scenario:
        return self._config.run.scenario

    def is_due(self, tick: int) -> bool:
        return self._active is None and not self._history and tick == self._config.phases.shock_tick

    def is_reversion_due(self, tick: int) -> bool:
        return self._active is not None and tick == self._config.phases.reversion_tick

    # -- application -------------------------------------------------------
    def apply(self, tick: int) -> ShockState:
        scenario = self.scenario
        shock = ShockState(scenario=scenario, applied_tick=tick)

        match scenario:
            case Scenario.RESOURCE_SCARCITY:
                settings = self._config.shocks.resource_scarcity
                self._zones.set_spawn_multiplier(
                    settings.target_resource, settings.spawn_multiplier
                )
                shock.details = {
                    "target_resource": settings.target_resource.value,
                    "spawn_multiplier": str(settings.spawn_multiplier),
                }
                if settings.stock_multiplier is not None:
                    # Pools hold several ticks of a resource, more where trips
                    # are few: without this the stock covered the shock.
                    self._zones.scale_stock(settings.target_resource, settings.stock_multiplier)
                    shock.details["stock_multiplier"] = str(settings.stock_multiplier)
                if settings.held_keep is not None:
                    self._buy_out_held(shock, settings.target_resource, settings.held_keep, tick)

            case Scenario.DEMAND_SURGE:
                settings = self._config.shocks.demand_surge
                shock.details = {
                    "target_product": settings.target_product.value,
                    "consumption_multiplier": str(settings.consumption_multiplier),
                }

            case Scenario.PRODUCTIVITY_DROP:
                settings = self._config.shocks.productivity_drop
                self._production.set_capacity_multiplier(
                    settings.target_specialization, settings.capacity_multiplier
                )
                shock.details = {
                    "target_specialization": settings.target_specialization.value,
                    "capacity_multiplier": str(settings.capacity_multiplier),
                }

            case Scenario.LIQUIDITY_LOSS:
                settings = self._config.shocks.liquidity_loss
                affected = self._affected_agents(settings)
                shock.affected_agents = affected
                for agent_id in affected:
                    account = self._state.account(agent_id)
                    frozen = account.freeze_gold(settings.frozen_gold_fraction, tick=tick)
                    shock.frozen_by_agent[agent_id] = frozen
                # Logged with the shock. The global view publishes only the
                # whitelisted details (PUBLIC_SHOCK_DETAILS in
                # orchestration/global_view.py), none of which is here: agents
                # see the shock's type, not who was frozen or by how much.
                shock.details = {
                    "affected_agents": affected,
                    "frozen_gold_fraction": str(settings.frozen_gold_fraction),
                    "frozen_total": str(sum(shock.frozen_by_agent.values(), ZERO)),
                }

            case Scenario.NOISY_INFO:
                raise ShockNotImplemented(
                    "noisy_info is admissible only if defined before the "
                    "final runs (design v1.0 s10); it is not implemented"
                )

        self._active = shock
        return shock

    def _buy_out_held(self, shock: ShockState, target: Resource, keep: int, tick: int) -> None:
        """Leave agents ``keep`` units of ``target`` in all, paid for at its guide value.

        A large stock at the shock sold cheap through it and hid it, most of it
        already on sell orders. Units on offer count
        too. Adventurers first: they are the ones who sell it, cheap, while
        artisans only craft with it, so the units left stay with them. One unit
        at a time from the largest holder (ties in agent order), so the stock
        left is spread rather than emptied from whoever is first.
        Free units go first; when the orders must give some up, they are
        withdrawn and the units left are offered again at the same prices, or
        none is when nothing is left.
        """
        price = Decimal(str(self._config.economy.initial_reference_price[target.value]))
        held = {a: self._state.account(a).quantity(target) for a in self._state.agent_ids}
        taken = dict.fromkeys(held, 0)
        sells = {a: self._state.account(a).role is Role.ADVENTURER for a in held}
        while sum(held.values()) > keep:
            # First of the largest, in agent order, among adventurers while any holds some.
            agent_id = max(held, key=lambda a: (sells[a] and held[a] > 0, held[a]))
            held[agent_id] -= 1
            taken[agent_id] += 1
        for agent_id, units in taken.items():
            if not units:
                continue
            account = self._state.account(agent_id)
            if units > account.available_quantity(target):
                self._cut_orders(shock, agent_id, target, held[agent_id], tick)
            account.consume(target, units)
            account.credit_gold(price * units)
            shock.confiscated[agent_id] = (units, price * units)
        shock.details["held_keep"] = keep
        shock.details["held_removed"] = sum(taken.values())
        shock.details["held_price"] = str(price)

    def _cut_orders(self, shock: ShockState, agent_id: str, target: Resource, left: int, tick: int) -> None:
        """Withdraw the agent's sell orders of ``target`` and offer ``left`` units again."""
        if self._market is None:
            raise RuntimeError("cutting sell orders needs the market")
        orders = [
            o
            for o in self._market.open_orders_of(agent_id)
            if o.good is target and o.side is OrderSide.SELL
        ]
        before = sum(o.remaining for o in orders)
        for order in orders:
            self._market.cancel(order.order_id)
        after = 0
        for order in orders:  # oldest first, as they were placed
            units = min(order.remaining, left - after)
            if units <= 0:
                break
            self._market.place(
                agent_id=agent_id,
                side=OrderSide.SELL,
                good=target,
                quantity=units,
                price=order.limit_price,
                tick=tick,
            )
            after += units
        shock.orders_cut[agent_id] = (before, after)

    def _affected_agents(self, settings: LiquidityLossConfig) -> list[str]:
        """Resolve the configured subset (design v1.0 s10).

        Never inferred from ordering. An explicit list is used as given; a
        role-stratified count takes that many agents of each role in id order,
        which keeps the shock's role composition an intentional, fixed choice.
        Either way the subset is identical across a seed-trio by construction,
        without consuming a random draw whose position in a stream could differ
        between conditions.
        """
        if settings.affected_agents is not None:
            unknown = [a for a in settings.affected_agents if a not in self._state.accounts]
            if unknown:
                raise ValueError(
                    f"liquidity_loss names agents that do not exist: {sorted(unknown)}"
                )
            return [a for a in self._state.agent_ids if a in set(settings.affected_agents)]

        assert settings.affected_by_role is not None  # guaranteed by config validation
        affected: list[str] = []
        for role in sorted(settings.affected_by_role, key=lambda r: r.value):
            count = settings.affected_by_role[role]
            affected.extend(self._state.agents_with_role(role)[:count])
        return [a for a in self._state.agent_ids if a in set(affected)]

    # -- reversion ---------------------------------------------------------
    def revert(self, tick: int) -> ShockState:
        shock = self._active
        if shock is None:  # pragma: no cover -- callers check is_reversion_due
            raise RuntimeError("no active shock to revert")

        match shock.scenario:
            case Scenario.RESOURCE_SCARCITY:
                self._zones.clear_spawn_multiplier(
                    self._config.shocks.resource_scarcity.target_resource
                )
            case Scenario.PRODUCTIVITY_DROP:
                self._production.clear_capacity_multiplier(
                    self._config.shocks.productivity_drop.target_specialization
                )
            case Scenario.LIQUIDITY_LOSS:
                released = ZERO
                for agent_id in sorted(shock.affected_agents):
                    released += self._state.account(agent_id).unfreeze_all_gold()
                # Logged with the reversion: what the freeze held at the end,
                # frozen at the shock plus frozen later from pre-shock bids.
                shock.details["unfrozen_total"] = str(released)
            case _:
                pass  # demand_surge is read through consumption_multiplier

        shock.reverted_tick = tick
        self._active = None
        self._history.append(shock)
        return shock

    # -- live parameters read by the engine --------------------------------
    def consumption_multiplier(self, product: object) -> Decimal:
        """Multiplier on a product's provisions per expedition while a surge is active."""
        settings = self._config.shocks.demand_surge
        if (
            self._active is not None
            and self._active.scenario is Scenario.DEMAND_SURGE
            and product == settings.target_product
        ):
            return settings.consumption_multiplier
        return Decimal(1)
