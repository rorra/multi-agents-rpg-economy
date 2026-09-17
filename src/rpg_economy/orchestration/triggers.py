"""Mechanical activation triggers T1-T5 (design v1.0 s8.2). Conditions B and C only.

**No LLM is involved here, and none may ever be.** These are fixed rules the
engine evaluates; the design's separation of mechanism from semantics (s2)
depends on it, and a poisoned-mock test asserts nothing in this module can reach
a language model.

The five triggers:

* **T1** -- a peer-to-peer message arrived since the agent's last activation.
  **The Condition C bulletin is explicitly excluded** and fires no trigger at
  all (s3.3). The exclusion is structural, not a check: T1 reads the message bus,
  and the bulletin never enters it.
* **T2** -- one of the agent's own orders was filled, expired or cancelled.
* **T3** -- the transaction price of a good in the agent's **fixed interest
  list** moved beyond a configured threshold since its last activation.
* **T4** -- one of the agent's own state thresholds was crossed: no inputs to
  produce with, idle production capacity, gold outside its bounds, critical
  equipment durability, or expedition recovery finished. It fires on the
  *crossing* -- a condition that was false becoming true -- not on every tick
  the condition holds. An artisan is nearly always either idle or out of
  inputs, so a level trigger would keep it asking to act for the whole run.
* **T5** -- N ticks of inactivity.

Thresholds and N are configuration, identical in B and C -- a binding control
(s12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Good, Role, Trigger, parse_good
from rpg_economy.engine import Engine
from rpg_economy.logging_ import EventType
from rpg_economy.messaging import MessageBus
from rpg_economy.orchestration.activation import ActivationRequest

ZERO = Decimal(0)


@dataclass
class AgentTriggerState:
    """What T1-T5 need to remember between one activation and the next."""

    last_activation_tick: int = 0
    order_event_since_activation: bool = False
    price_at_last_activation: dict[str, Decimal] = field(default_factory=dict)
    #: T4 conditions true at the previous evaluation, to detect a crossing.
    t4_previous: frozenset[str] = frozenset()
    #: T4 conditions that crossed since the last activation and still hold.
    t4_crossed: set[str] = field(default_factory=set)


class TriggerEvaluator:
    """Evaluates T1-T5 for every agent, once per tick."""

    def __init__(self, config: SimulationConfig, engine: Engine, messages: MessageBus) -> None:
        self._config = config
        self._engine = engine
        self._messages = messages
        self._state: dict[str, AgentTriggerState] = {
            agent_id: AgentTriggerState() for agent_id in engine.state.agent_ids
        }
        self._pending: dict[str, ActivationRequest] = {}
        self._interest: dict[str, list[Good]] = {}
        for agent_id in engine.state.agent_ids:
            account = engine.state.account(agent_id)
            self._interest[agent_id] = [
                parse_good(value)
                for value in config.observability.interest_of(
                    account.role.value, account.specialization
                )
            ]
        engine.add_order_observer(self._note_order_event)
        for agent_id in engine.state.agent_ids:
            self._snapshot_prices(agent_id)
            # A condition that already holds at the start was never crossed.
            self._state[agent_id].t4_previous = self._t4_conditions(agent_id, engine.tick)

    # -- bookkeeping -------------------------------------------------------
    def _note_order_event(self, event: EventType, agent_id: str) -> None:
        """Record that one of an agent's own orders changed status (T2)."""
        if event in {
            EventType.ORDER_FILLED,
            EventType.ORDER_PARTIALLY_FILLED,
            EventType.ORDER_EXPIRED,
            EventType.ORDER_CANCELLED,
        }:
            self._state[agent_id].order_event_since_activation = True

    def _snapshot_prices(self, agent_id: str) -> None:
        state = self._state[agent_id]
        state.price_at_last_activation = {
            good.value: self._reference(good) for good in self._interest[agent_id]
        }

    def _reference(self, good: Good) -> Decimal:
        """Latest transaction price, or zero if the good has never traded."""
        trades = self._engine.market.last_trades(good, limit=1)
        return trades[-1].price if trades else ZERO

    def on_activated(self, agent_id: str, tick: int) -> None:
        """Clear the agent's trigger state, once the activation has happened.

        Called *after* the agent acts, so its prompt could still contain the
        messages that triggered it. Prices are snapshotted here too, meaning T3
        measures movement since the agent last acted rather than since it was
        last scheduled.
        """
        state = self._state[agent_id]
        state.last_activation_tick = tick
        state.order_event_since_activation = False
        self._messages.drain(agent_id)
        self._snapshot_prices(agent_id)
        # A condition still true after acting is not a new crossing.
        state.t4_crossed.clear()
        state.t4_previous = self._t4_conditions(agent_id, tick)
        self._pending.pop(agent_id, None)

    @property
    def pending(self) -> dict[str, ActivationRequest]:
        return dict(self._pending)

    def clear_pending(self) -> None:
        """Discard all standing requests. Used by tests to reach a quiet cycle."""
        self._pending.clear()

    # -- the triggers ------------------------------------------------------
    def _t1_message_received(self, agent_id: str) -> bool:
        return self._messages.unread_count(agent_id) > 0

    def _t2_order_event(self, agent_id: str) -> bool:
        return self._state[agent_id].order_event_since_activation

    def _t3_price_move(self, agent_id: str) -> bool:
        threshold = self._config.triggers.t3_price_move_pct
        snapshot = self._state[agent_id].price_at_last_activation
        for good in self._interest[agent_id]:
            before = snapshot.get(good.value, ZERO)
            now = self._reference(good)
            if before <= ZERO or now <= ZERO:
                continue
            if abs(now - before) / before >= threshold:
                return True
        return False

    def _t4_conditions(self, agent_id: str, tick: int) -> frozenset[str]:
        """The agent's own-state conditions that hold right now."""
        account = self._engine.state.account(agent_id)
        settings = self._config.triggers
        held: set[str] = set()

        # Available gold: what the agent can actually spend. Gold reserved by
        # its own bids or frozen by a liquidity loss is not, so a freeze shows
        # here as it does in the agent's own state; identical in B and C.
        available = account.available_gold
        if available <= settings.t4_gold_low:
            held.add("gold_low")
        if available >= settings.t4_gold_high:
            held.add("gold_high")

        for product, remaining in account.equipment.items():
            full = self._config.economy.durability.get(product)
            if full and Decimal(remaining) / full <= settings.t4_durability_critical_pct:
                held.add(f"durability_critical:{product.value}")

        # Away -> available is the crossing that marks a finished recovery.
        if account.role is Role.ADVENTURER and not account.is_away(tick):
            held.add("available")
        # Back in town: the adventurer has its haul to sell and its gear to
        # replace, and only its rest to do it in.
        if account.role is Role.ADVENTURER and account.expedition is None:
            held.add("in_town")

        if account.role is Role.ARTISAN:
            specialization = self._engine.production.specialization_of(account)
            if specialization is not None:
                products = self._engine.production.products_for(specialization)
                blocked = all(
                    self._engine.production.missing_inputs(account, product) for product in products
                )
                if blocked:
                    held.add("no_inputs")
                elif self._engine.production.capacity_of(account) > 0:
                    held.add("idle_capacity")
        return frozenset(held)

    def _update_t4(self, agent_id: str, tick: int) -> None:
        """Record the conditions that crossed from false to true since last tick."""
        state = self._state[agent_id]
        now = self._t4_conditions(agent_id, tick)
        state.t4_crossed |= now - state.t4_previous
        state.t4_crossed &= now  # a crossing that reverted no longer stands
        state.t4_previous = now

    def _t4_own_state(self, agent_id: str, tick: int) -> bool:
        crossed = self._state[agent_id].t4_crossed
        return bool(crossed & self._t4_conditions(agent_id, tick))

    def _t5_inactivity(self, agent_id: str, tick: int) -> bool:
        elapsed = tick - self._state[agent_id].last_activation_tick
        return elapsed >= self._config.triggers.t5_inactivity_ticks

    # -- evaluation --------------------------------------------------------
    def evaluate(self, tick: int) -> list[ActivationRequest]:
        """Evaluate every trigger for every agent at ``tick``.

        Conditions phrased "since the last activation" re-evaluate to true until
        the agent is served, which is exactly the persistence design s8.3
        requires of unserved requests: they stand while their trigger holds.
        """
        checks = (
            (Trigger.T1_MESSAGE_RECEIVED, lambda a: self._t1_message_received(a)),
            (Trigger.T2_ORDER_EVENT, lambda a: self._t2_order_event(a)),
            (Trigger.T3_PRICE_MOVE, lambda a: self._t3_price_move(a)),
            (Trigger.T4_OWN_STATE_THRESHOLD, lambda a: self._t4_own_state(a, tick)),
            (Trigger.T5_INACTIVITY, lambda a: self._t5_inactivity(a, tick)),
        )

        fired: list[ActivationRequest] = []
        for agent_id in self._engine.state.agent_ids:
            self._update_t4(agent_id, tick)
            for trigger, predicate in checks:
                if not predicate(agent_id):
                    continue
                request = self._pending.get(agent_id)
                if request is None:
                    request = ActivationRequest(agent_id=agent_id, first_requested_tick=tick)
                    self._pending[agent_id] = request
                    fired.append(request)
                # Every firing is logged, per agent and tick (design s15), not
                # only the first on the request: ``pending`` marks one whose
                # trigger the agent's standing request already carries. The
                # request itself is unchanged by a repeat firing.
                pending = trigger in request.triggers
                request.add(trigger)
                self._engine.log_event(
                    EventType.TRIGGER_FIRED,
                    agent_id=agent_id,
                    payload={"trigger": trigger.value, "pending": pending},
                )
        for request in fired:
            self._engine.log_event(
                EventType.ACTIVATION_REQUESTED,
                agent_id=request.agent_id,
                payload={"triggers": request.trigger_values()},
            )
        return list(self._pending.values())

    def drop_resolved(self, tick: int) -> None:
        """Discard requests whose trigger conditions no longer hold.

        Only T4 can resolve without the agent acting -- gold moving back inside
        its bounds, for instance. The "since last activation" triggers stand
        until served, which is what design s8.3 intends by persistence.
        """
        for agent_id, request in list(self._pending.items()):
            still = {
                Trigger.T1_MESSAGE_RECEIVED: self._t1_message_received(agent_id),
                Trigger.T2_ORDER_EVENT: self._t2_order_event(agent_id),
                Trigger.T3_PRICE_MOVE: self._t3_price_move(agent_id),
                Trigger.T4_OWN_STATE_THRESHOLD: self._t4_own_state(agent_id, tick),
                Trigger.T5_INACTIVITY: self._t5_inactivity(agent_id, tick),
            }
            resolved = request.trigger_values()
            request.triggers = {t for t in request.triggers if still[t]}
            if not request.triggers:
                del self._pending[agent_id]
                # Logged, so a request that vanished unserved is accounted for
                # in the log rather than only by its absence (design s15).
                self._engine.log_event(
                    EventType.ACTIVATION_REQUEST_RESOLVED,
                    agent_id=agent_id,
                    payload={
                        "triggers": resolved,
                        "waiting_since_tick": request.first_requested_tick,
                    },
                )
