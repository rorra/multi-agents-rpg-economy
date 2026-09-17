"""Conditions B and C: triggers, neutral scheduler, K budget (design v1.0 s3.2, s8).

The flow of one macro-cycle (design v1.0 s8.1):

1. The engine runs its mechanical ticks. **After each tick** this orchestrator
   evaluates T1-T5 and accumulates activation requests, so a request raised
   early in a cycle is not lost by its end.
2. At the cycle boundary the neutral scheduler serves up to K of them.
3. Served agents act **sequentially, each seeing the previous one's effects**.
4. Unserved requests persist while their triggers still hold; the cycle closes
   early when nothing is pending.

No Director, no semantic coordinator. Nothing in this path can reach a language
model except the agent policy itself, which is invoked only for an activation
the scheduler has already granted.
"""

from __future__ import annotations

from collections.abc import Callable

from rpg_economy.config.models import SimulationConfig
from rpg_economy.engine import Engine
from rpg_economy.llm.budget import TokenBudget
from rpg_economy.logging_ import EventType
from rpg_economy.messaging import Directory, DirectoryEntry, MessageBus
from rpg_economy.orchestration.activation import ActivationRequest
from rpg_economy.orchestration.scheduler import NeutralScheduler
from rpg_economy.orchestration.triggers import TriggerEvaluator

#: What an activation does once granted: rule-based or LLM-backed; the
#: orchestration around it is identical either way.
ActivationHandler = Callable[[Engine, str], None]

#: Why a standing request was not served this cycle.
DEFERRED_K_LIMIT = "k_limit"
DEFERRED_BUDGET_EXHAUSTED = "budget_exhausted"


def build_directory(engine: Engine) -> Directory:
    """The full static public directory: id, role and specialization only."""
    return Directory(
        DirectoryEntry(
            agent_id=agent_id,
            role=engine.state.account(agent_id).role,
            specialization=engine.state.account(agent_id).specialization,
        )
        for agent_id in engine.state.agent_ids
    )


class DecentralizedOrchestrator:
    """Drives Conditions B and C for one run."""

    def __init__(
        self,
        config: SimulationConfig,
        engine: Engine,
        activate: ActivationHandler,
        *,
        messages: MessageBus | None = None,
        budget: TokenBudget | None = None,
    ) -> None:
        self._budget = budget or TokenBudget()
        self._config = config
        self._engine = engine
        self._activate = activate
        self._messages = messages or MessageBus(build_directory(engine))
        # An open REQUEST or OFFER lasts as long as an order rests on the book.
        self._messages.set_proposal_expiry(config.market.order_expiry_ticks, lambda: engine.tick)
        self._triggers = TriggerEvaluator(config, engine, self._messages)
        self._scheduler = NeutralScheduler(
            engine.state.agent_ids, engine.random, config.time.k_activations
        )
        self._cycles_closed_early = 0
        self._last_cycle: dict = {}
        engine.add_tick_observer(self._on_tick)
        engine.add_cycle_summary_provider(lambda: dict(self._last_cycle))

    # -- accessors ---------------------------------------------------------
    @property
    def messages(self) -> MessageBus:
        return self._messages

    @property
    def triggers(self) -> TriggerEvaluator:
        return self._triggers

    @property
    def scheduler(self) -> NeutralScheduler:
        return self._scheduler

    @property
    def cycles_closed_early(self) -> int:
        return self._cycles_closed_early

    # -- per-tick ----------------------------------------------------------
    def _on_tick(self, engine: Engine) -> None:
        """Evaluate triggers after every mechanical tick."""
        self._triggers.evaluate(engine.tick)

    # -- per-cycle ---------------------------------------------------------
    def __call__(self, engine: Engine) -> None:
        """The decision phase: serve up to K requests."""
        # Budget exhausted: activations cease because every activation is an LLM
        # call. Triggers keep firing and the engine keeps ticking -- orders
        # execute or expire, durability decays, scheduled shocks fire -- until
        # the planned horizon (design v1.0 s8.4).
        if not self._budget.may_call():
            # Resolved requests go first, as on the ordinary path: otherwise a
            # request whose triggers stopped holding would be logged as
            # deferred every cycle until the horizon, inflating cycle_end.
            self._triggers.drop_resolved(engine.tick)
            pending = self._triggers.pending
            for agent_id in sorted(pending):
                self._log_deferred(engine, pending[agent_id], DEFERRED_BUDGET_EXHAUSTED)
            self._last_cycle = {
                "effective_activations": 0,
                "requested": len(pending),
                "deferred": len(pending),
                "k": self._scheduler.k,
                "closed_early": False,
                "post_exhaustion": True,
            }
            return

        self._triggers.drop_resolved(engine.tick)
        pending = self._triggers.pending

        if not pending:
            # Nothing is asking to act: close the cycle early (design s8.1).
            self._cycles_closed_early += 1
            self._last_cycle = {
                "effective_activations": 0,
                "requested": 0,
                "deferred": 0,
                "k": self._scheduler.k,
                "closed_early": True,
                # The budget is not spent (checked above); keyed on every
                # branch so every cycle_end payload has one schema.
                "post_exhaustion": False,
            }
            return

        # The scheduler sees ids and nothing else: it is handed the sorted ids,
        # never the requests, so it could not read their triggers even by
        # mistake (design s8.3).
        served = self._scheduler.serve(sorted(pending))
        deferred = sorted(set(pending) - set(served))

        for agent_id in deferred:
            self._log_deferred(engine, pending[agent_id], DEFERRED_K_LIMIT)

        performed = 0
        for position, agent_id in enumerate(served):
            # Checked per activation, not once per cycle: every activation is an
            # LLM call, so a cycle-level check alone would let a whole K worth of
            # calls run past exhaustion (design v1.0 s8.4).
            if not self._budget.may_call():
                for skipped in served[position:]:
                    self._log_deferred(engine, pending[skipped], DEFERRED_BUDGET_EXHAUSTED)
                break
            request = pending[agent_id]
            engine.log_event(
                EventType.ACTIVATION_SERVED,
                agent_id=agent_id,
                payload={
                    "triggers": request.trigger_values(),
                    "waiting_ticks": engine.tick - request.first_requested_tick,
                    "accumulated_activations": self._scheduler.activations_of(agent_id),
                },
            )
            self._scheduler.record_activation(agent_id)
            # Activations run sequentially, each seeing the previous one's effects.
            #
            # The agent acts BEFORE its trigger state is cleared, so it can still
            # read the mail that triggered it. Clearing first would drain the
            # inbox unread: T1 would fire, the agent would be activated, and the
            # message would never reach its prompt or its token bucket.
            self._activate(engine, agent_id)
            self._triggers.on_activated(agent_id, engine.tick)
            performed += 1

        self._last_cycle = {
            "effective_activations": performed,
            "requested": len(pending),
            "deferred": len(deferred) + (len(served) - performed),
            "k": self._scheduler.k,
            "closed_early": False,
            "post_exhaustion": False,
        }

    @staticmethod
    def _log_deferred(engine: Engine, request: ActivationRequest, reason: str) -> None:
        """A standing request left unserved this cycle, and why (design s15)."""
        engine.log_event(
            EventType.ACTIVATION_DEFERRED,
            agent_id=request.agent_id,
            payload={
                "triggers": request.trigger_values(),
                "waiting_since_tick": request.first_requested_tick,
                "reason": reason,
            },
        )
