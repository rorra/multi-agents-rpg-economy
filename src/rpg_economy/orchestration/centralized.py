"""Condition A wiring: Director, TASK-driven activations, RESULT construction.

An activation in Condition A is one agent call, not two. Its structured output
carries the action together with the agent-supplied RESULT fields (``needs``,
``offers``, ``observations``), while ``status`` and ``outcome`` are filled in
from what the engine actually did -- an agent does not get to declare its own
success (design v1.0 s3.1.2, s3.1.3).

``action_taken`` is the structured action **with ``short_reason`` removed**
(design v1.0 s3.1.2): the reason is written to the log for qualitative analysis
and never travels to the Director.
"""

from __future__ import annotations

from rpg_economy.agents.llm_agent import LLMAgentRunner
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import ActionType, ResultStatus
from rpg_economy.engine import Engine
from rpg_economy.orchestration.protocol import Need, Offer, Result, ResultObservation, Task


class TaskActivationRunner:
    """Runs one TASK-driven activation and returns its RESULT."""

    def __init__(self, config: SimulationConfig, agents: LLMAgentRunner) -> None:
        self._config = config
        self._agents = agents
        self._next_result = 1

    def __call__(self, engine: Engine, agent_id: str, task: Task) -> Result:
        activation = self._agents.activate(engine, agent_id, task=task.to_agent_payload())

        result_id = f"result-{self._next_result}"
        self._next_result += 1

        return Result(
            result_id=result_id,
            task_id=task.task_id,
            agent_id=agent_id,
            status=self._status(activation),
            # Structured action, stripped of free-text reasoning (design s3.1.2).
            action_taken=activation.action.to_director_payload(),
            outcome={
                "accepted": activation.outcome.accepted,
                "engine_operation": activation.outcome.engine_operation,
                # A closed code, never the engine's text: that names agents and
                # quotes their holdings (design s5.2, s3.1.2).
                "rejection": (
                    None if activation.outcome.code is None else activation.outcome.code.value
                ),
                "wasted": activation.wasted,
                # place_orders: each order and whether the engine took it, with
                # its closed rejection code.
                **(
                    {"orders": [dict(r) for r in activation.outcome.order_results]}
                    if activation.outcome.order_results
                    else {}
                ),
            },
            needs=self._needs(activation),
            offers=self._offers(activation),
            observations=self._observations(activation),
        )

    @staticmethod
    def _status(activation) -> ResultStatus:
        """Derive the status from what the engine did, never from the agent.

        Design s3.1.2 fixes FAILED for output that stays invalid after its one
        retry. The rest follows the engine's verdict: a rejected action is
        BLOCKED, a deliberate WAIT is PARTIAL, and an accepted operation is
        SUCCESS. A place_orders the engine took only in part is PARTIAL.
        """
        if activation.invalid_outputs >= activation.attempts:
            return ResultStatus.FAILED
        if not activation.outcome.accepted:
            return ResultStatus.BLOCKED
        if activation.action.action is ActionType.WAIT:
            return ResultStatus.PARTIAL
        if any(not r["accepted"] for r in activation.outcome.order_results):
            return ResultStatus.PARTIAL
        return ResultStatus.SUCCESS

    # The fields were validated with the action (agents.parsing): a malformed
    # entry was invalid output and went through the retry, so what arrives here
    # is either validated or, after a failed retry, absent.
    @staticmethod
    def _needs(activation) -> list[Need]:
        return [] if activation.result_fields is None else list(activation.result_fields.needs)

    @staticmethod
    def _offers(activation) -> list[Offer]:
        return [] if activation.result_fields is None else list(activation.result_fields.offers)

    @staticmethod
    def _observations(activation) -> list[ResultObservation]:
        if activation.result_fields is None:
            return []
        return list(activation.result_fields.observations)
