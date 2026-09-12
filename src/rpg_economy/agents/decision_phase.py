"""A decision phase driven by rule-based policies.

It activates every agent once per macro-cycle in a fixed order, so the economy
runs end to end without the activation machinery (triggers, the neutral
scheduler and the K budget) and without the LLM layer.

Activations execute sequentially, each seeing the previous one's effects,
which is the sequencing rule design v1.0 s8.1 fixes identically for all three
conditions.
"""

from __future__ import annotations

from rpg_economy.agents.executor import execute
from rpg_economy.agents.rule_based import RuleBasedPolicy, policy_for
from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import Role
from rpg_economy.engine import Engine


class RuleBasedActivation:
    """Activates one agent with its rule-based policy.

    The activation handler the neutral scheduler calls, so offline runs can use
    the experiment's own activation regime (triggers, scheduler, K per cycle)
    instead of activating everyone every cycle.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self._policies: dict[Role, RuleBasedPolicy] = {
            role: policy_for(role, config) for role in Role
        }

    def __call__(self, engine: Engine, agent_id: str) -> None:
        account = engine.state.account(agent_id)
        action = self._policies[account.role].decide(engine, account, engine.random.agent(agent_id))
        execute(engine, agent_id, action)


class RuleBasedDecisionPhase:
    """Activates every agent once per macro-cycle."""

    def __init__(self, config: SimulationConfig) -> None:
        self._policies: dict[Role, RuleBasedPolicy] = {
            role: policy_for(role, config) for role in Role
        }

    def __call__(self, engine: Engine) -> None:
        for agent_id in engine.state.agent_ids:
            account = engine.state.account(agent_id)
            policy = self._policies[account.role]
            action = policy.decide(engine, account, engine.random.agent(agent_id))
            execute(engine, agent_id, action)
