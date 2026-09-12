"""Role logic, traits, prompt building and the agent action schema.

Rule-based policies and LLM-backed agents both emit
the same structured ``AgentAction`` and go through the same executor.
"""

from rpg_economy.agents.action import AgentAction, wait
from rpg_economy.agents.decision_phase import RuleBasedDecisionPhase
from rpg_economy.agents.executor import ExecutionOutcome, execute
from rpg_economy.agents.rule_based import (
    AdventurerPolicy,
    ArtisanPolicy,
    RuleBasedPolicy,
    policy_for,
    reference_price,
)

__all__ = [
    "AdventurerPolicy",
    "AgentAction",
    "ArtisanPolicy",
    "ExecutionOutcome",
    "RuleBasedDecisionPhase",
    "RuleBasedPolicy",
    "execute",
    "policy_for",
    "reference_price",
    "wait",
]
