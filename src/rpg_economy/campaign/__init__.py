"""Pilot and final run orchestration (design v1.0 s10, s11.4; IMPLEMENTATION.md phase 7).

The campaign layer decides *which* runs happen. It never decides anything about
how a run behaves: it composes conditions, scenarios, experiments and seeds into
a plan, prices that plan, and executes it. Nothing here may influence a run's
outcome.
"""

from rpg_economy.campaign.cost import (
    CostEstimate,
    TokenProfile,
    estimate_cost,
    measure_profile,
)
from rpg_economy.campaign.plan import (
    PLAN_SCHEMA_VERSION,
    CampaignPlan,
    PlannedTrio,
    ScenarioTier,
    build_plan,
    planned_secondary_scenarios,
    scenario_tier,
)

__all__ = [
    "PLAN_SCHEMA_VERSION",
    "CampaignPlan",
    "CostEstimate",
    "PlannedTrio",
    "ScenarioTier",
    "TokenProfile",
    "build_plan",
    "estimate_cost",
    "measure_profile",
    "planned_secondary_scenarios",
    "scenario_tier",
]
