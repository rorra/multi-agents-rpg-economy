"""Statistical contrasts and i18n-aware report generation.

``economy_check`` is a diagnostic that checks the simulated economy behaves
sensibly, independent of orchestration.
"""

from rpg_economy.analysis.economy_check import (
    EconomyCheck,
    PhaseStats,
    RoleHoldings,
    StockoutStats,
    analyse,
    phase_of,
    target_good_for,
)
from rpg_economy.analysis.render import render

__all__ = [
    "EconomyCheck",
    "PhaseStats",
    "RoleHoldings",
    "StockoutStats",
    "analyse",
    "phase_of",
    "render",
    "target_good_for",
]
