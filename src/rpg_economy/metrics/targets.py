"""Pre-registered primary targets per scenario (design v1.0 s11.3).

The unit of analysis is the run, so every primary metric must produce one
value per run before any contrast is applied. The rule "on the directly
affected good" only identifies a target unambiguously for resource scarcity;
this table extends it to every scenario, and it is fixed before the final runs.
It is fixed here rather than computed from the data, so it cannot be chosen
after seeing results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import (
    RESOURCE_TO_MAIN_PRODUCT,
    ArtisanSpecialization,
    Good,
    Product,
    Scenario,
)


class TargetKind(StrEnum):
    """What quantity the scenario's recovery is measured on."""

    #: The transaction price of a single good.
    PRICE = "price"
    #: Total traded value across all goods -- aggregate market activity.
    TRADED_VALUE = "traded_value"


@dataclass(frozen=True)
class PrimaryTarget:
    """The pre-registered target of one scenario."""

    scenario: Scenario
    kind: TargetKind
    good: Good | None
    #: Metrics that are primary for this scenario; the rest are secondary.
    primary_metrics: tuple[str, ...]

    @property
    def label_key(self) -> str:
        """i18n key for rendering, or the aggregate label."""
        if self.good is None:
            return "metric.volume"
        return f"{'resource' if self.good.value in _RESOURCES else 'product'}.{self.good.value}"


_RESOURCES = {"iron", "wood", "leather", "herbs", "arcane_crystal"}

#: The main finished product of each artisan specialization (design v1.0 s11.3).
#: Explicit rather than searched for in the resource -> product mapping: two of
#: that mapping's products (bow/gear and staff/focus) are both woodworking
#: recipes, so a search would pick whichever came first in a dict. Each entry
#: follows the frozen mapping: iron -> melee weapon, wood -> bow/gear,
#: herbs -> health potion.
SPECIALIZATION_MAIN_PRODUCT: dict[ArtisanSpecialization, Product] = {
    ArtisanSpecialization.SMITHING: Product.MELEE_WEAPON,
    ArtisanSpecialization.WOODWORKING: Product.BOW_GEAR,
    ArtisanSpecialization.ALCHEMY: Product.HEALTH_POTION,
}


def specialization_main_product(
    config: SimulationConfig, specialization: ArtisanSpecialization
) -> Product:
    """The pre-registered main product of a specialization, checked against the recipes.

    Fails loudly rather than guessing: a specialization with no entry, or an
    entry the configured recipes or the resource -> product mapping no longer
    support, would otherwise move the productivity-drop target silently.
    """
    product = SPECIALIZATION_MAIN_PRODUCT.get(specialization)
    if product is None:
        raise ValueError(f"no main product is pre-registered for {specialization.value}")
    if product not in RESOURCE_TO_MAIN_PRODUCT.values():
        raise ValueError(
            f"{product.value} is not a main product of the resource -> product mapping"
        )
    recipe = config.economy.recipes.get(product)
    if recipe is None or recipe.specialization is not specialization:
        raise ValueError(
            f"{product.value} is not a {specialization.value} recipe in this configuration"
        )
    return product


def primary_target(config: SimulationConfig) -> PrimaryTarget:
    """The primary target for this run's scenario (design v1.0 s11.3)."""
    scenario = config.run.scenario

    match scenario:
        case Scenario.RESOURCE_SCARCITY:
            # The resource whose appearance rate is reduced.
            return PrimaryTarget(
                scenario=scenario,
                kind=TargetKind.PRICE,
                good=config.shocks.resource_scarcity.target_resource,
                primary_metrics=("volatility", "stockouts", "recovery_time"),
            )

        case Scenario.DEMAND_SURGE:
            # The product whose consumption is increased.
            return PrimaryTarget(
                scenario=scenario,
                kind=TargetKind.PRICE,
                good=config.shocks.demand_surge.target_product,
                primary_metrics=("volatility", "stockouts", "recovery_time"),
            )

        case Scenario.PRODUCTIVITY_DROP:
            # The main finished product of the affected specialization, via the
            # fixed resource -> product mapping.
            product = specialization_main_product(
                config, config.shocks.productivity_drop.target_specialization
            )
            return PrimaryTarget(
                scenario=scenario,
                kind=TargetKind.PRICE,
                good=product,
                primary_metrics=("volatility", "stockouts", "recovery_time"),
            )

        case Scenario.LIQUIDITY_LOSS:
            # No single affected good: the target is aggregate market activity.
            # Volatility and stockouts become secondary here (design s11.3).
            return PrimaryTarget(
                scenario=scenario,
                kind=TargetKind.TRADED_VALUE,
                good=None,
                primary_metrics=("traded_value", "recovery_time"),
            )

    raise ValueError(f"no pre-registered target for scenario {scenario}")


def propagation_product(resource: Good) -> Good | None:
    """The main finished product of a resource (design v1.0 s11.3).

    Used by the pre-registered *secondary* propagation analysis: how a shock to
    a raw material reaches the finished good that consumes it.
    """
    return RESOURCE_TO_MAIN_PRODUCT.get(resource)
