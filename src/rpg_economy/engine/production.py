"""Crafting: raw materials into products (design v1.0 s5.1).

An artisan converts recipe inputs into one product, limited by its
specialization and by a per-macro-cycle capacity. Capacity is what the
``productivity_drop`` shock reduces (s10).

Capacity counts recipe runs (batches), not output units: one run consumes the
recipe's inputs once and yields its ``output_quantity``. A produce request's
quantity is likewise a number of runs.

Capacity is a credit that accrues at each cycle boundary at the
specialization's rate. A rate of one or more refills it to that rate every
cycle. A rate below one lets a run be made every few cycles instead: an
integer capacity can never be set below one run per cycle, which is far above
what an artisan actually crafts, so a cut to it would never bind.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from rpg_economy.config.models import EconomyConfig
from rpg_economy.domain import ArtisanSpecialization, Good, Product, Resource, Role
from rpg_economy.engine.state import AgentAccount

ONE = Decimal(1)
#: The most leftover run fraction kept between cycles: just under one run.
CARRY = ONE - Decimal("0.000001")
ZERO = Decimal(0)


def unit_cost(
    economy: EconomyConfig, last_price: Callable[[Good], Decimal | None], product: Good
) -> Decimal:
    """What the inputs of one unit of ``product`` cost at the last traded prices.

    Each input at its last trade (``last_price``), or at its starting reference
    price before any trade. The profit tax is assessed against it, and artisans
    see the same figure in their prompt.
    """
    recipe = economy.recipes[product]
    total = Decimal(0)
    for resource, quantity in recipe.inputs.items():
        price = last_price(resource)
        if price is None:
            price = Decimal(str(economy.initial_reference_price[resource.value]))
        total += price * quantity
    return (total / recipe.output_quantity).quantize(Decimal("0.01"))


class ProductionRejected(Exception):
    """Raised when a craft attempt is not permitted by the engine."""


@dataclass(frozen=True)
class ProductionResult:
    agent_id: str
    product: Product
    quantity: int
    consumed: dict[Resource, int]


class ProductionSystem:
    """Validates and executes crafting."""

    def __init__(self, config: EconomyConfig) -> None:
        self._config = config
        #: Per-specialization capacity multiplier, driven by shocks. 1 = normal.
        self._capacity_multiplier: dict[ArtisanSpecialization, Decimal] = {}

    # -- capacity ----------------------------------------------------------
    def capacity_multiplier(self, specialization: ArtisanSpecialization) -> Decimal:
        return self._capacity_multiplier.get(specialization, ONE)

    def set_capacity_multiplier(
        self, specialization: ArtisanSpecialization, multiplier: Decimal
    ) -> None:
        self._capacity_multiplier[specialization] = multiplier

    def clear_capacity_multiplier(self, specialization: ArtisanSpecialization) -> None:
        self._capacity_multiplier.pop(specialization, None)

    def rate_of(self, specialization: ArtisanSpecialization) -> Decimal:
        """Recipe runs per macro-cycle, after any productivity shock."""
        base = Decimal(self._config.production_capacity_per_cycle.get(specialization, 0))
        return base * self.capacity_multiplier(specialization)

    def _ceiling(self, specialization: ArtisanSpecialization) -> Decimal:
        """The most credit that can build up: one cycle's rate plus a carry.

        The carry, just under one run, keeps the fraction left after crafting,
        so a rate of 0.3 yields 3 runs every 10 cycles rather than being
        rounded down to one run every 4. It never adds a whole run: an idle
        artisan still crafts at most the rate's whole runs plus one leftover.
        """
        rate = self.rate_of(specialization)
        return rate + CARRY if rate > ZERO else ZERO

    def _credit(self, account: AgentAccount, specialization: ArtisanSpecialization) -> Decimal:
        ceiling = self._ceiling(specialization)
        if account.capacity_credit is None:
            return ceiling
        # A shock lowers the ceiling at once, so stored credit above the
        # shocked rate plus its carry is cut, not only the next refill.
        return min(account.capacity_credit, ceiling)

    def capacity_of(self, account: AgentAccount) -> int:
        """Recipe runs this artisan may still craft in the current macro-cycle."""
        specialization = self.specialization_of(account)
        if specialization is None:
            return 0
        return max(0, int(self._credit(account, specialization)))

    def refresh(self, account: AgentAccount) -> None:
        """Accrue one macro-cycle of capacity, at the close of the cycle."""
        specialization = self.specialization_of(account)
        if specialization is None or account.capacity_credit is None:
            return
        account.capacity_credit = min(
            account.capacity_credit + self.rate_of(specialization),
            self._ceiling(specialization),
        )

    @staticmethod
    def specialization_of(account: AgentAccount) -> ArtisanSpecialization | None:
        try:
            return ArtisanSpecialization(account.specialization)
        except ValueError:
            return None

    def products_for(self, specialization: ArtisanSpecialization) -> list[Product]:
        """Products this specialization can craft, in a fixed order."""
        return [
            product
            for product, recipe in self._config.recipes.items()
            if recipe.specialization is specialization
        ]

    def missing_inputs(self, account: AgentAccount, product: Product) -> dict[Resource, int]:
        """Inputs the artisan still lacks for one run of ``product``'s recipe."""
        recipe = self._config.recipes[product]
        return {
            resource: needed - account.available_quantity(resource)
            for resource, needed in sorted(recipe.inputs.items(), key=lambda kv: kv[0].value)
            if account.available_quantity(resource) < needed
        }

    # -- crafting ----------------------------------------------------------
    def produce(self, account: AgentAccount, product: Product, quantity: int) -> ProductionResult:
        """Craft ``quantity`` recipe runs, consuming the recipe's inputs once per run.

        The result's ``quantity`` is the units made: runs times ``output_quantity``.
        """
        if account.role is not Role.ARTISAN:
            raise ProductionRejected(f"{account.agent_id} is a {account.role}, not an artisan")
        if quantity <= 0:
            raise ProductionRejected(f"quantity must be positive, got {quantity}")

        recipe = self._config.recipes.get(product)
        if recipe is None:
            raise ProductionRejected(f"no recipe for {product}")

        specialization = self.specialization_of(account)
        if specialization is None or recipe.specialization is not specialization:
            raise ProductionRejected(
                f"{account.agent_id} ({account.specialization}) cannot craft {product}, "
                f"which requires {recipe.specialization}"
            )

        # Told in the prompt to stop, woodworkers still made 23 bows
        # for 18 sales and ended with their gold in unsold stock.
        limit = self._config.craft_stock_limit
        held = account.inventory.get(product, 0)
        if limit is not None and held >= limit:
            raise ProductionRejected(
                f"{account.agent_id} already holds {held} unsold {product}: no more can be "
                f"crafted until fewer than {limit} remain"
            )

        available_capacity = self.capacity_of(account)
        if quantity > available_capacity:
            raise ProductionRejected(
                f"{account.agent_id} has capacity for {available_capacity} more recipe "
                f"runs this cycle, not {quantity}"
            )

        consumed = {
            resource: needed * quantity
            for resource, needed in sorted(recipe.inputs.items(), key=lambda kv: kv[0].value)
        }
        for resource, needed in consumed.items():
            if account.available_quantity(resource) < needed:
                raise ProductionRejected(
                    f"{account.agent_id} needs {needed} {resource} but has "
                    f"{account.available_quantity(resource)} unreserved"
                )

        for resource, needed in consumed.items():
            account.consume(resource, needed)
        account.receive(product, recipe.output_quantity * quantity)
        account.capacity_credit = self._credit(account, specialization) - quantity

        return ProductionResult(
            agent_id=account.agent_id,
            product=product,
            quantity=recipe.output_quantity * quantity,
            consumed=consumed,
        )
