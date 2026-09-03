"""Configuration schema.

Every tunable named in design v1.0 lives here and is supplied from versioned
YAML under ``configs/``. Nothing in this schema may be duplicated as a literal
elsewhere in the code (IMPLEMENTATION.md section 3).

Values the design deliberately leaves to pilot calibration carry a
``pilot_calibrated`` marker in their field description and a matching comment in
``configs/base.yaml``. They must be fixed before the final runs (design v1.0
section 15).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rpg_economy.domain import (
    AdventurerSubclass,
    ArtisanSpecialization,
    Product,
    Resource,
    Role,
    Scenario,
)


class StrictModel(BaseModel):
    """Base for every config model: unknown keys are an error.

    A silently ignored key is how an experiment ends up running under parameters
    nobody intended, so a typo must fail loudly at load time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


#: The pre-registered primary scenario (design v1.0 s11.4). Mirrors
#: ``campaign.plan.PRIMARY_SCENARIO``, which imports this module and so cannot
#: be imported here; a test keeps the two equal.
_PRIMARY_SCENARIO = Scenario.RESOURCE_SCARCITY


def _duplicates[T](items: list[T]) -> list[T]:
    """Items that occur more than once, each listed once, in first-seen order."""
    seen: set[T] = set()
    repeated: list[T] = []
    for item in items:
        if item in seen and item not in repeated:
            repeated.append(item)
        seen.add(item)
    return repeated


# ---------------------------------------------------------------------------
# Time and activation budget
# ---------------------------------------------------------------------------
class TimeConfig(StrictModel):
    """Macro-cycle structure (design v1.0 s8.1)."""

    ticks_per_cycle: int = Field(
        default=1,
        ge=1,
        description="Mechanical ticks per macro-cycle. pilot_calibrated. Identical "
        "across conditions and experiments. At the default of 1, tick == macro-cycle.",
    )
    k_activations: int = Field(
        ge=1,
        description="K: max economic-agent activations per macro-cycle. Identical in "
        "A/B/C and in both experiments. pilot_calibrated (design v1.0 s13).",
    )
    horizon_ticks: int = Field(
        ge=1, description="Planned run horizon. Reached even after budget exhaustion."
    )

    @model_validator(mode="after")
    def _horizon_divides_into_whole_cycles(self) -> TimeConfig:
        if self.horizon_ticks % self.ticks_per_cycle:
            raise ValueError(
                f"horizon_ticks ({self.horizon_ticks}) must be a whole number of "
                f"macro-cycles of {self.ticks_per_cycle} ticks; a partial trailing "
                f"cycle would give its agents fewer activation opportunities"
            )
        return self

    @property
    def total_cycles(self) -> int:
        return self.horizon_ticks // self.ticks_per_cycle


class PhaseConfig(StrictModel):
    """Base / shock / recovery structure (design v1.0 s10)."""

    base_ticks: int = Field(ge=1, description="Ticks before the shock is applied.")
    shock_ticks: int = Field(
        ge=1,
        description="Shock duration. Every shock has an explicit finite duration; the "
        "reversion tick is the 'end of shock' for recovery-time measurement.",
    )
    recovery_ticks: int = Field(ge=1, description="Ticks observed after the reversion tick.")

    @property
    def shock_tick(self) -> int:
        """The tick at which the main shock is applied."""
        return self.base_ticks

    @property
    def reversion_tick(self) -> int:
        """The tick at which the shock ends -- the origin for recovery time.

        Every shock has an explicit finite duration, and "end of shock" is this
        tick (design v1.0 s10).
        """
        return self.base_ticks + self.shock_ticks


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
class PopulationConfig(StrictModel):
    """Agent population (design v1.0 s13). Identical across a seed-trio."""

    adventurers: dict[AdventurerSubclass, int]
    artisans: dict[ArtisanSpecialization, int]

    @property
    def total_agents(self) -> int:
        return sum(self.adventurers.values()) + sum(self.artisans.values())

    def members(self) -> list[tuple[str, Role, str]]:
        """Every agent as ``(agent_id, role, subclass or specialization)``, in creation order.

        The one place agent ids are minted (``<subclass>_<n>``,
        ``<specialization>_<n>``), shared by world construction and by config
        validation. It iterates the enums, not the config mapping, so the order
        is fixed by the domain vocabulary rather than by YAML key order.
        """
        members: list[tuple[str, Role, str]] = []
        for subclass in AdventurerSubclass:
            for index in range(1, self.adventurers.get(subclass, 0) + 1):
                members.append((f"{subclass.value}_{index}", Role.ADVENTURER, subclass.value))
        for specialization in ArtisanSpecialization:
            for index in range(1, self.artisans.get(specialization, 0) + 1):
                members.append(
                    (f"{specialization.value}_{index}", Role.ARTISAN, specialization.value)
                )
        return members


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------
class RecipeConfig(StrictModel):
    """One crafting recipe (design v1.0 s5.1). Quantities are pilot_calibrated."""

    inputs: dict[Resource, Annotated[int, Field(ge=1)]] = Field(
        min_length=1,
        description="Units of each resource one run consumes; at least one each, "
        "since a zero input would be no input at all and breaks batch sizing. A "
        "recipe with no inputs would make goods out of nothing.",
    )
    output_quantity: int = Field(default=1, ge=1)
    specialization: ArtisanSpecialization


class TaxConfig(StrictModel):
    """Taxes: the gold that leaves the economy, balancing expedition loot.

    A profit tax on finished products, paid by the seller on what a sale brings
    in above the cost of the product's inputs at their last traded prices (a
    sale at or below that cost pays nothing), and a periodic tax on personal
    wealth: every so many ticks each agent pays a share of the gold it holds
    above an exempt minimum. Nobody at or below the minimum pays, and whoever
    holds more pays more.
    """

    profit_rate: Decimal = Field(
        default=Decimal(0),
        ge=0,
        lt=1,
        description="Share of the profit of every sale of a finished product that the "
        "seller pays: the sale's value less the product's input cost (each input at its "
        "last traded price). Taxing profit rather than revenue spares an artisan whose "
        "margin is thin. Raw materials are not taxed. pilot_calibrated.",
    )
    wealth_rate: Decimal = Field(
        ge=0,
        lt=1,
        description="Share of an agent's gold above wealth_exemption paid each time the "
        "wealth tax is collected. pilot_calibrated.",
    )
    wealth_every_ticks: int = Field(
        default=50,
        ge=1,
        description="Ticks between wealth-tax collections. pilot_calibrated.",
    )
    wealth_exemption: Decimal = Field(
        ge=0,
        description="Gold below which no wealth tax is paid; above it, only the excess "
        "is taxed. pilot_calibrated.",
    )


class EconomyConfig(StrictModel):
    recipes: dict[Product, RecipeConfig]
    initial_gold: dict[str, Decimal] = Field(
        description="Starting gold per role key. pilot_calibrated."
    )
    initial_inventory: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Starting inventory per role or specialization key, good -> units; "
        "a specialization's entry replaces its role's. Part of the "
        "endowment held constant across a seed-trio (design v1.0 s12). pilot_calibrated.",
    )
    initial_reference_price: dict[str, Decimal] = Field(
        description="Fallback reference price per good, used for Gini valuation of goods "
        "untraded during the base phase (design v1.0 s11.1).",
    )
    durability: dict[Product, Annotated[int, Field(ge=1)]] = Field(
        default_factory=dict,
        description="Equipment durability in points; each expedition costs "
        "durability_loss_per_expedition, and nothing wears in town. Sustains recurring "
        "demand (design v1.0 s5.3). pilot_calibrated.",
    )
    taxes: TaxConfig = Field(
        description="The taxes that take gold out of the economy, balancing the loot "
        "expeditions bring in.",
    )
    storage_cost_per_unit: dict[Role, Decimal] = Field(
        default_factory=dict,
        description="Gold per unit of goods held, per macro-cycle, by role; it leaves "
        "the economy. What makes holding stock cost something: without it an artisan "
        "loses nothing by sitting on goods at a price nobody pays. "
        "pilot_calibrated.",
    )
    charge_floor: Decimal = Field(
        ge=0,
        description="Gold an agent keeps that storage is never charged below: enough "
        "for its next purchase, so the charge cannot put a producer out of "
        "business. Gold backing a resting bid counts toward it; gold owed as tax "
        "arrears or frozen by a liquidity shock does not. pilot_calibrated.",
    )
    craft_stock_limit: int | None = Field(
        default=None,
        ge=1,
        description="Units of a product an artisan may hold, its own sell orders included, "
        "and still craft more of it; at or above it the engine refuses the craft until "
        "some sell. None for no limit. pilot_calibrated.",
    )
    floor_at_initial_gold: bool = Field(
        default=False,
        description="When true, storage and the profit tax never take an agent below "
        "its role's initial gold (or charge_floor, if higher): nobody is charged into "
        "poverty. The wealth tax keeps its own exemption.",
    )
    production_capacity_per_cycle: dict[ArtisanSpecialization, Decimal] = Field(
        description="Recipe runs (batches) an artisan of each specialization can craft "
        "per macro-cycle; each run yields the recipe's output_quantity. May be "
        "fractional: a rate below one accrues from cycle to cycle, so one run every "
        "few cycles. The productivity_drop shock scales this. pilot_calibrated.",
    )

    @model_validator(mode="after")
    def _capacity_is_non_negative(self) -> EconomyConfig:
        negative = sorted(
            s.value for s, rate in self.production_capacity_per_cycle.items() if rate < 0
        )
        if negative:
            raise ValueError(f"negative production capacity for {negative}")
        negative = sorted(r.value for r, cost in self.storage_cost_per_unit.items() if cost < 0)
        if negative:
            raise ValueError(f"negative storage cost for {negative}")
        return self


class TraitsConfig(StrictModel):
    """An agent's behavioural traits (design v1.0 s7).

    Four dimensions, each in [0, 1]. They enter the prompt and shape how an agent
    reasons; they never touch engine validation. Fixed across a seed-trio, which
    is why they are configuration rather than a draw: design s12 requires the
    same agents with the same traits in matched runs, and stating them keeps that
    auditable rather than reproducible-only-by-replay.
    """

    risk_tolerance: Decimal = Field(ge=0, le=1)
    time_orientation: Decimal = Field(
        ge=0, le=1, description="0 = immediate payoff, 1 = long horizon."
    )
    commercial_aggressiveness: Decimal = Field(ge=0, le=1)
    adaptability: Decimal = Field(ge=0, le=1)


class AgentsConfig(StrictModel):
    """Per-agent behavioural configuration."""

    traits: dict[str, TraitsConfig] = Field(
        description="Traits keyed by subclass or specialization where one exists, "
        "otherwise by role. pilot_calibrated.",
    )
    trait_overrides: dict[str, TraitsConfig] = Field(
        default_factory=dict,
        description="Per-agent-id overrides, so two warriors need not be identical.",
    )

    def traits_of(self, agent_id: str, role: str, specialization: str | None) -> TraitsConfig:
        """The traits governing one agent, most specific source first."""
        if agent_id in self.trait_overrides:
            return self.trait_overrides[agent_id]
        if specialization and specialization in self.traits:
            return self.traits[specialization]
        try:
            return self.traits[role]
        except KeyError:
            raise KeyError(
                f"no traits configured for {agent_id!r} (role {role!r}, "
                f"specialization {specialization!r})"
            ) from None
