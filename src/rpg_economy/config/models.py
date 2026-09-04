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
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rpg_economy.domain import (
    AdventurerSubclass,
    ArtisanSpecialization,
    Product,
    Resource,
    RiskLevel,
    Role,
    Scenario,
    parse_good,
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


class ObservabilityConfig(StrictModel):
    """What each agent may see of the market (design v1.0 s5.2).

    An agent observes its own orders and trades, best bid/ask, and the last N
    trades of the goods in its **fixed interest list**. It never sees others'
    gold, inventories or plans, nor any aggregate.
    """

    interest_lists: dict[str, list[str]] = Field(
        description="Goods each role/subclass follows, keyed by subclass or "
        "specialization where one exists, otherwise by role. Fixed for the run.",
    )
    director_recent_results: int = Field(
        default=5,
        ge=0,
        description="Recent RESULTs the Director sees and may cite as TASK context "
        "(design v1.0 s3.1). pilot_calibrated.",
    )
    own_history: int = Field(
        default=5,
        ge=0,
        description="The agent's own most recent actions and their engine outcome, and "
        "its own most recent trades, shown at each activation (design v1.0 s5.2: an "
        "agent observes its own history, orders and trades). pilot_calibrated.",
    )

    @model_validator(mode="after")
    def _interest_lists_name_goods(self) -> ObservabilityConfig:
        """Every followed good must be a real one, checked at load.

        An unknown name would otherwise fail only when an agent's observation is
        built, halfway through a run, or be silently shown nothing.
        """
        for key, goods in sorted(self.interest_lists.items()):
            for good in goods:
                try:
                    parse_good(good)
                except ValueError:
                    raise ValueError(
                        f"observability.interest_lists.{key} names an unknown good {good!r}"
                    ) from None
        return self

    def interest_of(self, role: str, specialization: str | None) -> list[str]:
        """The interest list governing an agent, most specific key first."""
        if specialization and specialization in self.interest_lists:
            return self.interest_lists[specialization]
        return self.interest_lists.get(role, [])


class MarketConfig(StrictModel):
    """Central limit-order book (design v1.0 s5.2)."""

    order_expiry_ticks: int = Field(
        ge=1, description="Ticks after which a resting order expires. pilot_calibrated."
    )
    visible_last_n_trades: int = Field(
        ge=1,
        description="N: recent trades visible per good in an agent's fixed interest list.",
    )
    max_orders_per_activation: int = Field(
        default=4,
        ge=1,
        description="Most buy, sell and cancel_order orders one place_orders action "
        "may carry. The same in every condition. pilot_calibrated.",
    )
    price_step_pct: Decimal | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="The step a price signal suggests: a seller with no buyer is told to "
        "ask about this much less, a buyer with no seller to bid about this much more. "
        "None states the direction only. pilot_calibrated.",
    )
    usual_price_ticks: int | None = Field(
        default=None,
        ge=1,
        description="Window of the typical (median) price an agent is told for a good "
        "when the latest trade is far from it. None hides it. pilot_calibrated.",
    )
    usual_price_gap_pct: Decimal = Field(
        default=Decimal("0.25"),
        gt=0,
        description="How far the latest trade must be from the typical price, as a "
        "share of it, before the typical price is stated. pilot_calibrated.",
    )
    usual_price_min_trades: int = Field(
        default=5,
        ge=1,
        description="Below this many trades in the window, the typical price is the "
        "good's price guide value. pilot_calibrated.",
    )
    recent_sale_ticks: int | None = Field(
        default=None,
        ge=1,
        description="A good sold within this many ticks is still coming: a buyer whose "
        "bids lapsed is told the sale's price, not a step above it, since sellers come "
        "back between trips. None: always a step above. pilot_calibrated.",
    )
    input_margin_pct: Decimal = Field(
        default=Decimal(0),
        ge=0,
        description="Margin an artisan keeps over an input's cost in its input "
        "ceiling: the most the input is worth leaves the product this share above "
        "its inputs. pilot_calibrated.",
    )
    allow_self_trade: bool = Field(
        default=False,
        description="Whether an agent's orders may match each other. False per "
        "design v1.0 s5.2: a wash trade moves nothing but would inflate traded "
        "volume, the primary target of the liquidity_loss scenario.",
    )


class ZoneConfig(StrictModel):
    """One expedition zone: a difficulty and a resource distribution (design s5.3)."""

    id: str
    risk: RiskLevel
    spawn_per_tick: dict[Resource, Decimal] = Field(
        description="Units of each resource the zone accrues per tick. This is the "
        "'aparicion' the resource_scarcity shock reduces. pilot_calibrated.",
    )
    gold_per_tick: Decimal | None = Field(
        default=None,
        ge=0,
        description="Gold the zone accrues per tick, held up to pool_capacity_ticks of it "
        "and shared by the successes that draw on it like its resources. None: every "
        "success brings its full gold. pilot_calibrated.",
    )


class SubclassExpeditionConfig(StrictModel):
    """A subclass's expedition advantages (design v1.0 s4.1)."""

    success_bonus: dict[RiskLevel, Decimal] = Field(default_factory=dict)
    resource_affinity: dict[Resource, Decimal] = Field(
        default_factory=dict,
        description="Multiplier on the draw weight of resources this subclass is "
        "characteristically good at gathering.",
    )
    loot_multiplier: Decimal = Field(
        default=Decimal(1),
        ge=0,
        description="Multiplier on the gold a successful expedition brings back. A "
        "subclass whose gear costs more per trip (the mage burns a mana potion on "
        "every one) earns more per trip. pilot_calibrated.",
    )


class CrowdingConfig(StrictModel):
    """Success lost in a zone more adventurers set out for, and gained in one fewer do.

    A zone's modifier is ``per_trip`` times how far its trips over the last
    ``window_ticks`` fall short of an even share of all trips (negative when it
    gets more than its share), within plus or minus ``cap``. Fixed at departure.
    """

    window_ticks: int = Field(ge=1)
    per_trip: Decimal = Field(gt=0)
    cap: Decimal = Field(gt=0, lt=1)


class ExpeditionConfig(StrictModel):
    """Expeditions and wear (design v1.0 s5.3). All values pilot_calibrated."""

    duration_ticks: dict[RiskLevel, int]
    recovery_ticks_on_failure: dict[RiskLevel, int]
    success_probability: dict[RiskLevel, Decimal]
    expected_yield: dict[RiskLevel, int]
    yield_spread: dict[RiskLevel, int] = Field(
        description="Half-width of the uniform yield draw. Higher risk carries higher "
        "expected reward AND higher variance (design v1.0 s5.3).",
    )
    partial_yield_on_failure: Decimal = Field(
        ge=0,
        le=1,
        description="Fraction of the expected yield recovered on failure. Failure is "
        "partial loss and recovery time, never permanent death (design s5.3, s14).",
    )
    durability_loss_per_expedition: int = Field(ge=0)
    rest_ticks: int = Field(
        ge=0,
        description="Ticks after returning from any expedition before the adventurer "
        "can set out again. It can trade meanwhile, and its orders stay on the book "
        "while it is away. After a failure, the risk's recovery_ticks_on_failure is added. "
        "pilot_calibrated.",
    )
    gold_loot: dict[RiskLevel, Decimal] = Field(
        default_factory=dict,
        description="Gold a successful expedition brings back besides materials, by "
        "zone risk. With the taxes, what keeps gold circulating: adventurers "
        "otherwise only earn from the artisans they buy from, and a margin drains them "
        "for good. pilot_calibrated.",
    )
    min_success_probability: Decimal = Field(gt=0, le=1)
    max_success_probability: Decimal = Field(gt=0, le=1)
    pool_capacity_ticks: int = Field(
        ge=1,
        description="A zone pool holds at most spawn_per_tick x this many ticks of a "
        "resource, so scarcity bites instead of being absorbed by an unbounded stock.",
    )
    gold_capacity_ticks: int | None = Field(
        default=None,
        ge=1,
        description="A zone's gold pool holds at most gold_per_tick x this many ticks; "
        "None uses pool_capacity_ticks. pilot_calibrated.",
    )
    crowding: CrowdingConfig | None = Field(
        default=None,
        description="Success shifted by how crowded a zone has been; None turns it off. "
        "pilot_calibrated.",
    )
    zones: list[ZoneConfig]
    subclasses: dict[AdventurerSubclass, SubclassExpeditionConfig] = Field(default_factory=dict)
    gear: dict[AdventurerSubclass, dict[Product, Decimal]] = Field(
        description="What each subclass can carry on an expedition and the success "
        "probability each item adds (design v1.0 s4.1, s5.3). Items with durability "
        "are equipped and wear out; the others (potions) are used up on setting out. "
        "pilot_calibrated.",
    )
    starting_gear: dict[AdventurerSubclass, Product] = Field(
        default_factory=dict,
        description="The durable item each adventurer of a subclass starts with, "
        "equipped at full durability. Nothing is required to set out: gear raises the "
        "chance of success and the gold a success brings (gear_gold_bonus), and once "
        "it breaks an adventurer may buy another. pilot_calibrated.",
    )
    gear_gold_bonus: dict[Product, Decimal] = Field(
        default_factory=dict,
        description="Share added to a successful trip's gold for each item carried "
        "(equipped with durability left, or a potion taken along): equipment pays "
        "for itself in the gold it brings back, not only in the odds, so buying it "
        "is worth it in every zone. pilot_calibrated.",
    )

    @model_validator(mode="after")
    def _probability_bounds_ordered(self) -> ExpeditionConfig:
        if self.min_success_probability >= self.max_success_probability:
            raise ValueError("min_success_probability must be below max_success_probability")
        return self

    @model_validator(mode="after")
    def _zone_risks_are_parameterised(self) -> ExpeditionConfig:
        """Every risk level a zone uses has an entry in each map keyed by risk.

        A missing entry would otherwise surface as a KeyError the first time an
        adventurer sets out for, or returns from, such a zone. ``gold_loot`` and
        subclass ``success_bonus`` are optional and default to zero.
        """
        used = sorted({zone.risk for zone in self.zones}, key=lambda r: r.value)
        maps: dict[str, dict[RiskLevel, object]] = {
            "duration_ticks": self.duration_ticks,
            "recovery_ticks_on_failure": self.recovery_ticks_on_failure,
            "success_probability": self.success_probability,
            "expected_yield": self.expected_yield,
            "yield_spread": self.yield_spread,
        }
        for name, table in maps.items():
            missing = [risk.value for risk in used if risk not in table]
            if missing:
                raise ValueError(f"expeditions.{name} has no entry for zone risk levels {missing}")
        return self

    @model_validator(mode="after")
    def _zone_ids_unique(self) -> ExpeditionConfig:
        ids = [zone.id for zone in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError(f"zone ids must be unique, got {ids}")
        if not ids:
            raise ValueError("at least one expedition zone is required")
        return self


# ---------------------------------------------------------------------------
# Shocks
# ---------------------------------------------------------------------------
class ResourceScarcityConfig(StrictModel):
    """Reduce a resource's appearance rate (design v1.0 s10). Primary scenario."""

    target_resource: Resource = Resource.IRON
    spawn_multiplier: Decimal = Field(
        ge=0, lt=1, description="Factor applied to the target's zone spawn rates."
    )
    stock_multiplier: Decimal | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Factor applied, at the shock, to the target already in the zone "
        "pools, so a stock built while a zone went unvisited does not cover the "
        "shock. None leaves the pools as they are.",
    )
    held_keep: int | None = Field(
        default=None,
        ge=0,
        description="Units of the target agents may keep in all, unreserved, at the "
        "shock; the rest is bought out, from the largest holders first, so a stock "
        "gathered before the shock does not cover it. Units on resting sell orders "
        "are left alone. None leaves holdings as they are.",
    )


class DemandSurgeConfig(StrictModel):
    """Multiply the provisions of a product each expedition uses (design v1.0 s10)."""

    target_product: Product = Product.HEALTH_POTION
    consumption_multiplier: Decimal = Field(gt=1)


class ProductivityDropConfig(StrictModel):
    """Reduce an artisan specialization's capacity (design v1.0 s10)."""

    target_specialization: ArtisanSpecialization = ArtisanSpecialization.SMITHING
    capacity_multiplier: Decimal = Field(ge=0, lt=1)


class LiquidityLossConfig(StrictModel):
    """Reduce a subset of agents' purchasing capacity (design v1.0 s10).

    Implemented by *freezing* gold, not confiscating it. The design says
    "reducir capacidad de compra", which is a liquidity constraint: the shock
    models illiquidity, not a loss of wealth (taxes and storage are the gold
    sinks). The fraction applies to the available gold at the shock tick (free
    gold net of tax arrears owed) and, when released during the shock, to the
    gold then reserved behind bids, net of the arrears it will pay; all of it
    is released at the reversion tick.

    Who is affected is specified, never inferred (design v1.0 s10). Give either
    an explicit `affected_agents` list or a role-stratified `affected_by_role`
    count; exactly one. Both are configuration, so a seed-trio necessarily
    freezes the same agents, and the subset never depends on how agent ids sort.
    """

    frozen_gold_fraction: Decimal = Field(gt=0, le=1)
    affected_agents: list[str] | None = Field(
        default=None, description="Explicit agent ids to freeze. pilot_calibrated."
    )
    affected_by_role: dict[Role, Annotated[int, Field(ge=0)]] | None = Field(
        default=None,
        description="How many agents of each role to freeze, taken in id order "
        "within that role. pilot_calibrated.",
    )

    @model_validator(mode="after")
    def _exactly_one_specification(self) -> LiquidityLossConfig:
        given = [self.affected_agents is not None, self.affected_by_role is not None]
        if sum(given) != 1:
            raise ValueError(
                "liquidity_loss needs exactly one of affected_agents or affected_by_role"
            )
        if self.affected_agents is not None and not self.affected_agents:
            raise ValueError("affected_agents must not be empty")
        if self.affected_by_role is not None and not any(self.affected_by_role.values()):
            raise ValueError("affected_by_role must freeze at least one agent")
        return self


class ShockConfig(StrictModel):
    """Per-scenario shock parameters.

    Timing is not configured here: it derives from the phase structure, so a
    seed-trio necessarily receives the same shock at the same tick (design v1.0
    s10). Every shock has an explicit finite duration and a reversion tick.
    """

    resource_scarcity: ResourceScarcityConfig
    demand_surge: DemandSurgeConfig
    productivity_drop: ProductivityDropConfig
    liquidity_loss: LiquidityLossConfig


# ---------------------------------------------------------------------------
# Triggers and scheduler (Conditions B and C only)
# ---------------------------------------------------------------------------
class TriggerConfig(StrictModel):
    """T1-T5 thresholds (design v1.0 s8.2).

    Identical in B and C -- this is a binding control (design v1.0 s12). No LLM
    is ever consulted while evaluating these.
    """

    t3_price_move_pct: Decimal = Field(
        gt=0,
        description="T3: relative transaction-price move of a good in the agent's fixed "
        "interest list since its last activation. pilot_calibrated.",
    )
    t4_gold_low: Decimal = Field(
        ge=0,
        description="T4 lower bound on available gold (not reserved, not frozen, "
        "not owed as tax arrears). pilot_calibrated.",
    )
    t4_gold_high: Decimal = Field(
        gt=0,
        description="T4 upper bound on available gold (not reserved, not frozen, "
        "not owed as tax arrears). pilot_calibrated.",
    )
    t4_durability_critical_pct: Decimal = Field(
        gt=0, le=1, description="T4 critical equipment durability. pilot_calibrated."
    )
    t5_inactivity_ticks: int = Field(
        ge=1, description="N: ticks of inactivity before T5 fires. pilot_calibrated."
    )

    @model_validator(mode="after")
    def _gold_bounds_ordered(self) -> TriggerConfig:
        if self.t4_gold_low >= self.t4_gold_high:
            raise ValueError("t4_gold_low must be strictly below t4_gold_high")
        return self


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class MetricsConfig(StrictModel):
    """Metric parameters (design v1.0 s11)."""

    volatility_window_w: int = Field(
        ge=2,
        description="W: window for price standard deviation, and the base-phase "
        "window whose mean is the recovery reference value (design v1.0 s11.1).",
    )
    recovery_band_pct_x: Decimal = Field(
        gt=0,
        description="X: half-width of the recovery band, relative to the per-run "
        "base-phase reference value.",
    )
    recovery_hold_ticks_m: int = Field(
        ge=1, description="M: consecutive ticks inside the band required to declare recovery."
    )
    useful_production_ticks_t: int = Field(
        ge=1, description="T: ticks within which produced units must be sold or consumed."
    )
    unanswered_request_ticks_d: int = Field(
        ge=1, description="D: ticks after which a REQUEST counts as unanswered."
    )


# ---------------------------------------------------------------------------
# LLM and budgets
# ---------------------------------------------------------------------------
class ModelPricing(StrictModel):
    """Provider list price, for cost projection only (IMPLEMENTATION.md section 7).

    Never used by a run: the simulation's behaviour cannot depend on what the
    tokens cost. It exists so that a real-provider batch can be priced before
    it is launched.
    """

    input_per_1m_tokens: Decimal = Field(
        ge=0, description="List price per 1,000,000 input tokens, in `currency`."
    )
    output_per_1m_tokens: Decimal = Field(
        ge=0, description="List price per 1,000,000 output tokens, in `currency`."
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)
    quoted_on: str = Field(
        description="ISO date the price was taken from the provider's page. Prices "
        "move; an estimate is only as good as the date attached to it."
    )


class LLMConfig(StrictModel):
    """Provider settings.

    Recorded verbatim in run metadata (design v1.0 s15). The engine never reads
    this; only ``rpg_economy.llm.client`` does.
    """

    provider: Literal["openai"] = Field(default="openai")
    model: str = Field(description="Exact model identifier. Constant across all conditions.")
    temperature: float = Field(
        ge=0.0,
        le=2.0,
        description="Fixed across all conditions (design v1.0 s11.4). The gpt-5 family "
        "accepts only the provider default of 1.0.",
    )
    reasoning_effort: str | None = Field(
        description="Reasoning budget for models that reason before answering. Fixed "
        "and identical across all conditions and both experiments. "
        "None omits the parameter, which is correct for non-reasoning models.",
    )
    max_output_tokens: int = Field(
        ge=1,
        description="Completion cap, sent as max_completion_tokens. Must leave room "
        "for a reasoning model's hidden reasoning tokens plus the answer.",
    )
    model_snapshot: str | None = Field(
        default=None,
        description="The dated model the provider actually resolved the alias to, "
        "observed from a real call. Recorded in run metadata, because an alias "
        "can change what it points to.",
    )
    seed: int | None = Field(
        default=None, description="Provider-side seed when supported (design v1.0 s11.4)."
    )
    request_timeout_s: float = Field(
        default=60.0,
        gt=0,
        description="Seconds before one provider request is abandoned and treated as a "
        "transient failure.",
    )
    max_transient_retries: int = Field(
        default=5,
        ge=0,
        description="Retries for rate limits, timeouts, connection errors and 5xx before "
        "the run stops. Never applied to authentication, quota or request errors.",
    )
    retry_base_delay_s: float = Field(
        default=2.0, gt=0, description="First backoff delay; doubles per retry, jittered."
    )
    retry_max_delay_s: float = Field(default=60.0, gt=0, description="Cap on one backoff delay.")
    mock: bool = Field(
        default=True,
        description="Use the deterministic mock client. Default True so that no run "
        "reaches the network by accident. An execution mode, not part of the setup: it "
        "is excluded from the config hash and switched on the command line (--real).",
    )
    pricing: ModelPricing | None = Field(
        default=None,
        description="List price for cost projection. Required before any real-provider "
        "batch; never read during a run.",
    )


class BudgetConfig(StrictModel):
    """Token budget for Experiment 2 (design v1.0 s8.4)."""

    experiment_2_total_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Pooled median of per-run token totals across all resource_scarcity "
        "pilot runs of A, B and C under Experiment 1 (design v1.0 s8.4). Computed by "
        "script before the final runs, never hand-tuned. Null until derived.",
    )
    derived_from_mock: bool = Field(
        default=False,
        description="True when experiment_2_total_tokens was derived from offline "
        "mock-client pilot logs. Such a budget measures the mock, not the model, so "
        "real-provider runs refuse it. Part of the config hash: a mock-derived and a "
        "real-derived budget are different setups.",
    )
    derived_code_digest: str | None = Field(
        default=None,
        description="Code digest of the pilot runs the budget was derived from, written "
        "by derive-budget. A real run refuses a budget derived under other code: the "
        "budget is only fair if the final runs spend it the way the pilot did.",
    )


class CampaignConfig(StrictModel):
    """Which cells of the pre-registered matrix this campaign runs (s11.4).

    In config rather than as a code constant so that the scope is part of the
    config hash (design v1.0 s10).
    """

    secondary_scenarios: list[Scenario] = Field(
        default_factory=list,
        description="Secondary scenarios in scope. Experiment 1 only, per s11.4.",
    )
    primary_only: bool = Field(
        default=False,
        description="A reduced campaign: the primary scenario under Experiment 1 only, "
        "with any number of final seeds (the design's seed floors become warnings). "
        "Seeds can be added later; a resumed campaign skips the trios already run.",
    )
    secondary_final_seeds: int = Field(
        default=8,
        ge=1,
        description="Final seeds each secondary scenario runs: the first N of seeds.final "
        "(design v1.0 s11.4, s13).",
    )
    pilot_calibration_horizon: int = Field(
        ge=1,
        description="Horizon of the pilot's calibration stage, compressed to save cost; "
        "its phases scale in proportion. The budget-derivation stage runs the full horizon.",
    )

    @model_validator(mode="after")
    def _secondary_scope_is_sound(self) -> CampaignConfig:
        if _PRIMARY_SCENARIO in self.secondary_scenarios:
            raise ValueError(
                f"campaign.secondary_scenarios must not include the primary scenario "
                f"{_PRIMARY_SCENARIO.value}; it always runs"
            )
        duplicates = _duplicates(self.secondary_scenarios)
        if duplicates:
            raise ValueError(
                "campaign.secondary_scenarios has duplicates: "
                + ", ".join(s.value for s in duplicates)
            )
        return self
