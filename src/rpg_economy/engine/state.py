"""Canonical world state.

This is the engine's ground truth (design v1.0 s7): the only place gold and
inventory are held, and the only place they change. LLM agents propose actions;
the engine validates them against this state and executes them.

Money is ``Decimal``, never ``float``. The design forbids selling without
inventory and buying without gold (s5.2), and those are invariants under test --
a balance of -1e-16 from float drift would be a spurious violation, and a
reserved amount that fails to release exactly would leak value out of the
economy over a 300-tick run.

Holdings are split into *total* and *reserved*. Placing an order reserves what it
would need; only the unreserved remainder can back a further order. Without that,
one unit of iron could back two sell orders and the no-naked-sell rule would hold
at placement but fail at execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from rpg_economy.domain import MONEY_QUANTUM, Good, Product, RiskLevel, Role

ZERO: Decimal = Decimal(0)


class EngineInvariantError(RuntimeError):
    """Raised when a state mutation would break an invariant of the canonical state.

    Never an agent's mistake: every agent-supplied value is validated before it
    reaches a mutator, so a mutator refusing one means the engine's own
    bookkeeping went wrong -- possibly after goods or gold already moved. It is
    deliberately not a ``ValueError``, so the executor cannot mistake it for a
    rejected action and carry on with corrupted state: it ends the run.
    """


class InsufficientFunds(Exception):
    """Raised when an operation would overdraw an account's unreserved gold."""


class InsufficientInventory(Exception):
    """Raised when an operation would oversell an account's unreserved goods."""


@dataclass
class ExpeditionState:
    """An expedition in progress (design v1.0 s5.3)."""

    zone_id: str
    risk: RiskLevel
    started_tick: int
    resolves_at_tick: int
    #: The success probability each carried item added, fixed at departure:
    #: the gear an adventurer sets out with is the gear its odds rest on.
    gear_bonus: dict[Product, Decimal] = field(default_factory=dict)
    #: The success the zone's crowding added (negative: taken), fixed at departure.
    crowding: Decimal = Decimal(0)


@dataclass
class AgentAccount:
    """One economic agent's canonical holdings.

    ``gold`` and ``inventory`` are totals *including* what is reserved against
    resting orders; ``available_*`` is what remains free to commit.
    """

    agent_id: str
    role: Role
    gold: Decimal
    specialization: str | None = None
    inventory: dict[Good, int] = field(default_factory=dict)
    reserved_gold: Decimal = ZERO
    reserved_inventory: dict[Good, int] = field(default_factory=dict)

    #: Gold immobilised by the liquidity_loss shock. Not confiscated: the shock
    #: reduces purchasing capacity, and the freeze lifts at the reversion tick.
    frozen_gold: Decimal = ZERO
    #: While a liquidity freeze is active, the fraction of pre-freeze reserved
    #: gold that is frozen when its bid is withdrawn (design s10, liquidity_loss).
    freeze_rate: Decimal = ZERO
    #: Gold that was reserved behind bids when the freeze began and is still
    #: reserved. Only this gold is frozen on release; it is drawn down whenever
    #: a pre-freeze bid spends, refunds or releases its reservation.
    freezable_reserved: Decimal = ZERO
    #: Tick the freeze began, or None. Bids placed at or after it reserved gold
    #: that was already subject to the freeze, so their releases freeze nothing.
    freeze_tick: int | None = None

    #: Wealth tax assessed but not yet paid, because the agent's gold was
    #: reserved or frozen at the collection. Never forgiven, and a lien on the
    #: agent's gold: it is left out of ``available_gold``, so owed gold cannot
    #: back a new bid or be spent, and it is paid out of ``free_gold`` at the
    #: start of each cycle, before any new wealth tax.
    tax_arrears: Decimal = ZERO

    #: Equipped items and their remaining durability, in ticks of wear.
    equipment: dict[Product, int] = field(default_factory=dict)

    #: Set while away on an expedition; None when in town.
    expedition: ExpeditionState | None = None

    #: Tick until which this agent rests after its last return (longer after a failure).
    recovering_until_tick: int = 0

    #: Production capacity left, in recipe runs. Accrues each macro-cycle at the
    #: specialization's rate (engine/production.py); None until first used,
    #: which reads as full.
    capacity_credit: Decimal | None = None

    # -- queries -----------------------------------------------------------
    @property
    def free_gold(self) -> Decimal:
        """Gold neither reserved behind a bid nor frozen, before the tax lien.

        This is what wealth-tax arrears are collected from: the lien never
        reaches into reserved or frozen gold. Never below zero.
        """
        return max(ZERO, self.gold - self.reserved_gold - self.frozen_gold)

    @property
    def available_gold(self) -> Decimal:
        """Gold free to commit: neither reserved, frozen, nor owed as tax arrears.

        Arrears are a lien, so gold freed by an expiring bid cannot be re-bid
        ahead of the next collection. Paying arrears lowers ``gold`` and
        ``tax_arrears`` together and leaves this unchanged. Never below zero.
        """
        return max(ZERO, self.free_gold - self.tax_arrears)

    def is_away(self, tick: int) -> bool:
        """Whether the agent cannot set out now: on an expedition, or resting after one."""
        return self.expedition is not None or tick < self.recovering_until_tick

    def is_resting(self, tick: int) -> bool:
        """Back in town but not yet able to set out again. It can still trade."""
        return self.expedition is None and tick < self.recovering_until_tick

    def durability_of(self, product: Product) -> int:
        return self.equipment.get(product, 0)

    def is_equipped_with(self, product: Product) -> bool:
        return self.durability_of(product) > 0

    def quantity(self, good: Good) -> int:
        return self.inventory.get(good, 0)

    def reserved(self, good: Good) -> int:
        return self.reserved_inventory.get(good, 0)

    def available_quantity(self, good: Good) -> int:
        return self.quantity(good) - self.reserved(good)

    # -- gold --------------------------------------------------------------
    def reserve_gold(self, amount: Decimal) -> None:
        if amount < ZERO:
            raise EngineInvariantError("cannot reserve a negative amount of gold")
        if amount > self.available_gold:
            raise InsufficientFunds(
                f"{self.agent_id} has {self.available_gold} unreserved gold, needs {amount}"
            )
        self.reserved_gold += amount

    def release_gold(self, amount: Decimal, *, placed_tick: int | None = None) -> None:
        """Return reserved gold to the agent, e.g. when a bid expires or is cancelled.

        Under a liquidity freeze, ``freeze_rate`` of whatever part of it was
        already reserved when the freeze began is frozen: that gold is subject
        to the freeze once it comes back, as if it had been free at the time.
        Gold reserved by a bid placed after the freeze (``placed_tick`` at or
        after ``freeze_tick``) was free gold the freeze already applied to, so
        releasing it freezes nothing. Without ``placed_tick`` the release is
        treated as pre-freeze, still bounded by ``freezable_reserved``.
        """
        self._unreserve(amount)
        drawn = self._draw_freezable(amount, placed_tick)
        if drawn > ZERO:
            self.frozen_gold += (drawn * self.freeze_rate).quantize(MONEY_QUANTUM)

    def refund_gold(self, amount: Decimal, *, placed_tick: int | None = None) -> None:
        """Return the unspent part of a reservation when a bid fills below its limit.

        Unlike ``release_gold``, never frozen: the bid was filled, not withdrawn,
        and the change belongs to a purchase committed before any freeze. It
        still leaves the pre-freeze reservation, so it is drawn down.
        """
        self._unreserve(amount)
        self._draw_freezable(amount, placed_tick)

    def _draw_freezable(self, amount: Decimal, placed_tick: int | None) -> Decimal:
        """Take up to ``amount`` out of ``freezable_reserved``; return what was taken."""
        if self.freezable_reserved <= ZERO:
            return ZERO
        placed_after_freeze = (
            placed_tick is not None
            and self.freeze_tick is not None
            and placed_tick >= self.freeze_tick
        )
        if placed_after_freeze:
            return ZERO
        drawn = min(amount, self.freezable_reserved)
        self.freezable_reserved -= drawn
        return drawn

    def _unreserve(self, amount: Decimal) -> None:
        if amount < ZERO:
            raise EngineInvariantError("cannot release a negative amount of gold")
        if amount > self.reserved_gold:
            raise EngineInvariantError(
                f"{self.agent_id} would release {amount} gold but only "
                f"{self.reserved_gold} is reserved"
            )
        self.reserved_gold -= amount

    def spend_reserved_gold(self, amount: Decimal, *, placed_tick: int | None = None) -> None:
        """Pay ``amount`` out of reserved gold, removing it from the account.

        A pre-freeze bid that fills spends its reservation unfrozen, drawing
        down ``freezable_reserved``.
        """
        self._unreserve(amount)
        self._draw_freezable(amount, placed_tick)
        self.gold -= amount
        if self.gold < ZERO:  # pragma: no cover -- release_gold already guards this
            raise InsufficientFunds(f"{self.agent_id} overdrew: {self.gold}")

    def credit_gold(self, amount: Decimal) -> None:
        if amount < ZERO:
            raise EngineInvariantError("cannot credit a negative amount of gold")
        self.gold += amount

    def freeze_gold(self, fraction: Decimal, *, tick: int | None = None) -> Decimal:
        """Freeze ``fraction`` of the agent's available gold now, and of reserved gold later.

        Available gold is the gold free at the shock net of the tax arrears
        owed. Gold reserved behind a resting bid stays reserved -- the bid can
        still fill, since it was committed before the freeze -- but that
        reservation is snapshotted in ``freezable_reserved``: if the bid
        expires or is cancelled, ``fraction`` of what it releases is frozen
        too. Freezing only what was free at the shock tick would let the
        shock's size depend on whether the agent happened to have a bid out,
        which differs between the conditions of one trio. Bids placed from
        ``tick`` on are not in the snapshot and freeze nothing on release.

        Gold owed as tax arrears is not frozen: it is already not the agent's
        to use, and freezing it would block its collection. Arrears are paid
        from free gold first, so the part of them free gold cannot cover is
        taken out of the snapshot: that much of the reservation will go to
        the arrears once released, not back to the agent. So if the bid is
        released (expires or is cancelled) during the freeze, the agent ends
        the freeze with the same gold frozen as one that held that gold free
        at the shock. If the bid fills instead, its reservation is spent
        unfrozen: the purchase was committed before the freeze, and the
        freeze takes nothing from it. Returns what was frozen immediately.
        """
        frozen = (self.available_gold * fraction).quantize(MONEY_QUANTUM)
        self.freeze_rate = fraction
        self.freeze_tick = tick
        arrears_on_reserved = max(ZERO, self.tax_arrears - self.free_gold)
        self.freezable_reserved = max(ZERO, self.reserved_gold - arrears_on_reserved)
        if frozen <= ZERO:
            return ZERO
        self.frozen_gold += frozen
        return frozen

    def unfreeze_all_gold(self) -> Decimal:
        """Lift the freeze entirely: frozen gold becomes free and nothing more freezes."""
        released, self.frozen_gold = self.frozen_gold, ZERO
        self.freeze_rate = ZERO
        self.freezable_reserved = ZERO
        self.freeze_tick = None
        return released

    # -- inventory ---------------------------------------------------------
    def reserve_quantity(self, good: Good, quantity: int) -> None:
        if quantity < 0:
            raise EngineInvariantError("cannot reserve a negative quantity")
        if quantity > self.available_quantity(good):
            raise InsufficientInventory(
                f"{self.agent_id} has {self.available_quantity(good)} unreserved "
                f"{good}, needs {quantity}"
            )
        self.reserved_inventory[good] = self.reserved(good) + quantity

    def release_quantity(self, good: Good, quantity: int) -> None:
        if quantity < 0:
            raise EngineInvariantError("cannot release a negative quantity")
        if quantity > self.reserved(good):
            raise EngineInvariantError(
                f"{self.agent_id} would release {quantity} {good} but only "
                f"{self.reserved(good)} is reserved"
            )
        self.reserved_inventory[good] = self.reserved(good) - quantity

    def deliver_reserved(self, good: Good, quantity: int) -> None:
        """Hand over ``quantity`` from reserved inventory, removing it."""
        self.release_quantity(good, quantity)
        remaining = self.quantity(good) - quantity
        if remaining < 0:  # pragma: no cover -- release_quantity already guards this
            raise InsufficientInventory(f"{self.agent_id} oversold {good}")
        self.inventory[good] = remaining

    def receive(self, good: Good, quantity: int) -> None:
        if quantity < 0:
            raise EngineInvariantError("cannot receive a negative quantity")
        self.inventory[good] = self.quantity(good) + quantity

    def consume(self, good: Good, quantity: int) -> int:
        """Remove up to ``quantity`` unreserved units; return how many went.

        Consumption never touches reserved stock: those units are already
        promised to a resting sell order.
        """
        taken = min(quantity, self.available_quantity(good))
        if taken <= 0:
            return 0
        self.inventory[good] = self.quantity(good) - taken
        return taken


@dataclass
class WorldState:
    """The canonical state of the simulated economy."""

    accounts: dict[str, AgentAccount] = field(default_factory=dict)
    tick: int = 0
    cycle: int = 0

    def account(self, agent_id: str) -> AgentAccount:
        try:
            return self.accounts[agent_id]
        except KeyError:
            raise KeyError(f"unknown agent {agent_id!r}") from None

    @property
    def agent_ids(self) -> list[str]:
        """Agent ids in a fixed order, so iteration never depends on insertion luck."""
        return sorted(self.accounts)

    def total_gold(self) -> Decimal:
        """Total gold in the economy. Trade moves it; it is never created here."""
        return sum((a.gold for a in self.accounts.values()), ZERO)

    def total_quantity(self, good: Good) -> int:
        """Total units of ``good`` in the economy, across all accounts."""
        return sum(a.quantity(good) for a in self.accounts.values())

    def agents_with_role(self, role: Role) -> list[str]:
        return [a for a in self.agent_ids if self.accounts[a].role is role]
