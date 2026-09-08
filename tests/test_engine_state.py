"""World state, escrow accounting, and the seeded random streams."""

from __future__ import annotations

from decimal import Decimal

import pytest

from rpg_economy.domain import Resource, Role
from rpg_economy.engine import AgentAccount, RunRandom, build_world, derive_seed
from rpg_economy.engine.rng import STREAM_EXPEDITION, STREAM_SCHEDULER, STREAM_SPAWN
from rpg_economy.engine.state import (
    ZERO,
    EngineInvariantError,
    InsufficientFunds,
    InsufficientInventory,
)
from tests.conftest import make_config

IRON = Resource.IRON


@pytest.fixture
def account() -> AgentAccount:
    return AgentAccount(
        agent_id="a", role=Role.ARTISAN, specialization="smithing", gold=Decimal("100")
    )


# ---------------------------------------------------------------------------
# Escrow
# ---------------------------------------------------------------------------
def test_reserving_gold_reduces_only_what_is_available(account: AgentAccount) -> None:
    account.reserve_gold(Decimal("30"))
    assert account.gold == Decimal("100")
    assert account.available_gold == Decimal("70")


def test_cannot_reserve_more_gold_than_available(account: AgentAccount) -> None:
    with pytest.raises(InsufficientFunds):
        account.reserve_gold(Decimal("101"))


def test_cannot_reserve_more_inventory_than_available(account: AgentAccount) -> None:
    account.receive(IRON, 3)
    with pytest.raises(InsufficientInventory):
        account.reserve_quantity(IRON, 4)


def test_spending_reserved_gold_removes_it_from_the_account(account: AgentAccount) -> None:
    account.reserve_gold(Decimal("40"))
    account.spend_reserved_gold(Decimal("40"))
    assert account.gold == Decimal("60")
    assert account.reserved_gold == ZERO


def test_cannot_release_more_than_is_reserved(account: AgentAccount) -> None:
    account.reserve_gold(Decimal("10"))
    with pytest.raises(EngineInvariantError, match="only"):
        account.release_gold(Decimal("11"))


def test_delivering_reserved_inventory_removes_it(account: AgentAccount) -> None:
    account.receive(IRON, 5)
    account.reserve_quantity(IRON, 5)
    account.deliver_reserved(IRON, 2)
    assert account.quantity(IRON) == 3
    assert account.reserved(IRON) == 3
    assert account.available_quantity(IRON) == 0


@pytest.mark.parametrize("amount", [Decimal("-1")])
def test_negative_amounts_are_rejected(account: AgentAccount, amount: Decimal) -> None:
    with pytest.raises(EngineInvariantError, match="negative"):
        account.reserve_gold(amount)
    with pytest.raises(EngineInvariantError, match="negative"):
        account.credit_gold(amount)


def test_an_invariant_violation_is_not_a_refusal_type(account: AgentAccount) -> None:
    """The executor treats ValueError, KeyError and ArithmeticError as refusals.

    A broken invariant must not be one of them, or it would be logged as a
    rejected action and the run would carry on with corrupted state.
    """
    assert not issubclass(EngineInvariantError, (ValueError, KeyError, ArithmeticError))
    account.receive(IRON, 2)
    account.reserve_quantity(IRON, 1)
    with pytest.raises(EngineInvariantError, match="only 1"):
        account.release_quantity(IRON, 2)
    with pytest.raises(EngineInvariantError, match="negative"):
        account.receive(IRON, -1)


# ---------------------------------------------------------------------------
# World construction
# ---------------------------------------------------------------------------
def test_population_matches_the_configuration() -> None:
    config = make_config()
    state = build_world(config)
    roles = [state.account(a).role for a in state.agent_ids]

    assert len(state.accounts) == config.population.total_agents
    assert roles.count(Role.ADVENTURER) == sum(config.population.adventurers.values())
    assert roles.count(Role.ARTISAN) == sum(config.population.artisans.values())
    assert set(roles) == {Role.ADVENTURER, Role.ARTISAN}


def test_agent_identity_is_reproducible() -> None:
    """Same agents, same ids, same endowments across a seed-trio (design s12)."""
    first, second = build_world(make_config()), build_world(make_config())
    assert first.agent_ids == second.agent_ids
    assert [first.account(a).gold for a in first.agent_ids] == [
        second.account(a).gold for a in second.agent_ids
    ]


def test_agent_creation_order_does_not_depend_on_yaml_key_order() -> None:
    reordered = make_config(population={"adventurers": {"ranger": 2, "mage": 2, "warrior": 2}})
    assert build_world(reordered).agent_ids == build_world(make_config()).agent_ids


def test_endowments_follow_the_configured_role() -> None:
    state = build_world(make_config(economy={"initial_gold": {"artisan": 555}}))
    assert state.account("woodworking_1").gold == Decimal("555")


def test_missing_role_endowment_is_an_error() -> None:
    config = make_config()
    object.__setattr__(config.economy, "initial_gold", {"adventurer": Decimal(1)})
    with pytest.raises(KeyError, match="initial_gold"):
        build_world(config)


# ---------------------------------------------------------------------------
# Seeded randomness
# ---------------------------------------------------------------------------
def test_streams_are_reproducible_from_the_run_seed() -> None:
    first = RunRandom(42).stream(STREAM_EXPEDITION)
    second = RunRandom(42).stream(STREAM_EXPEDITION)
    assert [first.random() for _ in range(5)] == [second.random() for _ in range(5)]


def test_different_seeds_give_different_streams() -> None:
    assert RunRandom(1).stream(STREAM_SPAWN).random() != RunRandom(2).stream(STREAM_SPAWN).random()


def test_streams_are_independent_of_each_other() -> None:
    """A trio must share world randomness even though A, B and C draw differently."""
    run = RunRandom(7)
    baseline = [run.stream(STREAM_EXPEDITION).random() for _ in range(3)]

    other = RunRandom(7)
    # Consume a different amount from an unrelated stream first.
    for _ in range(50):
        other.stream(STREAM_SCHEDULER).random()

    assert [other.stream(STREAM_EXPEDITION).random() for _ in range(3)] == baseline


def test_an_undeclared_stream_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown random stream"):
        RunRandom(1).stream("improvised")


def test_seed_derivation_is_stable_across_processes() -> None:
    """SHA-256, not Python's salted hash, or reproducibility would break silently."""
    assert derive_seed(12345, STREAM_EXPEDITION) == derive_seed(12345, STREAM_EXPEDITION)
    assert derive_seed(12345, STREAM_EXPEDITION) != derive_seed(12345, STREAM_SPAWN)


def test_permutation_is_deterministic_and_seed_derived() -> None:
    """The scheduler's tie-break order comes from the run seed (design s8.3)."""
    agents = [f"agent_{i}" for i in range(8)]
    first = RunRandom(3).permutation(agents, STREAM_SCHEDULER)
    second = RunRandom(3).permutation(agents, STREAM_SCHEDULER)
    other = RunRandom(4).permutation(agents, STREAM_SCHEDULER)

    assert first == second
    assert sorted(first) == sorted(agents)
    assert first != other
