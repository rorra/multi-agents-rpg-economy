"""Central limit-order book: the invariants of design v1.0 s5.2."""

from __future__ import annotations

from decimal import Decimal

import pytest

from rpg_economy.domain import OrderSide, OrderStatus, Resource
from rpg_economy.engine import Market, OrderRejected, WorldState
from rpg_economy.engine.state import ZERO

IRON = Resource.IRON
WOOD = Resource.WOOD


def d(value: str) -> Decimal:
    return Decimal(value)


def stock(state: WorldState, agent_id: str, good: object, quantity: int) -> None:
    state.account(agent_id).receive(good, quantity)


# ---------------------------------------------------------------------------
# No naked sells
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_cannot_sell_without_inventory(market: Market) -> None:
    with pytest.raises(OrderRejected, match="cannot sell"):
        market.place(
            agent_id="woodworking_1",
            side=OrderSide.SELL,
            good=IRON,
            quantity=1,
            price=d("10"),
            tick=0,
        )


@pytest.mark.invariant
def test_cannot_sell_more_than_held(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_1", IRON, 3)
    with pytest.raises(OrderRejected, match="cannot sell"):
        market.place(
            agent_id="woodworking_1",
            side=OrderSide.SELL,
            good=IRON,
            quantity=4,
            price=d("10"),
            tick=0,
        )


@pytest.mark.invariant
def test_two_sell_orders_cannot_claim_the_same_unit(
    market: Market, two_traders: WorldState
) -> None:
    """Reservation is what makes the no-naked-sell rule hold at execution, not
    merely at placement."""
    stock(two_traders, "woodworking_1", IRON, 5)
    market.place(
        agent_id="woodworking_1", side=OrderSide.SELL, good=IRON, quantity=5, price=d("10"), tick=0
    )
    with pytest.raises(OrderRejected, match="cannot sell"):
        market.place(
            agent_id="woodworking_1",
            side=OrderSide.SELL,
            good=IRON,
            quantity=1,
            price=d("10"),
            tick=0,
        )


@pytest.mark.invariant
def test_a_rejected_order_reserves_nothing(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_1", IRON, 1)
    with pytest.raises(OrderRejected):
        market.place(
            agent_id="woodworking_1",
            side=OrderSide.SELL,
            good=IRON,
            quantity=9,
            price=d("10"),
            tick=0,
        )
    assert two_traders.account("woodworking_1").reserved(IRON) == 0
    assert two_traders.account("woodworking_1").available_quantity(IRON) == 1


# ---------------------------------------------------------------------------
# No overdrafts
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_cannot_bid_more_gold_than_held(market: Market) -> None:
    with pytest.raises(OrderRejected, match="cannot bid"):
        market.place(
            agent_id="woodworking_1",
            side=OrderSide.BUY,
            good=IRON,
            quantity=200,
            price=d("10"),
            tick=0,
        )


@pytest.mark.invariant
def test_two_buy_orders_cannot_claim_the_same_gold(market: Market) -> None:
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=100, price=d("10"), tick=0
    )
    with pytest.raises(OrderRejected, match="cannot bid"):
        market.place(
            agent_id="woodworking_1",
            side=OrderSide.BUY,
            good=WOOD,
            quantity=1,
            price=d("10"),
            tick=0,
        )


@pytest.mark.invariant
def test_gold_never_goes_negative_through_trading(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 100)
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=100, price=d("10"), tick=0
    )
    market.place(
        agent_id="woodworking_2",
        side=OrderSide.SELL,
        good=IRON,
        quantity=100,
        price=d("10"),
        tick=0,
    )
    for account in two_traders.accounts.values():
        assert account.gold >= ZERO
        assert account.available_gold >= ZERO


@pytest.mark.invariant
def test_trading_conserves_gold_and_goods(market: Market, two_traders: WorldState) -> None:
    """The market moves value; it never creates or destroys it."""
    stock(two_traders, "woodworking_2", IRON, 10)
    gold_before = two_traders.total_gold()
    iron_before = two_traders.total_quantity(IRON)

    market.place(
        agent_id="woodworking_1",
        side=OrderSide.BUY,
        good=IRON,
        quantity=6,
        price=d("12.50"),
        tick=0,
    )
    market.place(
        agent_id="woodworking_2",
        side=OrderSide.SELL,
        good=IRON,
        quantity=6,
        price=d("12.50"),
        tick=0,
    )

    assert two_traders.total_gold() == gold_before
    assert two_traders.total_quantity(IRON) == iron_before


# ---------------------------------------------------------------------------
# Price-time priority
# ---------------------------------------------------------------------------
def test_best_price_wins_among_asks(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 10)
    expensive = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("12"), tick=0
    )
    cheap = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("9"), tick=0
    )

    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("12"), tick=0
    )

    assert cheap.status is OrderStatus.FILLED
    assert expensive.status is OrderStatus.OPEN
    assert market.trades[-1].price == d("9")


def test_best_price_wins_among_bids(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 10)
    low = market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("9"), tick=0
    )
    high = market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("12"), tick=0
    )

    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("9"), tick=0
    )

    assert high.status is OrderStatus.FILLED
    assert low.status is OrderStatus.OPEN
    assert market.trades[-1].price == d("12")


def test_time_breaks_a_price_tie(market: Market, two_traders: WorldState) -> None:
    """Equal prices are served in arrival order, not by id or dict ordering."""
    stock(two_traders, "woodworking_2", IRON, 10)
    first = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("10"), tick=0
    )
    second = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("10"), tick=0
    )

    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("10"), tick=1
    )

    assert first.status is OrderStatus.FILLED
    assert second.status is OrderStatus.OPEN


def test_priority_holds_across_ticks(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 10)
    early = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("10"), tick=0
    )
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("10"), tick=3
    )

    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("10"), tick=5
    )
    assert early.status is OrderStatus.FILLED


def test_non_crossing_orders_rest_without_trading(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 10)
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("15"), tick=0
    )
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("10"), tick=0
    )

    assert market.trades == ()
    assert market.best_bid(IRON) == d("10")
    assert market.best_ask(IRON) == d("15")


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------
def test_partial_fill_leaves_the_remainder_resting(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 3)
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=3, price=d("10"), tick=0
    )
    buy = market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=10, price=d("10"), tick=0
    )

    assert buy.status is OrderStatus.PARTIALLY_FILLED
    assert buy.filled == 3
    assert buy.remaining == 7
    assert market.best_bid(IRON) == d("10")


def test_a_partial_fill_keeps_the_remainder_reserved(
    market: Market, two_traders: WorldState
) -> None:
    stock(two_traders, "woodworking_2", IRON, 3)
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=3, price=d("10"), tick=0
    )
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=10, price=d("10"), tick=0
    )

    buyer = two_traders.account("woodworking_1")
    assert buyer.gold == d("970")  # paid 3 x 10
    assert buyer.reserved_gold == d("70")  # 7 still bid at 10


def test_one_order_sweeps_several_price_levels(market: Market, two_traders: WorldState) -> None:
    """Sweeping must consume levels in price order and price each fill separately."""
    stock(two_traders, "woodworking_2", IRON, 10)
    for price in ("9", "10", "11"):
        market.place(
            agent_id="woodworking_2",
            side=OrderSide.SELL,
            good=IRON,
            quantity=2,
            price=d(price),
            tick=0,
        )

    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=5, price=d("11"), tick=0
    )

    assert [(t.quantity, t.price) for t in market.trades] == [
        (2, d("9")),
        (2, d("10")),
        (1, d("11")),
    ]


def test_partial_fills_across_ticks_accumulate(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 10)
    buy = market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=5, price=d("10"), tick=0
    )
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=2, price=d("10"), tick=1
    )
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=3, price=d("10"), tick=2
    )

    assert buy.status is OrderStatus.FILLED
    assert buy.filled == 5


# ---------------------------------------------------------------------------
# Trade pricing
# ---------------------------------------------------------------------------
def test_trade_executes_at_the_resting_order_price(market: Market, two_traders: WorldState) -> None:
    """The resting order set the terms; the incoming order accepted them."""
    stock(two_traders, "woodworking_2", IRON, 10)
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("8"), tick=0
    )
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("12"), tick=0
    )

    assert market.trades[-1].price == d("8")


def test_a_buyer_that_pays_less_than_it_bid_gets_the_difference_back(
    market: Market, two_traders: WorldState
) -> None:
    """Otherwise the surplus stays reserved forever and the agent slowly seizes up."""
    stock(two_traders, "woodworking_2", IRON, 10)
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("8"), tick=0
    )
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("12"), tick=0
    )

    buyer = two_traders.account("woodworking_1")
    assert buyer.gold == d("992")
    assert buyer.reserved_gold == ZERO
    assert buyer.available_gold == d("992")


def test_prices_are_exact_decimals(market: Market, two_traders: WorldState) -> None:
    """Money is Decimal: float drift would make the no-overdraft check unreliable."""
    stock(two_traders, "woodworking_2", IRON, 10)
    for _ in range(10):
        market.place(
            agent_id="woodworking_2",
            side=OrderSide.SELL,
            good=IRON,
            quantity=1,
            price=d("0.10"),
            tick=0,
        )
    market.place(
        agent_id="woodworking_1",
        side=OrderSide.BUY,
        good=IRON,
        quantity=10,
        price=d("0.10"),
        tick=0,
    )

    assert two_traders.account("woodworking_2").gold == d("1001.00")


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------
def test_order_expires_after_the_configured_lifetime(
    market: Market, two_traders: WorldState
) -> None:
    stock(two_traders, "woodworking_2", IRON, 5)
    order = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=5, price=d("10"), tick=0
    )
    assert order.expires_at_tick == 10

    assert market.expire_due(9) == []
    assert order.status is OrderStatus.OPEN

    assert market.expire_due(10) == [order]
    assert order.status is OrderStatus.EXPIRED


@pytest.mark.invariant
def test_expiry_releases_the_reservation(market: Market, two_traders: WorldState) -> None:
    """A reservation that outlives its order would strand the holding permanently."""
    stock(two_traders, "woodworking_2", IRON, 5)
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=5, price=d("10"), tick=0
    )
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=2, price=d("4"), tick=0
    )

    market.expire_due(10)

    seller = two_traders.account("woodworking_2")
    buyer = two_traders.account("woodworking_1")
    assert seller.reserved(IRON) == 0
    assert seller.available_quantity(IRON) == 5
    assert buyer.reserved_gold == ZERO
    assert buyer.available_gold == d("1000")


def test_expiry_releases_only_the_unfilled_remainder(
    market: Market, two_traders: WorldState
) -> None:
    stock(two_traders, "woodworking_2", IRON, 5)
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=5, price=d("10"), tick=0
    )
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=2, price=d("10"), tick=0
    )

    market.expire_due(10)

    seller = two_traders.account("woodworking_2")
    assert seller.quantity(IRON) == 3
    assert seller.reserved(IRON) == 0
    assert seller.available_quantity(IRON) == 3


def test_an_expired_order_leaves_the_book(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 5)
    market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=5, price=d("10"), tick=0
    )
    market.expire_due(10)

    assert market.best_ask(IRON) is None
    assert not market.has_resting_asks(IRON)


def test_expiry_is_ordered_by_order_id(market: Market, two_traders: WorldState) -> None:
    """Deterministic order matters: expiry emits log events."""
    stock(two_traders, "woodworking_2", IRON, 5)
    placed = [
        market.place(
            agent_id="woodworking_2",
            side=OrderSide.SELL,
            good=IRON,
            quantity=1,
            price=d("10"),
            tick=0,
        )
        for _ in range(5)
    ]
    assert [o.order_id for o in market.expire_due(10)] == [o.order_id for o in placed]


def test_a_filled_order_does_not_expire(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 1)
    order = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("10"), tick=0
    )
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("10"), tick=0
    )
    market.expire_due(100)
    assert order.status is OrderStatus.FILLED


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------
def test_cancel_releases_the_reservation(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 5)
    order = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=5, price=d("10"), tick=0
    )
    market.cancel(order.order_id)

    assert order.status is OrderStatus.CANCELLED
    assert two_traders.account("woodworking_2").available_quantity(IRON) == 5
    assert market.best_ask(IRON) is None


def test_cannot_cancel_a_filled_order(market: Market, two_traders: WorldState) -> None:
    stock(two_traders, "woodworking_2", IRON, 1)
    order = market.place(
        agent_id="woodworking_2", side=OrderSide.SELL, good=IRON, quantity=1, price=d("10"), tick=0
    )
    market.place(
        agent_id="woodworking_1", side=OrderSide.BUY, good=IRON, quantity=1, price=d("10"), tick=0
    )
    with pytest.raises(OrderRejected, match="not resting"):
        market.cancel(order.order_id)


# ---------------------------------------------------------------------------
# Self-trading (design v1.0 s5.2: never; the switch exists for tests only)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("first", "price"), [(OrderSide.SELL, "10"), (OrderSide.BUY, "12")])
def test_an_order_crossing_the_agents_own_is_refused(
    market: Market, two_traders: WorldState, first: OrderSide, price: str
) -> None:
    """Resting beside its own opposite order, it would leave the book crossed."""
    stock(two_traders, "woodworking_1", IRON, 5)
    market.place(
        agent_id="woodworking_1", side=first, good=IRON, quantity=1, price=d(price), tick=0
    )
    gold = two_traders.account("woodworking_1").available_gold
    iron = two_traders.account("woodworking_1").available_quantity(IRON)
    second = OrderSide.BUY if first is OrderSide.SELL else OrderSide.SELL
    crossing = d("12") if second is OrderSide.BUY else d("10")
    with pytest.raises(OrderRejected, match="own resting"):
        market.place(
            agent_id="woodworking_1", side=second, good=IRON, quantity=1, price=crossing, tick=0
        )
    assert market.trades == ()
    assert two_traders.account("woodworking_1").available_gold == gold
    assert two_traders.account("woodworking_1").available_quantity(IRON) == iron
    assert market.best_bid(IRON) is None or market.best_ask(IRON) is None
