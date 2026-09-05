"""Central limit-order book (design v1.0 s5.2).

The market executes valid operations; it never coordinates intentions. It holds
no official price -- prices exist only as the record of transactions that
happened.

Rules implemented here, all from s5.2:

* limit orders carrying good, quantity and price;
* price-time priority;
* no selling without inventory, no buying without gold, enforced by reserving
  holdings when an order is placed rather than when it executes;
* orders expire after a fixed number of ticks;
* every order and every trade is logged with counterparties, roles, quantity,
  price, tick and status.

Two further rules, also from s5.2:

Trade price is the resting order's limit price. When an incoming order
crosses a resting one, the trade executes at the price already on the book. This
is the standard convention for a continuous limit-order book: the resting order
advertised a price and the incoming one accepted it, so the taker cannot improve
on the terms it chose to accept.

An agent cannot trade with itself. A self-trade moves nothing and costs
nothing, but it would inflate traded volume -- the primary metric of the
liquidity-loss scenario (design v1.0 s11.3) -- with activity that is not
economic activity. An order that would cross the agent's own resting order on
the other side is refused, so the book is never crossed (best bid above best
ask) and nothing is cancelled behind the agent's back: the agent cancels its
own order first, in the same place_orders if it likes. Governed by
``MarketConfig.allow_self_trade`` so the choice is configuration, not code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from rpg_economy.domain import Good, OrderSide, OrderStatus, Role
from rpg_economy.engine.state import ZERO, WorldState


class OrderRejected(Exception):
    """Raised when an order cannot be accepted. The order never reaches the book."""


@dataclass
class Order:
    """A resting or in-flight limit order."""

    order_id: int
    sequence: int
    agent_id: str
    side: OrderSide
    good: Good
    limit_price: Decimal
    quantity: int
    placed_tick: int
    expires_at_tick: int
    remaining: int = 0
    status: OrderStatus = OrderStatus.OPEN

    def __post_init__(self) -> None:
        if not self.remaining:
            self.remaining = self.quantity

    @property
    def is_resting(self) -> bool:
        return self.status in {OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED}

    @property
    def filled(self) -> int:
        return self.quantity - self.remaining


@dataclass(frozen=True)
class Trade:
    """An executed transaction. The only source of price information."""

    trade_id: int
    tick: int
    good: Good
    quantity: int
    price: Decimal
    buyer_id: str
    seller_id: str
    buyer_role: Role
    seller_role: Role
    buy_order_id: int
    sell_order_id: int

    @property
    def value(self) -> Decimal:
        return self.price * self.quantity


@dataclass
class MarketEvents:
    """Callbacks the market invokes so the engine can log every transition.

    The market does not import the logger: it reports what happened and the
    engine decides how to record it.
    """

    on_order_placed: Callable[[Order], None] | None = None
    on_order_rejected: Callable[[Order, str], None] | None = None
    on_order_filled: Callable[[Order], None] | None = None
    on_order_partially_filled: Callable[[Order], None] | None = None
    on_order_expired: Callable[[Order], None] | None = None
    on_order_cancelled: Callable[[Order], None] | None = None
    on_trade: Callable[[Trade], None] | None = None

    def emit(self, hook: str, *args: object) -> None:
        callback = getattr(self, hook)
        if callback is not None:
            callback(*args)


class Market:
    """The central limit-order book and its matching engine."""

    def __init__(
        self,
        state: WorldState,
        *,
        order_expiry_ticks: int,
        visible_last_n_trades: int,
        allow_self_trade: bool = False,
        events: MarketEvents | None = None,
    ) -> None:
        self._state = state
        self._expiry_ticks = order_expiry_ticks
        self._visible_last_n = visible_last_n_trades
        self._allow_self_trade = allow_self_trade
        self._events = events or MarketEvents()

        self._orders: dict[int, Order] = {}
        self._book: dict[tuple[Good, OrderSide], list[Order]] = {}
        self._trades: list[Trade] = []
        self._next_order_id = 1
        self._next_trade_id = 1
        self._next_sequence = 1

    # -- introspection -----------------------------------------------------
    @property
    def trades(self) -> Sequence[Trade]:
        return tuple(self._trades)

    def order(self, order_id: int) -> Order:
        return self._orders[order_id]

    def resting_orders(self, good: Good, side: OrderSide) -> list[Order]:
        """Resting orders for one side, in price-time priority order."""
        return self._prioritized(self._book.get((good, side), ()), side)

    def best_bid(self, good: Good) -> Decimal | None:
        orders = self.resting_orders(good, OrderSide.BUY)
        return orders[0].limit_price if orders else None

    def best_ask(self, good: Good) -> Decimal | None:
        orders = self.resting_orders(good, OrderSide.SELL)
        return orders[0].limit_price if orders else None

    def last_trades(self, good: Good, limit: int | None = None) -> list[Trade]:
        """The most recent trades of ``good``, newest last (design v1.0 s5.2)."""
        count = self._visible_last_n if limit is None else limit
        matching = [t for t in self._trades if t.good == good]
        return matching[-count:] if count else []

    def median_price(self, good: Good, since_tick: int) -> Decimal | None:
        """The median price of ``good`` over its trades at or after ``since_tick``."""
        prices = sorted(t.price for t in self._trades if t.good == good and t.tick >= since_tick)
        if not prices:
            return None
        mid = len(prices) // 2
        if len(prices) % 2:
            return prices[mid]
        return ((prices[mid - 1] + prices[mid]) / 2).quantize(Decimal("0.01"))

    def trade_count(self, good: Good, since_tick: int) -> int:
        """How many trades of ``good`` happened at or after ``since_tick``."""
        return sum(1 for t in self._trades if t.good == good and t.tick >= since_tick)

    def trades_of(self, agent_id: str, limit: int) -> list[Trade]:
        """The agent's own most recent trades, as buyer or seller, newest last."""
        own = [t for t in self._trades if agent_id in (t.buyer_id, t.seller_id)]
        return own[-limit:] if limit else []

    def has_resting_asks(self, good: Good) -> bool:
        """Whether the good is currently offered at all.

        This is the primary stockout predicate (design v1.0 s11.1): a stockout
        holds at a tick when no ask rests on the book.
        """
        return bool(self._book.get((good, OrderSide.SELL)))

    # -- ordering ----------------------------------------------------------
    @staticmethod
    def _prioritized(orders: Iterable[Order], side: OrderSide) -> list[Order]:
        """Sort by price-time priority: best price first, then earliest arrival.

        Buyers compete by bidding higher, sellers by asking lower, so the price
        key is negated for bids. ``sequence`` is a strict arrival counter, which
        keeps time priority well defined when several orders share a tick.
        """
        resting = [order for order in orders if order.is_resting]
        if side is OrderSide.BUY:
            return sorted(resting, key=lambda o: (-o.limit_price, o.sequence))
        return sorted(resting, key=lambda o: (o.limit_price, o.sequence))

    # -- placement ---------------------------------------------------------
    def place(
        self,
        *,
        agent_id: str,
        side: OrderSide,
        good: Good,
        quantity: int,
        price: Decimal,
        tick: int,
    ) -> Order:
        """Validate, reserve for, and submit a limit order.

        A rejected order never enters the book and reserves nothing.
        """
        account = self._state.account(agent_id)

        if quantity <= 0:
            raise OrderRejected(f"quantity must be positive, got {quantity}")
        if price <= ZERO:
            raise OrderRejected(f"price must be positive, got {price}")

        # No selling without inventory, no buying without gold (design s5.2).
        if side is OrderSide.SELL:
            if account.available_quantity(good) < quantity:
                raise OrderRejected(
                    f"{agent_id} cannot sell {quantity} {good}: only "
                    f"{account.available_quantity(good)} unreserved"
                )
        else:
            required = price * quantity
            if account.available_gold < required:
                raise OrderRejected(
                    f"{agent_id} cannot bid {required} gold: only "
                    f"{account.available_gold} unreserved"
                )

        # Refused if it would cross the agent's own opposite order.
        if not self._allow_self_trade:
            opposite = OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY

            def crosses(resting: Order) -> bool:
                if side is OrderSide.BUY:
                    return price >= resting.limit_price
                return price <= resting.limit_price

            book = self._book.get((good, opposite), ())
            crossed_own = next((o for o in book if o.agent_id == agent_id and crosses(o)), None)
            if crossed_own is not None:
                raise OrderRejected(
                    f"order would cross {agent_id}'s own resting {opposite.value} "
                    f"order {crossed_own.order_id} at {crossed_own.limit_price}; "
                    f"cancel it first"
                )

        order = Order(
            order_id=self._next_order_id,
            sequence=self._next_sequence,
            agent_id=agent_id,
            side=side,
            good=good,
            limit_price=price,
            quantity=quantity,
            placed_tick=tick,
            expires_at_tick=tick + self._expiry_ticks,
        )
        self._next_order_id += 1
        self._next_sequence += 1

        # Reserve before matching, so partial fills and the resting remainder
        # are both backed by holdings nothing else can claim.
        if side is OrderSide.SELL:
            account.reserve_quantity(good, quantity)
        else:
            account.reserve_gold(price * quantity)

        self._orders[order.order_id] = order
        self._events.emit("on_order_placed", order)

        self._match(order, tick)

        if order.is_resting:
            self._book.setdefault((good, side), []).append(order)
        return order

    # -- matching ----------------------------------------------------------
    def _crossing(self, incoming: Order) -> list[Order]:
        """Resting orders on the opposite side that ``incoming`` can trade with."""
        opposite = OrderSide.SELL if incoming.side is OrderSide.BUY else OrderSide.BUY
        candidates = self._prioritized(self._book.get((incoming.good, opposite), ()), opposite)

        def crosses(resting: Order) -> bool:
            if not self._allow_self_trade and resting.agent_id == incoming.agent_id:
                return False
            if incoming.side is OrderSide.BUY:
                return resting.limit_price <= incoming.limit_price
            return resting.limit_price >= incoming.limit_price

        return [order for order in candidates if crosses(order)]

    def _match(self, incoming: Order, tick: int) -> None:
        for resting in self._crossing(incoming):
            if incoming.remaining == 0:
                break
            quantity = min(incoming.remaining, resting.remaining)
            # The resting order set the terms; the incoming order accepted them.
            self._execute(incoming, resting, quantity, resting.limit_price, tick)

    def _execute(
        self, incoming: Order, resting: Order, quantity: int, price: Decimal, tick: int
    ) -> None:
        buy = incoming if incoming.side is OrderSide.BUY else resting
        sell = resting if incoming.side is OrderSide.BUY else incoming

        buyer = self._state.account(buy.agent_id)
        seller = self._state.account(sell.agent_id)
        value = price * quantity

        # Goods move out of the seller's reservation; gold out of the buyer's.
        seller.deliver_reserved(sell.good, quantity)
        buyer.receive(buy.good, quantity)
        buyer.spend_reserved_gold(value, placed_tick=buy.placed_tick)
        seller.credit_gold(value)

        # A buyer that crossed a cheaper ask reserved more than it spent. The
        # difference must be released, or gold silently stays locked forever.
        improvement = (buy.limit_price - price) * quantity
        if improvement > ZERO:
            buyer.refund_gold(improvement, placed_tick=buy.placed_tick)

        for order in (buy, sell):
            order.remaining -= quantity
            order.status = (
                OrderStatus.FILLED if order.remaining == 0 else OrderStatus.PARTIALLY_FILLED
            )

        trade = Trade(
            trade_id=self._next_trade_id,
            tick=tick,
            good=buy.good,
            quantity=quantity,
            price=price,
            buyer_id=buyer.agent_id,
            seller_id=seller.agent_id,
            buyer_role=buyer.role,
            seller_role=seller.role,
            buy_order_id=buy.order_id,
            sell_order_id=sell.order_id,
        )
        self._next_trade_id += 1
        self._trades.append(trade)
        self._events.emit("on_trade", trade)

        for order in (buy, sell):
            if order.status is OrderStatus.FILLED:
                self._events.emit("on_order_filled", order)
            else:
                self._events.emit("on_order_partially_filled", order)

        self._compact(buy)
        self._compact(sell)

    def _compact(self, order: Order) -> None:
        """Drop a fully filled order from the book."""
        if order.is_resting:
            return
        key = (order.good, order.side)
        resting = self._book.get(key)
        if resting and order in resting:
            resting.remove(order)
            if not resting:
                del self._book[key]

    # -- lifecycle ---------------------------------------------------------
    def _unwind(self, order: Order) -> None:
        """Release whatever a no-longer-live order still has reserved."""
        account = self._state.account(order.agent_id)
        if order.side is OrderSide.SELL:
            account.release_quantity(order.good, order.remaining)
        else:
            account.release_gold(order.limit_price * order.remaining, placed_tick=order.placed_tick)

    def cancel(self, order_id: int, *, agent_id: str | None = None) -> Order:
        """Withdraw a resting order, releasing what it reserved.

        With ``agent_id``, only that agent's own order may be cancelled.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise OrderRejected(f"order {order_id} does not exist")
        if agent_id is not None and order.agent_id != agent_id:
            raise OrderRejected(f"order {order_id} belongs to another agent")
        if not order.is_resting:
            raise OrderRejected(f"order {order_id} is {order.status}, not resting")
        self._unwind(order)
        order.status = OrderStatus.CANCELLED
        self._compact(order)
        self._events.emit("on_order_cancelled", order)
        return order

    def expire_due(self, tick: int) -> list[Order]:
        """Expire every resting order whose lifetime has elapsed.

        Called from the engine's mechanical phase (design v1.0 s8.1). Orders
        are expired in id order so the sequence never depends on dict iteration.
        """
        due = [
            order
            for order in self._orders.values()
            if order.is_resting and tick >= order.expires_at_tick
        ]
        for order in sorted(due, key=lambda o: o.order_id):
            self._unwind(order)
            order.status = OrderStatus.EXPIRED
            self._compact(order)
            self._events.emit("on_order_expired", order)
        return due

    def expired_unfilled_of(self, agent_id: str, since_tick: int) -> list[Order]:
        """The agent's orders that expired at or after ``since_tick`` with nothing filled, newest first."""
        return sorted(
            (
                o
                for o in self._orders.values()
                if o.agent_id == agent_id
                and o.status is OrderStatus.EXPIRED
                and o.remaining == o.quantity
                and o.expires_at_tick >= since_tick
            ),
            key=lambda o: -o.order_id,
        )

    def open_orders_of(self, agent_id: str) -> list[Order]:
        return sorted(
            (o for o in self._orders.values() if o.agent_id == agent_id and o.is_resting),
            key=lambda o: o.order_id,
        )
