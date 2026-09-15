"""The global aggregated view (design v1.0 s5.2, s3.3).

**One generator, two consumers.** The Director's global view in Condition A and
the bulletin published to everyone in Condition C are produced by this single
function, so they are byte-identical by construction rather than by careful
maintenance of two code paths. That identity is what makes C a clean
observability ablation: if the two views could drift, a difference between A and
C would confound coordination with information, which is precisely the confound C
exists to remove (s3.3).

The aggregates are derived from canonical state: mean price and variation,
volume per good, available supply, unmet demand, and publicly observable shocks.
Its format is fixed before the final runs (s5.2).

Determinism matters as much as content: goods are visited in a fixed order and
every number is rendered as a string, so the same state always serialises to the
same bytes.
"""

from __future__ import annotations

from decimal import Decimal

from rpg_economy.config.models import SimulationConfig
from rpg_economy.domain import ALL_GOODS, MONEY_QUANTUM, OrderSide
from rpg_economy.engine import Engine

ZERO = Decimal(0)

#: Version of the aggregate format. Fixed before the final runs (design s5.2).
GLOBAL_VIEW_VERSION = "1.2"


def _mean(values: list[Decimal]) -> Decimal:
    return sum(values, ZERO) / len(values) if values else ZERO


def _stdev(values: list[Decimal]) -> Decimal:
    """Population standard deviation, in Decimal throughout.

    Computed directly rather than via ``statistics`` so no value is ever routed
    through a float, which would let the aggregate differ in its last digits
    between platforms and break the byte-identity the ablation depends on.
    """
    if len(values) < 2:
        return ZERO
    mean = _mean(values)
    variance = sum(((v - mean) ** 2 for v in values), ZERO) / len(values)
    return variance.sqrt()


def build_global_summary(config: SimulationConfig, engine: Engine) -> dict:
    """Build the aggregated view of the economy at the current tick."""
    window = config.metrics.volatility_window_w
    since = max(0, engine.tick - window)

    goods: list[dict] = []
    for good in ALL_GOODS:
        trades = [t for t in engine.market.trades if t.good == good and t.tick >= since]
        prices = [t.price for t in trades]
        units = sum(t.quantity for t in trades)
        value = sum((t.value for t in trades), ZERO)

        asks = engine.market.resting_orders(good, OrderSide.SELL)
        bids = engine.market.resting_orders(good, OrderSide.BUY)

        goods.append(
            {
                "good": good.value,
                "trades": len(trades),
                # Missing, never 0.00, for a good that did not trade: a reader
                # would take 0.00 for a price.
                "mean_price": (str(_mean(prices).quantize(MONEY_QUANTUM)) if prices else None),
                "price_variation": (
                    str(_stdev(prices).quantize(MONEY_QUANTUM)) if len(prices) >= 2 else None
                ),
                "volume_units": units,
                "volume_value": str(value.quantize(MONEY_QUANTUM)),
                # Supply and demand as the book shows them: what is actually
                # offered and actually bid for, not private inventories.
                "available_supply": sum(order.remaining for order in asks),
                "unmet_demand": sum(order.remaining for order in bids),
                "offered": bool(asks),
            }
        )

    return {
        "version": GLOBAL_VIEW_VERSION,
        "tick": engine.tick,
        "cycle": engine.cycle,
        "window": window,
        "goods": goods,
        # Only shocks already observable in the world are reported (design s5.2).
        "shock": _public_shock(engine),
    }


#: The shock details that name what a shock hit: a good or a specialization.
#: Everything else a shock records stays out of the view -- its exact
#: multipliers, and above all any aggregate of private gold, such as the total
#: frozen by a liquidity loss.
PUBLIC_SHOCK_DETAILS = ("target_resource", "target_product", "target_specialization")


def _public_shock(engine: Engine) -> dict | None:
    """The active shock as the world can observe it: its type and what it hit."""
    shock = engine.shocks.active
    if shock is None:
        return None
    return {
        "scenario": shock.scenario.value,
        **{k: shock.details[k] for k in PUBLIC_SHOCK_DETAILS if k in shock.details},
    }


def render_global_summary(summary: dict) -> str:
    """Canonical serialization. The byte string A and C must share.

    Columnar rather than JSON (IMPLEMENTATION.md section 4.8.1): the per-good
    field names appear once in a header row instead of once per good. The
    Director's view and the bulletin are byte-identical (design v1.0 s3.3):
    there is one generator and one renderer, so both consumers get the same text.
    """
    required = {"version", "tick", "cycle", "window", "goods", "shock"}
    missing = required - set(summary)
    if missing:
        # Loud, because a global view that is not the canonical one would be
        # rendered differently for A and C and silently break s3.3 byte-identity.
        raise ValueError(
            f"not a canonical global summary; missing {sorted(missing)}. "
            f"Build it with build_global_summary rather than by hand."
        )

    header = (
        f"global_view v{summary['version']} tick={summary['tick']} "
        f"cycle={summary['cycle']} window={summary['window']}"
    )
    # Part of the canonical text, so the Director and the bulletin carry the
    # same definitions, byte for byte.
    legend = (
        f"columns: trades, volume_units, volume_value = trades, units and gold traded "
        f"in the last {summary['window']} ticks; mean_price, price_variation = mean and "
        f"standard deviation of those trade prices (- when too few trades); "
        f"available_supply = units offered for sale on the book now; unmet_demand = "
        f"units bid for and not yet filled; offered = whether any offer rests"
    )
    shock = summary["shock"]
    shock_line = "shock: none" if shock is None else "shock: " + _pairs(shock)

    goods = summary["goods"]
    if not goods:
        return "\n".join([header, legend, shock_line, "goods: none"])

    columns = [key for key in sorted(goods[0]) if key != "good"]
    rows = ["good|" + "|".join(columns)]
    rows += [
        str(entry["good"]) + "|" + "|".join(_cell(entry[column]) for column in columns)
        for entry in goods
    ]
    return "\n".join([header, legend, shock_line, *rows])


def _pairs(payload: dict) -> str:
    return " ".join(f"{key}={_cell(value)}" for key, value in sorted(payload.items()))


def _cell(value: object) -> str:
    """One field, rendered so the row is unambiguous and stable across runs."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        # Explicit, because "True"/"1" would read as a count in a numeric column.
        return "yes" if value else "no"
    return str(value)
