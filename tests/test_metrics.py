"""Per-run metrics, with the censoring rule of design v1.0 s11.1."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from rpg_economy.domain import Scenario
from rpg_economy.logging_ import EventLogWriter, EventType
from rpg_economy.metrics.run_metrics import (
    PHASE_BASE,
    PHASE_RECOVERY,
    PHASE_SHOCK,
    Duration,
    RunLog,
    compute_run_metrics,
    gini,
    phase_of,
    price_series,
    recovery_reference,
    recovery_time,
    useful_production,
    volatility_by_phase,
    volatility_phase_level,
    volatility_windows,
)
from rpg_economy.metrics.targets import TargetKind, primary_target, propagation_product
from rpg_economy.runner import run_condition, run_trio
from tests.conftest import make_config


def short_config(**overrides):
    """A compressed run: 10 base, 10 shock, 20 recovery."""
    base = {
        "phases": {"base_ticks": 10, "shock_ticks": 10, "recovery_ticks": 20},
        "time": {"horizon_ticks": 40},
        "metrics": {"volatility_window_w": 5, "recovery_hold_ticks_m": 3},
    }
    base.update(overrides)
    return make_config(**base)


def flat_series(value: float, horizon: int) -> list[Decimal | None]:
    return [Decimal(str(value))] * horizon


def _log_with_iron_trades(path: Path, config, ticks: list[tuple[int, Decimal]]) -> RunLog:
    """A synthetic log holding exactly the given iron trades and nothing else.

    Built by hand so the trade pattern is the test's fixture rather than an
    accident of whatever the rule agents happened to do that seed.
    """
    from tests.conftest import make_metadata

    with EventLogWriter(path, make_metadata(config)) as log:
        for tick, price in ticks:
            log.write(
                EventType.TRADE_EXECUTED,
                tick=tick,
                cycle=tick // config.time.ticks_per_cycle,
                payload={
                    "good": "iron",
                    "quantity": 1,
                    "price": str(price),
                    "value": str(price),
                    "buyer": "warrior_1",
                    "seller": "woodworking_1",
                    "buyer_role": "adventurer",
                    "seller_role": "artisan",
                },
            )
    return RunLog(path)


# ---------------------------------------------------------------------------
# Pre-registered targets (design v1.0 s11.3)
# ---------------------------------------------------------------------------
def test_each_scenario_has_a_fixed_primary_target() -> None:
    for scenario in (
        Scenario.RESOURCE_SCARCITY,
        Scenario.DEMAND_SURGE,
        Scenario.PRODUCTIVITY_DROP,
        Scenario.LIQUIDITY_LOSS,
    ):
        target = primary_target(make_config(run={"scenario": scenario.value}))
        assert target.scenario is scenario
        assert target.primary_metrics


def test_resource_scarcity_targets_the_scarce_resource() -> None:
    target = primary_target(make_config(run={"scenario": "resource_scarcity"}))
    assert target.kind is TargetKind.PRICE
    assert target.good.value == "herbs"


def test_liquidity_loss_targets_aggregate_activity() -> None:
    """No single affected good, so volatility and stockouts become secondary."""
    target = primary_target(make_config(run={"scenario": "liquidity_loss"}))
    assert target.kind is TargetKind.TRADED_VALUE
    assert target.good is None
    assert "traded_value" in target.primary_metrics
    assert "volatility" not in target.primary_metrics


def test_productivity_drop_uses_the_resource_product_mapping() -> None:
    target = primary_target(make_config(run={"scenario": "productivity_drop"}))
    assert target.good.value == "melee_weapon"


def test_the_propagation_mapping_is_the_design_one() -> None:
    from rpg_economy.domain import Resource

    assert propagation_product(Resource.IRON).value == "melee_weapon"
    assert propagation_product(Resource.HERBS).value == "health_potion"


# ---------------------------------------------------------------------------
# Recovery and censoring (design v1.0 s11.1) -- the load-bearing rule
# ---------------------------------------------------------------------------
def test_recovery_is_measured_from_the_reversion_tick() -> None:
    config = short_config()
    series = flat_series(10.0, config.time.horizon_ticks)
    result = recovery_time(series, config)

    assert result.observed
    # Flat at the reference: in band from the first recovery tick, so recovery
    # is dated to where the qualifying stretch began.
    assert result.value == 0


@pytest.mark.invariant
def test_a_run_that_never_recovers_is_censored_never_imputed() -> None:
    """Design v1.0 s11.1: mark censored, record the max observed time."""
    config = short_config()
    series = flat_series(10.0, config.time.horizon_ticks)
    # Push the target far outside the band for the whole recovery phase.
    for tick in range(config.phases.reversion_tick, config.time.horizon_ticks):
        series[tick] = Decimal("1000")

    result = recovery_time(series, config)
    assert result.censored
    assert result.value == config.time.horizon_ticks - config.phases.reversion_tick


@pytest.mark.invariant
def test_a_censored_duration_is_never_confused_with_a_measurement() -> None:
    censored = Duration(value=200, censored=True)
    observed = Duration(value=200, censored=False)

    assert censored.observed is False
    assert observed.observed is True
    assert censored.to_payload()["censored"] is True


def test_recovery_requires_m_consecutive_ticks_in_band() -> None:
    config = short_config()
    horizon = config.time.horizon_ticks
    series = flat_series(10.0, horizon)
    start = config.phases.reversion_tick

    # In band, but never for M consecutive ticks: alternate in and out.
    for offset in range(start, horizon):
        series[offset] = Decimal("10") if (offset - start) % 2 == 0 else Decimal("1000")

    assert recovery_time(series, config).censored


def test_an_unknown_value_breaks_the_in_band_run() -> None:
    """A tick with no trade is not evidence of recovery."""
    config = short_config()
    horizon = config.time.horizon_ticks
    series = flat_series(10.0, horizon)
    for offset in range(config.phases.reversion_tick, horizon):
        series[offset] = None

    assert recovery_time(series, config).censored


def test_the_reference_is_the_base_phase_mean(  # design v1.0 s11.1
) -> None:
    config = short_config()
    series = flat_series(0.0, config.time.horizon_ticks)
    window = config.metrics.volatility_window_w
    for tick in range(config.phases.shock_tick - window, config.phases.shock_tick):
        series[tick] = Decimal("42")

    assert recovery_reference(series, config) == Decimal("42")


def test_recovery_over_the_common_active_horizon_can_censor_a_recovered_run() -> None:
    """Experiment 2's secondary analysis stops early; the rule still holds."""
    config = short_config()
    horizon = config.time.horizon_ticks
    series = flat_series(1000.0, horizon)
    for tick in range(config.phases.shock_tick - 5, config.phases.shock_tick):
        series[tick] = Decimal("10")
    for tick in range(horizon - 5, horizon):
        series[tick] = Decimal("10")

    full = recovery_time(series, config)
    truncated = recovery_time(series, config, horizon=config.phases.reversion_tick + 2)
    assert truncated.censored
    assert truncated.value <= full.value


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------
def test_price_series_is_none_where_the_good_did_not_trade() -> None:
    series = price_series([], "iron", horizon=5, window=2)
    assert series == [None] * 5


def test_phases_partition_the_horizon() -> None:
    config = short_config()
    assert phase_of(0, config) == PHASE_BASE
    assert phase_of(config.phases.shock_tick, config) == PHASE_SHOCK
    assert phase_of(config.phases.reversion_tick, config) == PHASE_RECOVERY


# ---------------------------------------------------------------------------
# Other economic metrics
# ---------------------------------------------------------------------------
def test_gini_is_zero_when_wealth_is_equal() -> None:
    config = short_config()
    log = RunLog.__new__(RunLog)
    log.trades = []
    holdings = {f"a{i}": {"gold": "100", "inventory": {}} for i in range(5)}
    assert gini(log, config, holdings) == pytest.approx(0.0, abs=1e-9)


def test_gini_rises_with_concentration() -> None:
    config = short_config()
    log = RunLog.__new__(RunLog)
    log.trades = []
    equal = {f"a{i}": {"gold": "100", "inventory": {}} for i in range(5)}
    skewed = {"a0": {"gold": "500", "inventory": {}}} | {
        f"a{i}": {"gold": "1", "inventory": {}} for i in range(1, 5)
    }
    assert gini(log, config, skewed) > gini(log, config, equal)


@pytest.mark.invariant
def test_gini_uses_fixed_reference_prices_not_concurrent_ones() -> None:
    """Design s11.1: valuing at current prices would make Gini an
    artifact of volatility, which is itself a dependent variable."""
    import inspect

    from rpg_economy.metrics import run_metrics

    source = inspect.getsource(run_metrics.gini)
    assert "base_phase_reference_prices" in source


def test_useful_production_complements_overproduction() -> None:
    config = short_config()
    log = RunLog.__new__(RunLog)
    log.events = []
    log.trades = []
    result = useful_production(log, config)
    assert result["useful"] + result["overproduction"] == result["produced"]


# ---------------------------------------------------------------------------
# End to end on a real run
# ---------------------------------------------------------------------------
def test_metrics_compute_from_a_real_log(tmp_path: Path) -> None:
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    metrics = compute_run_metrics(result.log_path, config)

    assert metrics.condition == "B"
    assert metrics.scenario == "resource_scarcity"
    assert metrics.target_good == "herbs"
    assert set(metrics.volatility) == {PHASE_BASE, PHASE_SHOCK, PHASE_RECOVERY}
    assert metrics.coordination["activations"] > 0
    assert metrics.tokens["authoritative_total"] > 0


def test_one_primary_value_per_run(tmp_path: Path) -> None:
    """Design v1.0 s11.3: the unit of analysis is the run."""
    from rpg_economy.analysis.results import CONTINUOUS_METRICS, metric_value

    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    metrics = compute_run_metrics(result.log_path, config)

    for metric in CONTINUOUS_METRICS:
        value = metric_value(metrics, metric)
        # None is a legitimate answer for a metric this run cannot measure
        # (design s11.3); what s11.3 forbids is more than one value.
        assert value is None or isinstance(value, float)


def test_metrics_carry_the_run_identity(tmp_path: Path) -> None:
    trio = run_trio(short_config(), tmp_path, seed=1001, cycles=10)
    for condition, result in trio.items():
        metrics = compute_run_metrics(result.log_path, result.config)
        assert metrics.condition == condition.value
        assert metrics.seed == 1001


# ---------------------------------------------------------------------------
# Insufficient data is missing, never zero (design v1.0 s11.1, s11.3)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_phase_with_no_trades_reports_missing_volatility(tmp_path: Path) -> None:
    """A phase with no price is not a phase with a stable price.

    If every window falls below two trades the deviation list is empty; scoring
    that 0.0 would read as perfect stability, the most favourable value the
    metric can take, and would systematically flatter whichever condition
    traded least.
    """
    config = short_config()
    log = _log_with_iron_trades(tmp_path / "empty.jsonl", config, ticks=[])

    result = volatility_by_phase(log, config, "iron")
    assert result[PHASE_SHOCK] is None
    assert result[PHASE_SHOCK] != 0.0


@pytest.mark.invariant
def test_windows_below_two_trades_are_excluded_not_scored_zero(tmp_path: Path) -> None:
    """Two lone trades in separate windows must not read as zero volatility.

    Iron moving from 36.07 to 43.89 -- a 22% swing -- across two windows holding
    one trade each is not a stable price.
    """
    config = short_config()
    shock = config.phases.shock_tick
    window = config.metrics.volatility_window_w
    log = _log_with_iron_trades(
        tmp_path / "singles.jsonl",
        config,
        ticks=[(shock, Decimal("36.07")), (shock + window, Decimal("43.89"))],
    )

    assert volatility_by_phase(log, config, "iron")[PHASE_SHOCK] is None


@pytest.mark.invariant
def test_a_window_with_two_trades_is_measured(tmp_path: Path) -> None:
    """The complement: enough data yields a real number, not None."""
    config = short_config()
    shock = config.phases.shock_tick
    log = _log_with_iron_trades(
        tmp_path / "pair.jsonl",
        config,
        ticks=[(shock, Decimal("36.07")), (shock + 1, Decimal("43.89"))],
    )

    value = volatility_by_phase(log, config, "iron")[PHASE_SHOCK]
    assert value is not None and value > 0.0


def test_the_window_census_explains_a_missing_volatility(tmp_path: Path) -> None:
    config = short_config()
    shock = config.phases.shock_tick
    log = _log_with_iron_trades(tmp_path / "census.jsonl", config, ticks=[(shock, Decimal("10"))])

    census = volatility_windows(log, config, "iron")[PHASE_SHOCK]
    assert census["measurable"] == 0
    assert census["thin"] == census["windows"] > 0
    assert census["trades"] == 1


@pytest.mark.invariant
def test_gini_comes_from_the_log_without_being_handed_holdings(tmp_path: Path) -> None:
    """Design v1.0 s15: a run is reconstructable from its log alone.

    Final holdings are read from the log, so a caller that does not pass them
    still gets a measured Gini rather than 0.0 -- perfect equality -- built from
    no data at all.
    """
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)

    from_log = compute_run_metrics(result.log_path, config)
    from_state = compute_run_metrics(
        result.log_path,
        config,
        holdings={
            agent_id: {
                "gold": str(account.gold),
                "tax_arrears": str(account.tax_arrears),
                "inventory": {g.value: n for g, n in account.inventory.items() if n},
            }
            for agent_id, account in result.state.accounts.items()
        },
    )

    assert from_log.gini is not None
    assert from_log.gini > 0.0
    assert from_log.gini == pytest.approx(from_state.gini)


@pytest.mark.invariant
def test_gini_is_missing_rather_than_zero_without_holdings(tmp_path: Path) -> None:
    """An unmeasurable distribution is not an equal one (design s11.3)."""
    config = short_config()
    log = _log_with_iron_trades(tmp_path / "nobody.jsonl", config, ticks=[])

    assert gini(log, config, {}) is None


@pytest.mark.invariant
def test_phase_level_volatility_measures_where_the_window_mean_cannot(tmp_path: Path) -> None:
    """The phase-level companion (design s11.3): two trades in the phase are enough.

    The 36.07 -> 43.89 case is unmeasurable under the
    window-mean definition because the two trades land in different windows. The
    phase-level definition prices it, which is the point: whether a swing is
    measurable must not depend on how the treatment spaced the trades.
    """
    config = short_config()
    shock = config.phases.shock_tick
    window = config.metrics.volatility_window_w
    log = _log_with_iron_trades(
        tmp_path / "spread.jsonl",
        config,
        ticks=[(shock, Decimal("36.07")), (shock + window, Decimal("43.89"))],
    )

    assert volatility_by_phase(log, config, "iron")[PHASE_SHOCK] is None
    phase_level = volatility_phase_level(log, config, "iron")[PHASE_SHOCK]
    assert phase_level is not None and phase_level > 5.0


@pytest.mark.invariant
def test_phase_level_volatility_is_still_missing_below_two_trades(tmp_path: Path) -> None:
    """Unmeasurable by treatment is reported, not imputed (design s11.3)."""
    config = short_config()
    log = _log_with_iron_trades(
        tmp_path / "one.jsonl", config, ticks=[(config.phases.shock_tick, Decimal("12"))]
    )

    assert volatility_phase_level(log, config, "iron")[PHASE_SHOCK] is None


# ---------------------------------------------------------------------------
# Missing, not zero (design s11.3)
# ---------------------------------------------------------------------------
def test_the_reference_is_the_smoothed_value_at_the_last_base_tick() -> None:
    """The series is already a W-tick window; averaging it again would widen it."""
    config = short_config()
    series = flat_series(10.0, config.time.horizon_ticks)
    series[config.phases.shock_tick - 1] = Decimal("42")

    assert recovery_reference(series, config) == Decimal("42")


@pytest.mark.invariant
def test_recovery_without_a_reference_is_missing_not_censored() -> None:
    """No trades at the end of the base phase: recovery cannot be measured."""
    config = short_config()
    series: list[Decimal | None] = flat_series(10.0, config.time.horizon_ticks)
    series[config.phases.shock_tick - 1] = None

    assert recovery_time(series, config) is None


def _synthetic_log(events: list[tuple[EventType, int, str | None, dict]]) -> RunLog:
    from rpg_economy.logging_ import Event

    log = RunLog.__new__(RunLog)
    log.events = [
        Event(seq=i, type=kind, tick=tick, cycle=0, agent_id=agent, payload=payload)
        for i, (kind, tick, agent, payload) in enumerate(events)
    ]
    log.trades = []
    return log


def _trade(tick: int, seller: str, good: str = "melee_weapon", quantity: int = 1):
    return (
        EventType.TRADE_EXECUTED,
        tick,
        None,
        {"good": good, "quantity": quantity, "seller_id": seller, "buyer_id": "warrior_1"},
    )


def _produced(tick: int, producer: str, quantity: int = 1, product: str = "melee_weapon"):
    return (
        EventType.PRODUCTION_COMPLETED,
        tick,
        producer,
        {"product": product, "quantity": quantity},
    )


def test_useful_production_counts_only_the_producers_own_sales() -> None:
    config = short_config()
    window = config.metrics.useful_production_ticks_t
    log = _synthetic_log(
        [
            _produced(1, "smith_1", quantity=3),
            _trade(2, "woodworking_1"),  # someone else's unit: says nothing about smith_1
            _trade(3, "smith_1"),  # placed in time
            _trade(1 + window + 1, "smith_1"),  # too late
        ]
    )
    result = useful_production(log, config)

    assert result["produced"] == 3
    assert result["useful"] == 1
    assert result["overproduction"] == 2


def test_useful_production_is_missing_when_nothing_was_produced() -> None:
    result = useful_production(_synthetic_log([_trade(2, "smith_1")]), short_config())
    assert result["useful_fraction"] is None


def test_coordination_efficiency_is_missing_without_activations() -> None:
    from rpg_economy.metrics.run_metrics import coordination

    result = coordination(_synthetic_log([]))
    assert result["coordination_efficiency"] is None
    assert result["activation_concentration"] is None


def test_coordination_reports_messages_concentration_and_director_figures() -> None:
    from rpg_economy.metrics.run_metrics import coordination

    output = (EventType.AGENT_OUTPUT, 1, None, {})
    log = _synthetic_log(
        [
            (output[0], 1, "a", {}),
            (output[0], 1, "a", {}),
            (output[0], 1, "b", {}),
            (EventType.MESSAGE_SENT, 1, "a", {"type": "OFFER", "recipient_id": "b"}),
            (EventType.MESSAGE_SENT, 1, "b", {"type": "OFFER", "recipient_id": "a"}),
            (EventType.MESSAGE_SENT, 1, "b", {"type": "REQUEST", "recipient_id": "a"}),
            (EventType.DIRECTOR_CALL, 1, None, {}),
            (EventType.DIRECTOR_CALL, 1, None, {}),
            (EventType.DIRECTOR_ERROR, 1, None, {}),
            (EventType.CYCLE_END, 1, None, {"director_retries": 1, "closed_by_cap": True}),
        ]
    )
    result = coordination(log)

    assert result["messages_by_type"] == {"OFFER": 2, "REQUEST": 1}
    assert result["activation_concentration"] == pytest.approx((2 / 3) ** 2 + (1 / 3) ** 2)
    assert result["director_retries"] == 1
    assert result["cycles_closed_by_cap"] == 1
    assert result["director_error_rate"] == pytest.approx(0.5)
