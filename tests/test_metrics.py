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
    Trade,
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


def test_a_reply_in_the_same_tick_answers_a_request() -> None:
    """Delivery is immediate, so an agent activated later that cycle can reply."""
    from rpg_economy.metrics.run_metrics import unanswered_requests

    log = _synthetic_log(
        [
            (
                EventType.MESSAGE_SENT,
                5,
                "a",
                {"type": "REQUEST", "recipient_id": "b", "good": "iron"},
            ),
            (
                EventType.MESSAGE_SENT,
                5,
                "b",
                {"type": "OFFER", "recipient_id": "a", "good": "iron"},
            ),
            (
                EventType.MESSAGE_SENT,
                6,
                "c",
                {"type": "REQUEST", "recipient_id": "d", "good": "wood"},
            ),
        ]
    )
    assert unanswered_requests(log, short_config()) == 1


def test_a_reply_about_another_good_does_not_answer_a_request() -> None:
    from rpg_economy.metrics.run_metrics import unanswered_requests

    log = _synthetic_log(
        [
            (
                EventType.MESSAGE_SENT,
                5,
                "a",
                {"type": "REQUEST", "recipient_id": "b", "good": "iron"},
            ),
            (
                EventType.MESSAGE_SENT,
                6,
                "b",
                {"type": "OFFER", "recipient_id": "a", "good": "wood"},
            ),
        ]
    )
    assert unanswered_requests(log, short_config()) == 1


def test_common_horizon_metrics_stop_at_the_horizon(tmp_path: Path) -> None:
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    cut = config.phases.reversion_tick
    metrics = compute_run_metrics(result.log_path, config, common_active_horizon=cut)
    full = compute_run_metrics(result.log_path, config)

    assert full.common_horizon is None
    assert metrics.common_horizon is not None
    assert metrics.common_horizon["horizon"] == cut
    assert metrics.common_horizon["traded_value"] <= sum(full.volume_value.values())
    assert metrics.common_horizon["stockout_ticks"] <= full.stockouts["total_ticks"]


def test_k_binding_rate_is_the_share_of_cycles_that_used_all_of_k() -> None:
    from rpg_economy.metrics.run_metrics import coordination

    def cycle_end(effective: int, **extra):
        return (EventType.CYCLE_END, 1, None, {"k": 4, "effective_activations": effective, **extra})

    log = _synthetic_log(
        [
            cycle_end(4),
            cycle_end(2),
            cycle_end(5),
            cycle_end(0),
            # Not counted: after budget exhaustion nothing could bind.
            cycle_end(0, post_exhaustion=True),
            # Not counted: no K recorded.
            (EventType.CYCLE_END, 1, None, {"effective_activations": 3}),
        ]
    )
    assert coordination(log)["k_binding_rate"] == pytest.approx(2 / 4)


def test_k_binding_rate_is_missing_without_cycle_records() -> None:
    from rpg_economy.metrics.run_metrics import coordination

    assert coordination(_synthetic_log([]))["k_binding_rate"] is None


def test_truncated_outputs_count_agent_and_director_replies_cut_off() -> None:
    from rpg_economy.metrics.run_metrics import coordination

    log = _synthetic_log(
        [
            (EventType.AGENT_OUTPUT_INVALID, 1, "a", {"truncated": True}),
            (EventType.AGENT_OUTPUT_INVALID, 1, "a", {"truncated": False}),
            (EventType.AGENT_OUTPUT_INVALID, 1, "b", {}),
            (EventType.DIRECTOR_CALL, 1, None, {"finish_reason": "length"}),
            (EventType.DIRECTOR_CALL, 1, None, {"finish_reason": "stop"}),
        ]
    )
    result = coordination(log)
    assert result["truncated_outputs"] == 2
    assert result["invalid_outputs"] == 3


def test_k_binding_rate_comes_from_a_real_log(tmp_path: Path) -> None:
    from rpg_economy.metrics.run_metrics import coordination

    result = run_condition(short_config(), tmp_path / "run.jsonl")

    rate = coordination(RunLog(result.log_path))["k_binding_rate"]
    assert rate is not None
    assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# The common active horizon: missing, never zero (design s8.4, s11.3)
# ---------------------------------------------------------------------------
def test_a_common_horizon_before_the_shock_leaves_every_shock_metric_missing(
    tmp_path: Path,
) -> None:
    """No shock tick was observed, so nothing about the shock phase is zero."""
    config = short_config()
    trades = [(t, Decimal("10") + t % 3) for t in range(0, 30)]
    path = tmp_path / "run.jsonl"
    _log_with_iron_trades(path, config, trades)

    for horizon in (config.phases.shock_tick, config.phases.shock_tick - 3):
        common = compute_run_metrics(path, config, common_active_horizon=horizon).common_horizon
        assert common["horizon"] == horizon
        assert common["traded_value"] is None
        assert common["stockout_ticks"] is None
        assert common["volatility_shock"] is None
        assert common["volatility_shock_phase"] is None


def test_the_window_volatility_is_clipped_at_the_common_horizon(tmp_path: Path) -> None:
    """Both volatility definitions stop at the horizon, not only the phase-level one."""
    # Iron trades below: the scarce resource is pinned to iron here.
    config = short_config(shocks={"resource_scarcity": {"target_resource": "iron"}})
    trades = [
        (10, Decimal("10")),
        (11, Decimal("12")),
        (15, Decimal("10")),
        (16, Decimal("30")),
    ]
    path = tmp_path / "run.jsonl"
    log = _log_with_iron_trades(path, config, trades)

    common = compute_run_metrics(path, config, common_active_horizon=15).common_horizon
    first_window = volatility_by_phase(log, config, "iron", until=15)[PHASE_SHOCK]
    assert common["volatility_shock"] == pytest.approx(first_window)
    assert common["volatility_shock"] == pytest.approx(2**0.5)  # stdev(10, 12)
    assert volatility_by_phase(log, config, "iron")[PHASE_SHOCK] > common["volatility_shock"]
    assert common["volatility_shock_phase"] == pytest.approx(2**0.5)
    assert common["traded_value"] == pytest.approx(22.0)


def test_recovery_short_of_the_reversion_tick_is_censored_not_missing() -> None:
    """s8.4: the censoring rule applies on the common active horizon too."""
    config = short_config()
    series = flat_series(10.0, config.time.horizon_ticks)
    # Every horizon that has observed the reference at the last base tick.
    for horizon in (
        config.phases.reversion_tick,
        config.phases.shock_tick + 1,
        config.phases.shock_tick,
    ):
        observed = recovery_time(series, config, horizon=horizon)
        assert observed == Duration(value=0, censored=True)


def test_recovery_without_a_reference_is_missing_once_the_base_phase_is_observed() -> None:
    """Missing, not censored, on any horizon that saw the last base tick (s11.3).

    A horizon that ends before it never observed the reference, so it censors
    at zero ticks like any other horizon without a recovery phase.
    """
    config = short_config()
    series: list[Decimal | None] = [None] * config.time.horizon_ticks
    for horizon in (config.phases.shock_tick, config.phases.reversion_tick, None):
        assert recovery_time(series, config, horizon=horizon) is None
    early = recovery_time(series, config, horizon=config.phases.shock_tick - 1)
    assert early == Duration(value=0, censored=True)


# ---------------------------------------------------------------------------
# The productivity-drop target comes from an explicit table (design s11.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("specialization", "product"),
    [("smithing", "melee_weapon"), ("woodworking", "bow_gear"), ("alchemy", "health_potion")],
)
def test_each_specialization_has_one_pre_registered_main_product(
    specialization: str, product: str
) -> None:
    config = make_config(
        run={"scenario": "productivity_drop"},
        shocks={"productivity_drop": {"target_specialization": specialization}},
    )
    assert primary_target(config).good.value == product


def test_a_specialization_without_a_main_product_fails_loudly(monkeypatch) -> None:
    from rpg_economy.domain import ArtisanSpecialization
    from rpg_economy.metrics import targets

    monkeypatch.delitem(targets.SPECIALIZATION_MAIN_PRODUCT, ArtisanSpecialization.WOODWORKING)
    config = make_config(
        run={"scenario": "productivity_drop"},
        shocks={"productivity_drop": {"target_specialization": "woodworking"}},
    )
    with pytest.raises(ValueError, match="woodworking"):
        primary_target(config)


def test_a_main_product_of_another_specialization_fails_loudly(monkeypatch) -> None:
    from rpg_economy.domain import ArtisanSpecialization, Product
    from rpg_economy.metrics import targets

    monkeypatch.setitem(
        targets.SPECIALIZATION_MAIN_PRODUCT, ArtisanSpecialization.ALCHEMY, Product.MELEE_WEAPON
    )
    config = make_config(
        run={"scenario": "productivity_drop"},
        shocks={"productivity_drop": {"target_specialization": "alchemy"}},
    )
    with pytest.raises(ValueError, match="not a alchemy recipe"):
        primary_target(config)


# ---------------------------------------------------------------------------
# Pre-registered secondary analyses (design s11.1, s11.3)
# ---------------------------------------------------------------------------
def test_volume_is_reported_per_good_and_phase(tmp_path: Path) -> None:
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    metrics = compute_run_metrics(result.log_path, config)

    for phase in (PHASE_BASE, PHASE_SHOCK, PHASE_RECOVERY):
        units = sum(cells[phase]["units"] for cells in metrics.volume_by_good.values())
        value = sum(cells[phase]["value"] for cells in metrics.volume_by_good.values())
        assert units == metrics.volume_units[phase]
        assert value == pytest.approx(metrics.volume_value[phase])


def test_the_scarcity_shock_propagates_to_the_main_product(tmp_path: Path) -> None:
    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    metrics = compute_run_metrics(result.log_path, config)

    assert metrics.propagation is not None
    assert metrics.propagation["source"] == "herbs"
    assert metrics.propagation["good"] == "health_potion"
    assert set(metrics.propagation["mean_price"]) == {PHASE_BASE, PHASE_SHOCK, PHASE_RECOVERY}
    assert set(metrics.propagation["stockout_ticks"]) == {PHASE_BASE, PHASE_SHOCK, PHASE_RECOVERY}
    payload = metrics.to_payload()
    assert payload["propagation"]["good"] == "health_potion"
    assert "propagation_recovery" in payload


def test_a_product_target_has_no_propagation_block(tmp_path: Path) -> None:
    config = short_config(run={"condition": "B", "scenario": "demand_surge"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    metrics = compute_run_metrics(result.log_path, config)
    assert metrics.propagation is None
    assert metrics.propagation_recovery is None


def test_the_inventory_stockout_variant_replays_the_world_inventory(tmp_path: Path) -> None:
    """Reconstructed from the log alone, and consistent with the starting world."""
    from rpg_economy.engine import build_world
    from rpg_economy.metrics.run_metrics import _inventory_delta, inventory_stockouts

    config = short_config(run={"condition": "B"})
    result = run_condition(config, tmp_path / "run.jsonl", cycles=config.time.total_cycles)
    log = RunLog(result.log_path)
    world = build_world(config)

    for good in ("iron", "melee_weapon", "health_potion"):
        variant = inventory_stockouts(log, config, good)
        assert variant is not None
        assert variant["total_ticks"] == sum(variant["by_phase"].values())
        final = sum(h["inventory"].get(good, 0) for h in log.final_holdings.values())
        initial = final - sum(_inventory_delta(e, good) for e in log.events)
        from rpg_economy.domain import parse_good

        assert initial == sum(world.account(a).quantity(parse_good(good)) for a in world.agent_ids)


def test_the_inventory_stockout_counts_ticks_with_nothing_held(tmp_path: Path) -> None:
    from rpg_economy.metrics.run_metrics import inventory_stockouts

    config = short_config()
    log = _synthetic_log(
        [
            (EventType.TICK, 0, None, {}),
            (EventType.EXPEDITION_RESOLVED, 1, "warrior_1", {"yielded": {"iron": 2}}),
            (EventType.TICK, 1, None, {}),
            (
                EventType.PRODUCTION_COMPLETED,
                2,
                "smithing_1",
                {"product": "melee_weapon", "quantity": 1, "consumed": {"iron": 2}},
            ),
            (EventType.TICK, 2, None, {}),
            (EventType.TICK, 3, None, {}),
            (EventType.RUN_END, 3, None, {"holdings": {"a": {"gold": "0", "inventory": {}}}}),
        ]
    )
    variant = inventory_stockouts(log, config, "iron")
    assert variant["total_ticks"] == 3  # ticks 0, 2 and 3: nothing held
    assert variant["episodes"] == 2
    assert variant["longest"] == 2


def test_the_inventory_stockout_is_missing_without_final_holdings(tmp_path: Path) -> None:
    from rpg_economy.metrics.run_metrics import inventory_stockouts

    log = _synthetic_log([(EventType.TICK, 0, None, {})])
    assert inventory_stockouts(log, short_config(), "iron") is None


def test_the_multi_good_aggregate_is_the_median_of_base_normalized_goods(tmp_path: Path) -> None:
    """s11.3: normalize each good by its base phase, then the median across goods."""
    from rpg_economy.metrics.run_metrics import multi_good_aggregate, volume_by_good

    config = short_config()  # base [0, 10), shock [10, 20)

    def trade(tick: int, good: str, price: str):
        return (
            EventType.TRADE_EXECUTED,
            tick,
            None,
            {"good": good, "quantity": 1, "price": price, "value": price},
        )

    events = [
        # iron: 2 base trades, 4 shock trades -> volume ratio 2.0
        trade(1, "iron", "10"),
        trade(2, "iron", "12"),
        *(trade(t, "iron", "11") for t in (11, 12, 13, 14)),
        # wood: 2 base, 1 shock -> 0.5
        trade(1, "wood", "5"),
        trade(2, "wood", "7"),
        trade(11, "wood", "6"),
        # leather: 2 base, 2 shock -> 1.0
        trade(1, "leather", "8"),
        trade(2, "leather", "8"),
        trade(11, "leather", "8"),
        trade(12, "leather", "8"),
    ]
    log = _synthetic_log(events)
    log.trades = [
        Trade(
            tick=tick,
            good=payload["good"],
            quantity=1,
            price=Decimal(payload["price"]),
            value=Decimal(payload["value"]),
        )
        for _, tick, _, payload in events
    ]
    aggregate = multi_good_aggregate(log, config, volume_by_good(log, config))

    assert aggregate["per_good"]["iron"]["volume"] == pytest.approx(2.0)
    assert aggregate["per_good"]["wood"]["volume"] == pytest.approx(0.5)
    assert aggregate["per_good"]["leather"]["volume"] == pytest.approx(1.0)
    # Goods that never traded in the base phase have no ratio, and no vote.
    assert aggregate["per_good"]["armor"]["volume"] is None
    assert aggregate["volume"] == pytest.approx(1.0)
    # Volatility: iron's shock dispersion is 0 over a base of stdev(10, 12);
    # wood has one shock trade (missing), leather a base of 0 (missing).
    assert aggregate["per_good"]["iron"]["volatility"] == pytest.approx(0.0)
    assert aggregate["per_good"]["wood"]["volatility"] is None
    assert aggregate["per_good"]["leather"]["volatility"] is None
    assert aggregate["volatility"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Useful production is strictly first-in first-out (design s11.1)
# ---------------------------------------------------------------------------
def test_a_late_sale_consumes_the_oldest_lot_not_a_newer_one() -> None:
    """An expired lot that still holds units is sold first; the sale is late."""
    config = short_config()
    window = config.metrics.useful_production_ticks_t
    first, second = 1, 2
    assert second + window < config.time.horizon_ticks  # neither lot is right-censored
    log = _synthetic_log(
        [
            _produced(first, "smith_1"),
            _produced(second, "smith_1"),
            # Past the first lot's window, inside the second's: under FIFO it
            # sells the first lot's unit, late. Crediting the second lot would
            # count a unit as placed that is still on the shelf.
            _trade(first + window + 1, "smith_1"),
        ]
    )
    result = useful_production(log, config)

    assert result["produced"] == 2
    assert result["useful"] == 0
    assert result["overproduction"] == 2


def test_fifo_moves_on_to_the_next_lot_once_the_oldest_is_sold() -> None:
    config = short_config()
    window = config.metrics.useful_production_ticks_t
    log = _synthetic_log(
        [
            _produced(1, "smith_1", quantity=2),
            _produced(3, "smith_1", quantity=2),
            _trade(2, "smith_1", quantity=3),  # both units of lot 1, one of lot 2
            _trade(3 + window, "smith_1"),  # the last unit of lot 2, on its last tick
            _trade(3 + window + 1, "smith_1"),  # nothing left to sell from a lot
        ]
    )
    result = useful_production(log, config)

    assert result["produced"] == 4
    assert result["useful"] == 4


# ---------------------------------------------------------------------------
# Recovery unobserved within the common horizon is censored (design s8.4)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_a_horizon_before_the_reference_tick_is_censored_not_missing() -> None:
    config = short_config()
    series = flat_series(10.0, config.time.horizon_ticks)
    last_base = config.phases.shock_tick - 1

    # The reference sits at the last base tick; a horizon ending on or before
    # it never observed that tick, and recovery was not observed either.
    assert recovery_time(series, config, horizon=last_base) == Duration(0, True)
    assert recovery_time(series, config, horizon=last_base - 3) == Duration(0, True)
    assert recovery_time(series, config, horizon=0) == Duration(0, True)
    # One tick later it has; there is no recovery phase yet, so it is censored.
    assert recovery_time(series, config, horizon=last_base + 1) == Duration(0, True)


@pytest.mark.invariant
def test_a_horizon_before_the_reference_tick_reads_nothing_past_it() -> None:
    """Censoring there needs no reference: ticks past the horizon are not read."""
    config = short_config()
    last_base = config.phases.shock_tick - 1

    class Guarded(list):
        def __getitem__(self, index):  # type: ignore[no-untyped-def]
            if isinstance(index, int) and index >= last_base:
                raise AssertionError(f"read tick {index} past the horizon")
            return super().__getitem__(index)

    series = Guarded(flat_series(10.0, config.time.horizon_ticks))
    # Even a series with no reference at all is censored, not missing.
    series_without_reference: list[Decimal | None] = [None] * config.time.horizon_ticks

    assert recovery_time(series, config, horizon=last_base) == Duration(0, True)
    assert recovery_time(series_without_reference, config, horizon=last_base) == Duration(0, True)


# ---------------------------------------------------------------------------
# Stockouts over every good, the liquidity_loss secondary (design s11.3)
# ---------------------------------------------------------------------------
def test_the_multi_good_aggregate_carries_stockouts_net_of_base() -> None:
    from rpg_economy.metrics.run_metrics import multi_good_aggregate, volume_by_good

    config = short_config()  # base [0, 10), shock [10, 20)
    events: list = []
    order_id = 0
    # iron has an ask resting in ticks [0, 8) and [10, 14): stocked out 2 of 10
    # base ticks and 6 of 10 shock ticks, a difference of 0.4. wood rests on
    # [0, 9) and [10, 19): 1 and 1, a difference of 0. Every other good is never
    # offered, so it is stocked out throughout: difference 0.
    spans = {"iron": [(0, 8), (10, 14)], "wood": [(0, 9), (10, 19)]}
    closing: dict[int, list[int]] = {}
    for good, ranges in spans.items():
        for lo, hi in ranges:
            order_id += 1
            events.append(
                (
                    EventType.ORDER_PLACED,
                    lo,
                    None,
                    {"good": good, "side": "sell", "order_id": order_id},
                )
            )
            closing.setdefault(hi, []).append(order_id)
    by_tick: list = []
    for tick in range(config.time.horizon_ticks):
        for oid in closing.get(tick, []):
            by_tick.append((EventType.ORDER_EXPIRED, tick, None, {"order_id": oid}))
        by_tick.extend(e for e in events if e[1] == tick)
        by_tick.append((EventType.TICK, tick, None, {}))
    log = _synthetic_log(by_tick)

    aggregate = multi_good_aggregate(log, config, volume_by_good(log, config))

    assert aggregate["per_good"]["iron"]["stockouts"] == pytest.approx(0.4)
    assert aggregate["per_good"]["wood"]["stockouts"] == pytest.approx(0.0)
    assert aggregate["per_good"]["armor"]["stockouts"] == pytest.approx(0.0)
    assert aggregate["stockouts"] == pytest.approx(0.0)
