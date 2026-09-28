"""Bilingual reporting and figures, rendered from one results file (design v1.0 s11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpg_economy.analysis.report import generate_all_locales, render_report
from rpg_economy.analysis.results import build_results
from rpg_economy.i18n import clear_cache
from rpg_economy.metrics.run_metrics import Duration
from tests.test_statistics import SEEDS, fake_run, sample_results


@pytest.fixture(autouse=True)
def _reset_locale():
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Bilingual reports and figures from one results file
# ---------------------------------------------------------------------------
@pytest.mark.invariant
def test_both_languages_render_from_the_same_results(tmp_path: Path) -> None:
    """IMPLEMENTATION.md section 2: en and es from one results file."""
    results = sample_results()
    paths = generate_all_locales(results, tmp_path)

    english = paths["en"].read_text(encoding="utf-8")
    spanish = paths["es"].read_text(encoding="utf-8")

    assert "Resource scarcity" in english
    assert "Escasez de recurso" in spanish
    assert english != spanish

    # Same numbers underneath.
    left = json.loads((tmp_path / "en" / "results.json").read_text(encoding="utf-8"))
    right = json.loads((tmp_path / "es" / "results.json").read_text(encoding="utf-8"))
    assert left == right


def test_reports_go_to_per_locale_directories(tmp_path: Path) -> None:
    paths = generate_all_locales(sample_results(), tmp_path)
    assert paths["en"].parent.name == "en"
    assert paths["es"].parent.name == "es"


def test_the_report_states_the_censoring_rule(tmp_path: Path) -> None:
    text = render_report(sample_results(), locale="en", figure_paths={})
    assert "censored observation is a lower bound" in text


def test_the_report_names_the_primary_contrast(tmp_path: Path) -> None:
    text = render_report(sample_results(), locale="en", figure_paths={})
    assert "Primary contrast" in text
    assert "Holm" in text


def test_the_report_shows_recovered_and_censored_counts(tmp_path: Path) -> None:
    text = render_report(sample_results(), locale="en", figure_paths={})
    assert "Recovered" in text
    assert "Censored" in text


def test_figures_are_written_for_each_locale(tmp_path: Path) -> None:
    generate_all_locales(sample_results(), tmp_path)
    for locale in ("en", "es"):
        assert (tmp_path / locale / "recovery_km.png").is_file()
        assert (tmp_path / locale / "paired_differences.png").is_file()


@pytest.mark.invariant
def test_the_condition_palette_is_cvd_validated() -> None:
    """Categorical hues assigned in fixed order and checked, not eyeballed."""
    from rpg_economy.analysis.figures import CONDITION_COLOURS

    assert list(CONDITION_COLOURS) == ["A", "B", "C"]
    assert len(set(CONDITION_COLOURS.values())) == 3
    # Recorded from the validator run: worst adjacent pair dE 9.2 (deutan),
    # normal-vision floor 24.0. Both clear their thresholds (8.0 / 15.0).
    assert CONDITION_COLOURS["A"] == "#2a78d6"


@pytest.mark.invariant
def test_conditions_carry_a_second_encoding_beyond_colour() -> None:
    """Two conditions can trace the same path; hue alone would hide one.

    B and C coincide exactly whenever the bulletin changes no decision, and one
    line then covers the other completely while the legend still claims three
    series. Distinct dash patterns keep an overplotted series readable and carry
    identity for readers who cannot separate the hues.
    """
    from rpg_economy.analysis.figures import CONDITION_COLOURS, CONDITION_LINESTYLES

    assert set(CONDITION_LINESTYLES) == set(CONDITION_COLOURS)
    assert len({str(v) for v in CONDITION_LINESTYLES.values()}) == 3


def test_overplotted_series_still_render(tmp_path: Path) -> None:
    """The case the second encoding exists for: identical B and C."""
    from rpg_economy.analysis.figures import price_trajectory
    from rpg_economy.i18n import set_locale

    set_locale("en")
    shared = [float(i) for i in range(50)]
    path = price_trajectory(
        {"A": [float(i) * 2 for i in range(50)], "B": shared, "C": list(shared)},
        good_label="Iron",
        shock_tick=10,
        reversion_tick=30,
        path=tmp_path / "price.png",
    )
    assert path.is_file()
    assert path.stat().st_size > 0


@pytest.mark.invariant
def test_measurability_is_reported_beside_the_medians() -> None:
    """A median over few runs must be read knowing how few, like censoring."""
    runs = []
    for seed in range(1, 11):
        runs.append(fake_run("A", seed, volatility=2.0))
        b = fake_run("B", seed, volatility=1.0)
        if seed <= 4:
            b.volatility["shock"] = None
        runs.append(b)

    results = build_results(runs)
    census = results.measurability["volatility_shock"]

    assert census["A"] == {"runs": 10, "measured": 10, "missing": 0}
    assert census["B"] == {"runs": 10, "measured": 6, "missing": 4}

    report = render_report(results, locale="en", figure_paths={})
    assert "6/10" in report


@pytest.mark.invariant
def test_measurability_is_reported_as_an_outcome_not_only_a_footnote() -> None:
    """Thin-to-death markets are a treatment effect, so they get an outcome row."""
    runs = []
    for seed in range(1, 11):
        runs.append(fake_run("A", seed, volatility=2.0))
        b = fake_run("B", seed, volatility=1.0)
        if seed <= 6:
            b.volatility_phase["shock"] = None
        runs.append(b)

    results = build_results(runs)
    outcome = results.companion_outcomes["volatility_shock_phase"]
    # Design s11.3: the phase-level volatility sustains the comparison, so it
    # is contrasted on its measurable pairs, never suppressed.
    assert "volatility_shock_phase" not in results.suppressed_contrasts
    contrast = next(c for c in results.contrasts["volatility_shock_phase"] if c["right"] == "B")
    assert contrast["n_pairs"] == 4
    assert contrast["n_dropped"] == 6

    assert outcome["fraction"] == {"A": 1.0, "B": 0.4}
    assert outcome["counts"]["B"] == {"measurable": 4, "runs": 10}

    paired = next(row for row in outcome["paired_binary"] if row["right"] == "B")
    assert paired["discordant_pairs"] == 6
    assert paired["p_value"] < 0.05

    report = render_report(results, locale="en", figure_paths={})
    assert "4/10" in report


def test_both_languages_report_the_suppression() -> None:
    runs = []
    for seed in range(1, 11):
        runs.append(fake_run("A", seed, volatility=2.0))
        b = fake_run("B", seed, volatility=1.0)
        if seed <= 4:
            b.volatility["shock"] = None
        runs.append(b)
    results = build_results(runs)

    english = render_report(results, locale="en", figure_paths={})
    spanish = render_report(results, locale="es", figure_paths={})

    assert "Contrast suppressed" in english
    assert "Contraste suprimido" in spanish


def test_the_spanish_report_uses_a_decimal_comma(tmp_path: Path) -> None:
    results = sample_results()
    spanish = render_report(results, locale="es", figure_paths={})
    english = render_report(results, locale="en", figure_paths={})
    assert "0,6" in spanish
    assert "0.6" in english


def test_primary_metrics_are_reported_before_secondary_ones() -> None:
    text = render_report(sample_results(), locale="en", figure_paths={})
    primary = text.index("Primary metrics")
    secondary = text.index("Secondary metrics")
    assert primary < text.index("Stockout ticks") < secondary < text.index("Gini")


def test_the_report_has_a_coordination_and_system_section() -> None:
    text = render_report(sample_results(), locale="en", figure_paths={})
    assert "Coordination and system metrics" in text
    assert "Director retries" in text


def test_the_paired_figure_skips_a_suppressed_contrast(tmp_path: Path, monkeypatch) -> None:
    from rpg_economy.analysis import figures

    runs = []
    for seed in SEEDS:
        a = fake_run("A", seed, volatility=1.0)
        b = fake_run("B", seed, volatility=2.0)
        c = fake_run("C", seed, volatility=1.5)
        a.primary_metrics = ("volatility",)
        b.primary_metrics = c.primary_metrics = a.primary_metrics
        if seed <= 3:
            b.volatility["shock"] = None
        runs += [a, b, c]
    results = build_results(runs)
    assert "volatility_shock" in results.suppressed_contrasts

    drawn: list[str] = []
    original = figures.paired_differences

    def spy(differences, *, metric_label, path):
        drawn.append(metric_label)
        return original(differences, metric_label=metric_label, path=path)

    monkeypatch.setattr(figures, "paired_differences", spy)
    generate_all_locales(results, tmp_path)

    # The window contrast's A-B pair was suppressed, so the figure draws the
    # phase-level companion instead of the pairs that survived.
    from rpg_economy.i18n import t

    assert drawn
    assert all(
        label == t("report.metric.volatility_shock_phase", locale="es")
        or label == t("report.metric.volatility_shock_phase", locale="en")
        for label in drawn
    )


def test_rendering_all_locales_keeps_the_callers_language(tmp_path: Path) -> None:
    from rpg_economy.i18n import get_catalog, set_locale

    set_locale("en")
    generate_all_locales(sample_results(), tmp_path)
    assert get_catalog().locale == "en"


# ---------------------------------------------------------------------------
# The economy check reconciles gold against loot, taxes and storage
# ---------------------------------------------------------------------------
def _economy_run(tmp_path: Path):
    from rpg_economy.agents import RuleBasedDecisionPhase
    from rpg_economy.analysis.economy_check import analyse as check
    from rpg_economy.analysis.economy_check import target_good_for
    from rpg_economy.engine import Engine, build_world
    from rpg_economy.logging_ import EventLogWriter
    from tests.conftest import make_config, make_metadata

    config = make_config(
        phases={"base_ticks": 10, "shock_ticks": 10, "recovery_ticks": 10},
        time={"horizon_ticks": 30},
        metrics={"volatility_window_w": 3, "recovery_hold_ticks_m": 2},
    )
    initial, final = build_world(config), build_world(config)
    path = tmp_path / "run.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        Engine(config, final, log).run(RuleBasedDecisionPhase(config))
    return path, config, final, initial, target_good_for(config), check


def test_the_economy_check_accounts_for_every_gold_flow(tmp_path: Path) -> None:
    from tests.conftest import gold_flow

    path, config, final, initial, target, check = _economy_run(tmp_path)
    result = check(path, config, final, initial, target)

    assert result.gold_taxed > 0
    assert result.gold_storage >= 0
    assert result.gold_looted >= 0
    sinks = result.gold_taxed + result.gold_storage
    sources = result.gold_looted + result.gold_confiscation_paid
    assert sources - sinks == gold_flow(path)
    assert result.total_gold_start + sources - sinks == result.total_gold_end
    assert result.gold_reconciles


def test_the_economy_check_flags_gold_that_does_not_reconcile(tmp_path: Path) -> None:
    from decimal import Decimal

    path, config, final, initial, target, check = _economy_run(tmp_path)
    final.account("woodworking_1").gold += Decimal("1")

    assert not check(path, config, final, initial, target).gold_reconciles


def _synthetic_economy(tmp_path: Path, events):
    """Initial and final worlds joined only by the given events."""
    from rpg_economy.engine import build_world
    from rpg_economy.logging_ import EventLogWriter
    from tests.conftest import make_config, make_metadata

    config = make_config(
        phases={"base_ticks": 10, "shock_ticks": 10, "recovery_ticks": 10},
        time={"horizon_ticks": 30},
        metrics={"volatility_window_w": 3, "recovery_hold_ticks_m": 2},
    )
    path = tmp_path / "synthetic.jsonl"
    with EventLogWriter(path, make_metadata(config)) as log:
        for kind, tick, agent, payload in events:
            log.write(kind, tick=tick, cycle=0, agent_id=agent, payload=payload)
    return path, config, build_world(config), build_world(config)


def test_the_economy_check_reconciles_a_payment_of_tax_arrears(tmp_path: Path) -> None:
    """A TAX_PAID flagged ``arrears`` is gold leaving the agent like any tax."""
    from decimal import Decimal

    from rpg_economy.analysis.economy_check import analyse as check
    from rpg_economy.logging_ import EventType

    events = [
        (
            EventType.TAX_PAID,
            5,
            "woodworking_1",
            {"tax": "wealth", "amount": "4.00", "due": "10.00", "owed": "6.00"},
        ),
        (
            EventType.TAX_PAID,
            10,
            "woodworking_1",
            {"tax": "wealth", "amount": "6.00", "arrears": True, "owed": "0"},
        ),
    ]
    path, config, initial, final = _synthetic_economy(tmp_path, events)
    final.account("woodworking_1").gold -= Decimal("10.00")

    result = check(path, config, final, initial, None)
    assert result.gold_taxed == Decimal("10.00")
    assert result.gold_reconciles
    assert result.total_gold_start - result.gold_taxed == result.total_gold_end

    # Leaving the arrears payment out of the final gold no longer reconciles.
    final.account("woodworking_1").gold += Decimal("6.00")
    assert not check(path, config, final, initial, None).gold_reconciles


@pytest.mark.invariant
def test_the_economy_check_judges_bankruptcy_on_gold_net_of_arrears(tmp_path: Path) -> None:
    """Arrears are a lien (user ruling): gold owed to the tax is not spendable."""
    from decimal import Decimal

    from rpg_economy.analysis.economy_check import analyse as check

    path, config, initial, final = _synthetic_economy(tmp_path, [])
    assert not check(path, config, final, initial, None).bankrupt_agents

    account = final.account("woodworking_1")
    account.tax_arrears = account.gold - Decimal("0.01")
    result = check(path, config, final, initial, None)
    assert result.bankrupt_agents == ["woodworking_1"]
    # Arrears are not a gold flow: the balance still reconciles on gross gold.
    assert result.gold_reconciles


def test_the_richest_share_is_of_gold_net_of_arrears(tmp_path: Path) -> None:
    from decimal import Decimal

    from rpg_economy.analysis.economy_check import analyse as check

    path, config, initial, final = _synthetic_economy(tmp_path, [])
    for agent_id in final.agent_ids:
        final.account(agent_id).gold = Decimal("100")
    final.account(final.agent_ids[0]).gold = Decimal("300")
    before = check(path, config, final, initial, None).richest_share
    # The richest agent owes most of its gold: its net share falls.
    final.account(final.agent_ids[0]).tax_arrears = Decimal("250")
    after = check(path, config, final, initial, None).richest_share
    n = len(final.agent_ids)
    assert before == Decimal("300") / (Decimal("300") + 100 * (n - 1))
    assert after == Decimal("100") / (Decimal("50") + 100 * (n - 1))


# ---------------------------------------------------------------------------
# Mock runs are never presented as results
# ---------------------------------------------------------------------------
def _mock_results():
    runs = []
    for seed in SEEDS:
        for condition, volatility in (("A", 1.0), ("B", 2.0), ("C", 1.5)):
            run = fake_run(condition, seed, volatility=volatility + seed * 0.01)
            run.mock = True
            runs.append(run)
    return build_results(runs)


@pytest.mark.invariant
@pytest.mark.parametrize("locale", ["en", "es"])
def test_a_report_from_mock_runs_says_so(locale: str) -> None:
    from rpg_economy.i18n import set_locale, t

    report = render_report(_mock_results(), locale=locale, figure_paths={})
    set_locale(locale)
    banner = t("report.note.mock_runs")

    # Up front, beside the title, not in a footnote.
    assert banner in report.splitlines()[:5]


def test_a_report_from_model_runs_has_no_mock_banner() -> None:
    from rpg_economy.i18n import set_locale, t

    report = render_report(sample_results(), locale="en", figure_paths={})
    set_locale("en")
    assert t("report.note.mock_runs") not in report


# ---------------------------------------------------------------------------
# Pilot seeds are exploration only (design s11.4)
# ---------------------------------------------------------------------------
@pytest.mark.invariant
@pytest.mark.parametrize("locale", ["en", "es"])
def test_a_report_on_pilot_seeds_says_so_up_front(locale: str) -> None:
    from rpg_economy.i18n import set_locale, t

    results = sample_results()
    results.pilot_seeds_included = True
    report = render_report(results, locale=locale, figure_paths={})
    set_locale(locale)
    assert t("report.note.pilot_seeds") in report.splitlines()[:6]
    # The numbers are kept.
    assert t("report.section.contrasts_for", metric=t("report.metric.gini")) in report


def test_a_final_seed_report_has_no_pilot_banner() -> None:
    from rpg_economy.i18n import set_locale, t

    report = render_report(sample_results(), locale="en", figure_paths={})
    set_locale("en")
    assert t("report.note.pilot_seeds") not in report


# ---------------------------------------------------------------------------
# Tables describe the runs they count (design s11.3)
# ---------------------------------------------------------------------------
def _median_row(report: str, label: str) -> list[str]:
    line = next(line for line in report.splitlines() if line.startswith(label))
    return line[len(label) :].split()


def test_per_condition_medians_are_over_the_pairable_seeds() -> None:
    from rpg_economy.i18n import set_locale, t

    runs = []
    for seed in range(1, 11):
        a = fake_run("A", seed, volatility=1.0)
        b = fake_run("B", seed, volatility=1.0)
        a.gini = float(seed)
        b.gini = float(seed)
        if seed > 5:
            b.gini = None  # B measured on seeds 1-5 only
        runs += [a, b]
    results = build_results(runs)
    report = render_report(results, locale="en", figure_paths={})
    set_locale("en")

    # Over seeds 1-5 for both, n = 5: A's full-set median (5.5) would describe
    # runs the n does not count.
    assert _median_row(report, t("report.metric.gini")) == ["3", "3", "5"]


def test_the_median_recovery_time_is_the_kaplan_meier_median() -> None:
    from rpg_economy.i18n import set_locale, t

    runs = [fake_run(c, s, volatility=1.0) for c in "AB" for s in range(1, 6)]
    for run in runs:
        # B: recoveries at 21 and 22, three censored runs: the curve never
        # reaches one half, so there is no median.
        if run.condition == "B" and run.seed >= 3:
            run.recovery = Duration(40, True)
    results = build_results(runs)
    report = render_report(results, locale="en", figure_paths={})
    set_locale("en")

    rows = {
        line.split()[0]: line
        for line in report.splitlines()
        if line.startswith((t("condition.A"), t("condition.B")))
    }
    assert results.survival["kaplan_meier"]["B"]["median"] is None
    assert rows[t("condition.B").split()[0]].rstrip().endswith(t("report.value.missing"))
    assert results.survival["kaplan_meier"]["A"]["median"] == pytest.approx(23.0)


# ---------------------------------------------------------------------------
# Suppression rendered per pair
# ---------------------------------------------------------------------------
def test_the_report_names_the_suppressed_pairs_and_keeps_the_rest() -> None:
    from rpg_economy.analysis.report import contrast_label
    from rpg_economy.i18n import set_locale

    runs = []
    for seed in SEEDS:
        a = fake_run("A", seed, volatility=1.0 + seed * 0.01)
        b = fake_run("B", seed, volatility=2.0 + seed * 0.01)
        c = fake_run("C", seed, volatility=1.5 + seed * 0.01)
        if seed <= 3:
            c.volatility["shock"] = None
        runs += [a, b, c]
    results = build_results(runs)
    report = render_report(results, locale="en", figure_paths={})
    set_locale("en")

    assert contrast_label({"left": "A", "right": "C"}) in report
    assert "Contrast suppressed" in report
    section = report.split("Contrasts: Volatility (shock phase)\n", 1)[1]
    assert contrast_label({"left": "A", "right": "B"}) in section.split("\n\n", 1)[0]


# ---------------------------------------------------------------------------
# liquidity_loss: no target good, so no per-good tables and value figures
# ---------------------------------------------------------------------------
def _liquidity_results():
    from tests.test_statistics import _liquidity_run

    return build_results([_liquidity_run(c, s) for c in "ABC" for s in SEEDS])


def test_a_liquidity_loss_report_has_no_empty_target_good_tables() -> None:
    from rpg_economy.i18n import set_locale, t

    report = render_report(_liquidity_results(), locale="en", figure_paths={})
    set_locale("en")
    assert t("report.note.no_pairs") not in report
    assert t("report.metric.volatility_shock") not in report
    assert t("report.metric.stockout_ticks") not in report
    assert t("report.section.companion_outcome") not in report
    assert t("report.metric.traded_value") in report


def test_the_liquidity_loss_figure_plots_traded_value(tmp_path: Path, monkeypatch) -> None:
    from rpg_economy.analysis import figures

    calls: list[dict] = []
    original = figures.price_trajectory

    def spy(series, **kwargs):
        calls.append(kwargs)
        return original(series, **kwargs)

    monkeypatch.setattr(figures, "price_trajectory", spy)
    series = {c: [1.0, 2.0, None, 3.0] for c in "ABC"}
    generate_all_locales(
        _liquidity_results(), tmp_path, price_series=series, shock_tick=1, reversion_tick=2
    )
    assert calls and all(call["traded_value"] for call in calls)

    calls.clear()
    generate_all_locales(
        sample_results(), tmp_path / "price", price_series=series, shock_tick=1, reversion_tick=2
    )
    assert calls and not any(call["traded_value"] for call in calls)


def test_the_report_renders_the_secondary_analyses(tmp_path: Path) -> None:
    from rpg_economy.i18n import set_locale, t

    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        run.propagation = {
            "source": "iron",
            "good": "melee_weapon",
            "mean_price": {"base": 30.0, "shock": 35.0 + run.seed, "recovery": 31.0},
            "stockout_ticks": {"base": 0, "shock": run.seed % 3, "recovery": 0},
        }
        run.propagation_recovery = Duration(5 + run.seed, False)
        run.volume_by_good = {
            "iron": {
                "base": {"units": 10, "value": 100.0},
                "shock": {"units": 4, "value": 60.0},
                "recovery": {"units": 8, "value": 80.0},
            }
        }
    results = build_results(runs)
    for locale in ("en", "es"):
        report = render_report(results, locale=locale, figure_paths={})
        set_locale(locale)
        assert (
            t(
                "report.section.propagation",
                good=t("product.melee_weapon"),
                resource=t("resource.iron"),
            )
            in report
        )
        assert t("report.section.volume_by_good") in report
        assert t("report.metric.propagation_price_shock") in report


# ---------------------------------------------------------------------------
# The economy check uses the pre-registered target (design s11.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("resource_scarcity", "herbs"),
        ("demand_surge", "health_potion"),
        ("productivity_drop", "melee_weapon"),
        ("liquidity_loss", None),
    ],
)
def test_the_economy_check_targets_the_primary_target(scenario: str, expected) -> None:
    from rpg_economy.analysis.economy_check import target_good_for
    from rpg_economy.metrics.targets import primary_target
    from tests.conftest import make_config

    config = make_config(run={"scenario": scenario})
    good = target_good_for(config)
    assert good == primary_target(config).good
    assert (None if good is None else good.value) == expected


def test_the_economy_check_renders_traded_value_without_a_target_good(tmp_path: Path) -> None:
    from rpg_economy.analysis.render import render
    from rpg_economy.i18n import set_locale, t

    path, config, final, initial, _, check = _economy_run(tmp_path)
    result = check(path, config, final, initial, None)
    set_locale("en")
    text = render(result, config)
    assert result.target_good is None
    assert result.stockouts == []
    assert t("check.section.traded_value") in text
    assert t("check.section.stockouts") not in text


def test_stockout_episodes_are_a_whole_run_row(tmp_path: Path) -> None:
    from rpg_economy.analysis.economy_check import StockoutStats
    from rpg_economy.analysis.render import render
    from rpg_economy.i18n import set_locale, t

    path, config, final, initial, target, check = _economy_run(tmp_path)
    result = check(path, config, final, initial, target)
    result.stockouts = [
        StockoutStats(
            good="iron",
            episodes=3,
            total_ticks=9,
            longest=5,
            by_phase={"base": 2, "shock": 5, "recovery": 2},
        )
    ]
    set_locale("en")
    lines = render(result, config).splitlines()
    base = next(line for line in lines if line.startswith(t("phase.base")) and "[" not in line)
    whole = next(line for line in lines if line.startswith(t("check.row.whole_run")))
    assert base.split()[1:] == ["2"]
    assert whole[len(t("check.row.whole_run")) :].split() == ["9", "3", "5"]


# ---------------------------------------------------------------------------
# What the report renders from the results it was given
# ---------------------------------------------------------------------------
def _cells(line: str) -> list[str]:
    """A rendered table row's cells: columns are separated by two spaces or more."""
    import re

    return re.split(r"\s{2,}", line.strip())


def _section(text: str, heading: str) -> list[str]:
    """The lines from ``heading`` to the next blank line after its table."""
    lines = text.splitlines()
    start = lines.index(heading)
    block: list[str] = []
    for line in lines[start + 1 :]:
        if not line and block:
            break
        if line:
            block.append(line)
    return block


def test_the_system_section_shows_tokens_per_bucket() -> None:
    from rpg_economy.i18n import set_locale, t

    runs = []
    for seed in SEEDS:
        for condition in "ABC":
            run = fake_run(condition, seed, volatility=1.0 + seed * 0.01)
            run.tokens = {
                "by_bucket": {"agent": 900 + seed, "director": 100 if condition == "A" else 0},
                "authoritative_total": 1000 + seed,
                "budget": {},
            }
            runs.append(run)
    results = build_results(runs)
    assert results.system["A"]["tokens_by_bucket_median"]["agent"] > 0

    text = render_report(results, locale="en", figure_paths={})
    set_locale("en")
    agent = next(
        line
        for line in text.splitlines()
        if line.startswith(t("report.system.tokens_bucket", bucket=t("bucket.agent")))
    )
    director = next(
        line
        for line in text.splitlines()
        if line.startswith(t("report.system.tokens_bucket", bucket=t("bucket.director")))
    )
    assert _cells(agent)[1:] == ["906"] * 3  # median of 901..912, no decimals
    assert _cells(director)[1:] == ["100", "0", "0"]


def _secondary_recovery_results():
    runs = []
    for seed in SEEDS:
        for condition in "ABC":
            run = fake_run(condition, seed, volatility=1.0 + seed * 0.01)
            run.recovery_common_horizon = Duration(10 + seed, condition == "B" and seed % 2 == 0)
            run.propagation = {
                "source": "iron",
                "good": "melee_weapon",
                "mean_price": {"base": 30.0, "shock": 35.0, "recovery": 31.0},
                "stockout_ticks": {"base": 0, "shock": 2, "recovery": 0},
            }
            run.propagation_recovery = Duration(5 + seed, condition == "C" and seed % 3 == 0)
            runs.append(run)
    return build_results(runs)


@pytest.mark.parametrize("locale", ["en", "es"])
def test_secondary_recoveries_report_their_paired_tests(locale: str) -> None:
    """The common-horizon and propagation recoveries show both tests, like the primary."""
    from rpg_economy.i18n import set_locale, t

    results = _secondary_recovery_results()
    assert results.survival_common_horizon["paired_binary"]
    assert results.survival_propagation["stratified_logrank"]

    text = render_report(results, locale=locale, figure_paths={})
    set_locale(locale)
    lines = text.splitlines()
    binary = [i for i, line in enumerate(lines) if line == t("report.test.paired_binary")]
    logrank = [i for i, line in enumerate(lines) if line == t("report.test.logrank")]
    # Primary, common horizon and propagation: three of each.
    assert len(binary) == 3
    assert len(logrank) == 3

    common = lines.index(t("report.section.recovery_common"))
    propagation = lines.index(
        t(
            "report.section.propagation",
            good=t("product.melee_weapon"),
            resource=t("resource.iron"),
        )
    )
    assert common < binary[1] < logrank[1] < propagation < binary[2] < logrank[2]


def test_each_contrast_row_shows_its_pairs_used_and_dropped() -> None:
    """Design s11.3: pairs used and pairs dropped, reported per contrast."""
    from rpg_economy.i18n import set_locale, t

    runs = []
    for seed in range(1, 11):
        runs.append(fake_run("A", seed, volatility=2.0 + seed * 0.01))
        b = fake_run("B", seed, volatility=1.0 + seed * 0.02)
        if seed <= 4:
            b.volatility_phase["shock"] = None
        runs.append(b)
        runs.append(fake_run("C", seed, volatility=1.5 + seed * 0.03))
    results = build_results(runs)

    text = render_report(results, locale="en", figure_paths={})
    set_locale("en")
    heading = t("report.section.contrasts_for", metric=t("report.metric.volatility_shock_phase"))
    block = _section(text, heading)
    header = _cells(block[0])
    n_col = header.index(t("report.col.n"))
    dropped_col = header.index(t("report.col.dropped"))
    rows = {
        _cells(line)[0]: _cells(line)
        for line in block[2:]
        if " / " in line and not line.startswith("-")
    }
    ab = rows[f"{t('condition.A')} / {t('condition.B')}"]
    ac = rows[f"{t('condition.A')} / {t('condition.C')}"]
    bc = rows[f"{t('condition.B')} / {t('condition.C')}"]
    assert (ab[n_col], ab[dropped_col]) == ("6", "4")
    assert (ac[n_col], ac[dropped_col]) == ("10", "0")
    assert (bc[n_col], bc[dropped_col]) == ("6", "4")
    assert t("report.note.dropped_pairs_column") in text


@pytest.mark.invariant
@pytest.mark.parametrize("locale", ["en", "es"])
def test_a_suppressed_metric_prints_no_medians(locale: str) -> None:
    """The median table must not show the comparison the suppression withholds."""
    from rpg_economy.i18n import set_locale, t

    runs = []
    for seed in range(1, 11):
        runs.append(fake_run("A", seed, volatility=2.0 + seed * 0.01))
        b = fake_run("B", seed, volatility=1.0 + seed * 0.01)
        if seed <= 4:
            b.volatility["shock"] = None
        runs.append(b)
    results = build_results(runs)
    assert "volatility_shock" in results.suppressed_contrasts

    text = render_report(results, locale=locale, figure_paths={})
    set_locale(locale)
    label = t("report.metric.volatility_shock")
    row = next(line for line in text.splitlines() if _cells(line)[0] == label)
    cells = _cells(row)
    assert cells[1:3] == [t("report.value.suppressed")] * 2
    assert cells[3] == "6"  # the pairable seeds are still counted
    assert t("report.note.median_suppressed") in text
    # The companion is not suppressed, so it keeps its medians.
    companion = next(
        line
        for line in text.splitlines()
        if _cells(line)[0] == t("report.metric.volatility_shock_phase")
    )
    assert t("report.value.suppressed") not in companion


def test_no_suppression_note_without_a_suppressed_metric() -> None:
    from rpg_economy.i18n import set_locale, t

    text = render_report(sample_results(), locale="en", figure_paths={})
    set_locale("en")
    assert t("report.note.median_suppressed") not in text


# ---------------------------------------------------------------------------
# The economy check's header and numbers
# ---------------------------------------------------------------------------
def test_the_economy_check_names_the_regime_it_ran(tmp_path: Path) -> None:
    """Rule-based agents under the decentralized regime, not the configured condition."""
    from rpg_economy.analysis.render import render
    from rpg_economy.i18n import set_locale, t

    path, config, final, initial, target, check = _economy_run(tmp_path)
    assert config.run.condition.value == "A"
    set_locale("en")
    header = render(check(path, config, final, initial, target), config).splitlines()[1]
    assert header.startswith(t("check.regime.rule_based"))
    assert t("condition.A") not in header


def test_the_economy_check_formats_money_in_spanish(tmp_path: Path) -> None:
    from decimal import Decimal

    from rpg_economy.analysis.render import _money, render
    from rpg_economy.i18n import format_number, set_locale

    set_locale("es")
    assert _money(Decimal("1035.456")) == "1.035,46"
    assert _money(Decimal("-3.5")) == "-3,50"
    set_locale("en")
    assert _money(Decimal("1035.456")) == "1,035.46"

    path, config, final, initial, target, check = _economy_run(tmp_path)
    result = check(path, config, final, initial, target)
    set_locale("es")
    spanish = render(result, config)
    start = format_number(result.total_gold_start.quantize(Decimal("0.01")), 2)
    assert "," in start
    assert start in spanish
    set_locale("en")
    assert format_number(result.total_gold_start.quantize(Decimal("0.01")), 2) in render(
        result, config
    )


# ---------------------------------------------------------------------------
# Numbers are localized, counts included
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "english", "spanish"),
    [
        (0.125, "12.5%", "12,5%"),
        (0.0625, "6.3%", "6,3%"),  # half up, not float half-to-even
        (0.5, "50.0%", "50,0%"),
        (1.0, "100.0%", "100,0%"),
        (0.0, "0.0%", "0,0%"),
        (12.345, "1,234.5%", "1.234,5%"),
    ],
)
def test_percentages_are_localized_and_rounded_half_up(
    value: float, english: str, spanish: str
) -> None:
    from rpg_economy.analysis.report import _percent
    from rpg_economy.i18n import set_locale

    set_locale("en")
    assert _percent(value) == english
    set_locale("es")
    assert _percent(value) == spanish


@pytest.mark.parametrize(
    ("value", "places", "english", "spanish"),
    [
        (1234.5678, 3, "1,235", "1.235"),
        (-2500.0, 3, "-2,500", "-2.500"),
        (12.5, 3, "12.5", "12,5"),
        (0.25, 2, "0.25", "0,25"),
        (3.0, 3, "3", "3"),
        (100.0, 3, "100", "100"),
        (-0.0001, 3, "0", "0"),
        (7.0, 0, "7", "7"),
        # Rounds up across the thousands boundary: grouped, and no stray decimals.
        (999.9996, 3, "1,000", "1.000"),
        (None, 3, None, None),
        (float("nan"), 3, None, None),
    ],
)
def test_numbers_are_localized_through_the_shared_formatter(
    value: float | None, places: int, english: str | None, spanish: str | None
) -> None:
    from rpg_economy.analysis.report import _fmt
    from rpg_economy.i18n import set_locale, t

    for locale, expected in (("en", english), ("es", spanish)):
        set_locale(locale)
        assert _fmt(value, places) == (expected or t("report.value.missing"))


def test_p_values_are_localized_and_nan_is_missing() -> None:
    from rpg_economy.analysis.report import _p
    from rpg_economy.i18n import set_locale, t

    set_locale("en")
    assert _p(0.0421) == "0.042"
    assert _p(0.0001) == "<0.001"
    assert _p(float("nan")) == t("report.value.missing")
    assert _p(None) == t("report.value.missing")
    set_locale("es")
    assert _p(0.0421) == "0,042"
    assert _p(0.0001) == "<0,001"
    assert _p(float("nan")) == t("report.value.missing")


def test_counts_in_the_system_section_are_localized() -> None:
    runs = []
    for seed in SEEDS:
        for condition in "ABC":
            run = fake_run(condition, seed, volatility=1.0 + seed * 0.01)
            run.coordination["messages_by_type"] = {"REQUEST": 200}
            run.coordination["invalid_outputs"] = 150
            run.unanswered_requests = 100
            runs.append(run)
    results = build_results(runs)

    from rpg_economy.i18n import set_locale, t

    # Summed over 12 seeds: 2400 messages, 1800 invalid, 1200 unanswered.
    for locale, sep in (("en", ","), ("es", ".")):
        text = render_report(results, locale=locale, figure_paths={})
        set_locale(locale)
        rows = {_cells(line)[0]: _cells(line)[1:] for line in text.splitlines() if line}
        for key, kwargs, expected in (
            ("report.system.messages", {"kind": t("message.REQUEST")}, f"2{sep}400"),
            ("report.system.invalid", {}, f"1{sep}800"),
            ("report.system.unanswered", {}, f"1{sep}200"),
        ):
            assert rows[t(key, **kwargs)] == [expected] * 3, (locale, key)


# ---------------------------------------------------------------------------
# Per-actor token detail (design s11.2)
# ---------------------------------------------------------------------------
def test_the_system_section_shows_input_output_and_calls_per_bucket() -> None:
    from rpg_economy.i18n import set_locale, t

    runs = []
    for seed in SEEDS:
        for condition in "ABC":
            run = fake_run(condition, seed, volatility=1.0 + seed * 0.01)
            director = condition == "A"
            run.tokens = {
                "by_bucket": {"agent": 900, "director": 100 if director else 0, "bulletin": 0},
                "input_by_bucket": {"agent": 800, "director": 90 if director else 0, "bulletin": 0},
                "output_by_bucket": {
                    "agent": 100,
                    "director": 10 if director else 0,
                    "bulletin": 0,
                },
                "calls_by_bucket": {"agent": 20 + seed, "director": 5 if director else 0},
                "authoritative_total": 1000,
                "budget": {},
            }
            runs.append(run)
    results = build_results(runs)
    system = results.system
    assert system["A"]["input_tokens_by_bucket_median"] == {
        "agent": 800,
        "bulletin": 0,
        "director": 90,
    }
    assert system["B"]["calls_by_bucket_median"]["director"] == 0
    assert results.to_payload()["system"]["A"]["output_tokens_by_bucket_median"]["agent"] == 100

    text = render_report(results, locale="en", figure_paths={})
    set_locale("en")
    lines = text.splitlines()

    def cells(key: str, bucket: str) -> list[str]:
        label = t(key, bucket=t(f"bucket.{bucket}"))
        return _cells(next(line for line in lines if line.startswith(label)))[1:]

    assert cells("report.system.input_tokens_bucket", "agent") == ["800"] * 3
    assert cells("report.system.output_tokens_bucket", "director") == ["10", "0", "0"]
    assert cells("report.system.calls_bucket", "agent") == ["26"] * 3  # median of 21..32
    assert cells("report.system.calls_bucket", "director") == ["5", "0", "0"]
    # A bucket no condition used gets no row.
    bulletin = t("report.system.input_tokens_bucket", bucket=t("bucket.bulletin"))
    assert not any(line.startswith(bulletin) for line in lines)


# ---------------------------------------------------------------------------
# Seeds left out are named (design s11.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("locale", ["en", "es"])
def test_the_report_names_the_seeds_of_incomplete_trios(locale: str) -> None:
    from rpg_economy.i18n import set_locale, t

    results = sample_results()
    clean = render_report(results, locale=locale, figure_paths={})
    for key in ("incomplete_trios", "skipped_seeds"):
        assert t(f"report.note.{key}", locale=locale, seeds="", count="").strip() not in clean
    assert t("report.note.skipped_cells", locale=locale, cells="", count="").strip() not in clean
    results.incomplete_trios = [2014, 2003]
    results.skipped_seeds = [1001]
    results.skipped_cells = {"demand_surge/experiment_1": ["A", "B"]}
    text = render_report(results, locale=locale, figure_paths={})
    set_locale(locale)
    assert t("report.note.incomplete_trios", seeds="2003, 2014", count="2") in text
    assert t("report.note.skipped_seeds", seeds="1001", count="1") in text
    cell = f"{t('scenario.demand_surge')} / {t('experiment.experiment_1')} (A, B)"
    assert t("report.note.skipped_cells", cells=cell, count="1") in text

    payload = results.to_payload()
    assert payload["incomplete_trios"] == [2003, 2014]
    assert payload["skipped_seeds"] == [1001]
    assert payload["skipped_cells"] == {"demand_surge/experiment_1": ["A", "B"]}


# ---------------------------------------------------------------------------
# The common-horizon section carries every economic metric (s8.4, s11.2)
# ---------------------------------------------------------------------------
def _common_horizon_results():
    runs = []
    for seed in SEEDS:
        for condition in "ABC":
            run = fake_run(condition, seed, volatility=1.0 + seed * 0.01)
            offset = {"A": 0.0, "B": 0.1, "C": 0.2}[condition]
            run.multi_good = {"volatility": 1.2, "volume": 0.8, "stockouts": 1.5}
            run.stockouts_inventory = {"by_phase": {"base": 0, "shock": 2, "recovery": 0}}
            run.common_horizon = {
                "horizon": 30,
                "traded_value": 40.0 + seed,
                "stockout_ticks": 2,
                "volatility_shock": 1.0 + seed * 0.01,
                "volatility_shock_phase": 1.0 + seed * 0.01,
                "stockout_ticks_inventory": 1 + seed % 2,
                "useful_production": 0.5 + offset + seed * 0.001,
                "gini": 0.3 + offset + seed * 0.002,
                "volatility_all_goods": 1.1,
                "volume_all_goods": 0.7 + offset,
                "stockouts_all_goods": 1.4,
                "propagation_price_shock": None,
                "propagation_stockout_ticks": None,
            }
            run.recovery_common_horizon = Duration(10 + seed, False)
            runs.append(run)
    return build_results(runs)


def test_every_common_horizon_metric_is_contrasted() -> None:
    results = _common_horizon_results()
    for metric in (
        "useful_production_common",
        "gini_common",
        "stockout_ticks_inventory_common",
        "volatility_all_goods_common",
        "volume_all_goods_common",
        "stockouts_all_goods_common",
    ):
        assert metric in results.contrasts, metric
    # No propagation block in these runs: its common-horizon twin does not exist.
    assert "propagation_price_shock_common" not in results.contrasts
    by_pair = {(c["left"], c["right"]): c for c in results.contrasts["gini_common"]}
    assert by_pair[("A", "C")]["median_difference"] == pytest.approx(-0.2)


@pytest.mark.parametrize("locale", ["en", "es"])
def test_the_common_horizon_section_lists_every_metric(locale: str) -> None:
    from rpg_economy.i18n import set_locale, t

    results = _common_horizon_results()
    text = render_report(results, locale=locale, figure_paths={})
    set_locale(locale)
    section = text.index(t("report.section.common_horizon"))
    secondary = text.index(t("report.section.secondary"))
    recovery = text.index(t("report.section.recovery_common"))
    assert section < recovery
    for metric in ("gini_common", "useful_production_common", "volume_all_goods_common"):
        label = t(f"report.metric.{metric}")
        # The median row sits in the common-horizon section, not the secondary table.
        first = next(i for i, line in enumerate(text.splitlines()) if line.startswith(label))
        assert sum(len(line) + 1 for line in text.splitlines()[:first]) > section, metric
    assert secondary < section


# ---------------------------------------------------------------------------
# The volume-by-good table is in the reader's alphabet
# ---------------------------------------------------------------------------
def test_collation_ignores_case_and_accents() -> None:
    from rpg_economy.analysis.report import _collation_key

    assert sorted(["zeta", "Árbol", "arco", "Bastón", "barco"], key=_collation_key) == [
        "Árbol",
        "arco",
        "barco",
        "Bastón",
        "zeta",
    ]


@pytest.mark.parametrize("locale", ["en", "es"])
def test_the_volume_by_good_table_is_sorted_by_localized_name(locale: str) -> None:
    from rpg_economy.analysis.report import _collation_key, good_label
    from rpg_economy.i18n import set_locale, t

    goods = [
        "iron",
        "wood",
        "leather",
        "herbs",
        "arcane_crystal",
        "melee_weapon",
        "armor",
        "bow_gear",
        "staff_focus",
        "health_potion",
        "mana_potion",
    ]
    runs = [fake_run(c, s, volatility=1.0) for c in "ABC" for s in SEEDS]
    for run in runs:
        run.volume_by_good = {
            good: {phase: {"units": 5, "value": 50.0} for phase in ("base", "shock", "recovery")}
            for good in goods
        }
    text = render_report(build_results(runs), locale=locale, figure_paths={})
    set_locale(locale)
    section = text.split(t("report.section.volume_by_good"), 1)[1].split("\n\n", 2)[1]
    rows = section.splitlines()[2:]
    labels = [good_label(good) for good in goods]
    shown = [next(label for label in labels if row.startswith(label + " ")) for row in rows]
    assert shown == sorted(labels, key=_collation_key)


# ---------------------------------------------------------------------------
# The report states the metric parameters and what the checks left out
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("locale", ["en", "es"])
def test_the_report_states_the_metric_parameters_in_one_line(locale: str) -> None:
    from rpg_economy.i18n import set_locale, t

    results = sample_results()
    assert "metric_parameters" in results.to_payload()
    results.metric_parameters = {
        "volatility_window_w": 20,
        "recovery_band_pct_x": "0.15",
        "recovery_hold_ticks_m": 10,
        "useful_production_ticks_t": 30,
        "unanswered_request_ticks_d": 12,
    }
    text = render_report(results, locale=locale, figure_paths={})
    set_locale(locale)
    band = "0.15" if locale == "en" else "0,15"
    params = f"W = 20, X = {band}, M = 10, T = 30, D = 12"
    assert t("report.note.metric_parameters", params=params) in text
    assert results.to_payload()["metric_parameters"]["recovery_band_pct_x"] == "0.15"


@pytest.mark.parametrize("locale", ["en", "es"])
def test_the_report_names_what_the_checks_left_out(locale: str) -> None:
    from rpg_economy.i18n import set_locale, t

    results = sample_results()
    clean = render_report(results, locale=locale, figure_paths={})
    results.skipped_logs = [
        {"seed": 2002, "reason": "phase_adhoc"},
        {"seed": 2002, "reason": "phase_adhoc"},
        {"seed": 1001, "reason": "seed_not_analysed"},
    ]
    results.rejected_trios = {
        "resource_scarcity/experiment_1": [{"seed": 2003, "reason": "horizon_mismatch"}]
    }
    results.refused_cells = {"demand_surge/experiment_1": ["code_digest", "python_version"]}
    text = render_report(results, locale=locale, figure_paths={})
    set_locale(locale)
    reasons = (
        f"{t('report.skip_reason.phase_adhoc')}: 2; {t('report.skip_reason.seed_not_analysed')}: 1"
    )
    lines = (
        t("report.note.skipped_logs", reasons=reasons, count="3"),
        t(
            "report.note.rejected_trios",
            trios=f"{t('scenario.resource_scarcity')} / {t('experiment.experiment_1')} 2003 "
            f"({t('report.reject_reason.horizon_mismatch')})",
            count="1",
        ),
        t(
            "report.note.refused_cells",
            cells=f"{t('scenario.demand_surge')} / {t('experiment.experiment_1')} "
            "(code_digest, python_version)",
            count="1",
        ),
    )
    for line in lines:
        assert line in text
        assert line not in clean
    payload = results.to_payload()
    assert payload["refused_cells"] == {
        "demand_surge/experiment_1": ["code_digest", "python_version"]
    }
    assert payload["rejected_trios"]["resource_scarcity/experiment_1"][0]["seed"] == 2003


# ---------------------------------------------------------------------------
# The economy check's counts use the shared formatter
# ---------------------------------------------------------------------------
def test_economy_check_counts_are_localized() -> None:
    import importlib
    import inspect

    from rpg_economy.i18n import set_locale

    render_module = importlib.import_module("rpg_economy.analysis.render")

    set_locale("es")
    assert render_module._count(12345) == "12.345"
    set_locale("en")
    assert render_module._count(12345) == "12,345"
    assert render_module._count(None) == "-"
    assert "str(" not in inspect.getsource(render_module)
