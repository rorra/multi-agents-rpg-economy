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
