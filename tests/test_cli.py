"""CLI behaviour, including bilingual output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rpg_economy.cli import main, peek_locale
from rpg_economy.i18n import clear_cache, t
from tests.conftest import BASE_CONFIG


@pytest.fixture(autouse=True)
def _reset_locale() -> None:
    clear_cache()
    yield
    clear_cache()


def test_config_show_reports_the_hash(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config-show"]) == 0
    out = capsys.readouterr().out
    assert "configs/base.yaml" in out
    assert len(json.dumps(out)) > 0


def test_config_show_renders_in_spanish(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--locale", "es", "config-show"]) == 0
    assert "Configuración cargada" in capsys.readouterr().out


def test_config_show_renders_in_english(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--locale", "en", "config-show"]) == 0
    assert "Configuration loaded" in capsys.readouterr().out


def test_invalid_config_exits_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("time: {k_activations: 0}", encoding="utf-8")
    assert main(["config-show", "--config", str(bad)]) == 2
    assert capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--locale", "es", "config-show"], "es"),
        (["--locale=es", "config-show"], "es"),
        (["config-show"], None),
        (["--locale"], None),
    ],
)
def test_peek_locale(argv: list[str], expected: str | None) -> None:
    assert peek_locale(argv) == expected


def test_help_text_is_translated(capsys: pytest.CaptureFixture[str]) -> None:
    """argparse help is user-facing CLI text and must honour --locale."""
    with pytest.raises(SystemExit):
        main(["--locale", "es", "--help"])
    assert "Idioma de salida" in capsys.readouterr().out


SHORT_LAYER = """
phases: {base_ticks: 6, shock_ticks: 6, recovery_ticks: 8}
time: {horizon_ticks: 20}
metrics: {volatility_window_w: 3, recovery_hold_ticks_m: 2}
"""


def test_trio_runs_offline_and_renders_its_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layer = tmp_path / "short.yaml"
    layer.write_text(SHORT_LAYER, encoding="utf-8")
    out = tmp_path / "trio"

    code = main(["--locale", "es", "trio", "--layer", str(layer), "--seed", "3", "--out", str(out)])

    assert code == 0
    assert {p.name for p in out.glob("*.jsonl")} == {"A.jsonl", "B.jsonl", "C.jsonl"}
    assert (out / "replay" / "es" / "replay.html").exists()
    assert (out / "replay" / "en" / "replay.html").exists()
    printed = capsys.readouterr().out
    assert t("cli.trio.mock", locale="es") in printed
    # Status lines stay in the caller's language even though an English page
    # was rendered along the way.
    assert "Reproducción [en] escrita" in printed


def test_a_real_provider_trio_is_refused_without_a_price(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layer = tmp_path / "real.yaml"
    layer.write_text(SHORT_LAYER + "llm: {mock: false}\n", encoding="utf-8")

    code = main(["trio", "--layer", str(layer), "--out", str(tmp_path / "trio")])

    assert code == 5
    assert not (tmp_path / "trio").exists()
    assert "cost estimate" in capsys.readouterr().err


def test_replay_takes_layers_so_a_layered_run_keeps_its_guide(tmp_path: Path) -> None:
    """Runs made with a layer only verify against the same layered config."""
    import yaml

    from rpg_economy.runner import run_trio
    from tests.conftest import make_config

    layer = tmp_path / "short.yaml"
    layer.write_text(
        yaml.safe_dump(
            {
                "phases": {"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 4},
                "time": {"horizon_ticks": 12},
                "metrics": {"volatility_window_w": 2, "recovery_hold_ticks_m": 2},
            }
        )
    )
    config = make_config(**yaml.safe_load(layer.read_text()))
    cell = config.for_cell(scenario=config.run.scenario, experiment=config.run.experiment, seed=5)
    run_trio(cell, tmp_path / "runs", seed=5, cycles=cell.time.total_cycles)

    out = tmp_path / "page.html"
    assert main(["replay", str(tmp_path / "runs"), "--layer", str(layer), "--out", str(out)]) == 0
    assert 'id="guide"' in out.read_text()


def test_a_missing_median_stockout_change_prints_a_dash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    from rpg_economy.cli import _render_efficacy

    effect = SimpleNamespace(
        seed=1, price_change=0.1, stockout_change=None, supply_change=None, value_change=-0.2
    )

    stub = SimpleNamespace(
        effects=[effect],
        median=lambda name: getattr(effect, name),
        effective=lambda: False,
    )

    assert _render_efficacy(stub) is False
    out = capsys.readouterr().out
    assert "+0" not in out
    assert "Median: price +10%, stockout ticks -, gathered -, traded value -20%" in out
    assert "seed 1: price +10%, stockout ticks -, gathered -" in out


# ---------------------------------------------------------------------------
# Real-provider gates, exercised offline: every one refuses before any call
# ---------------------------------------------------------------------------
def _short_config_file(tmp_path: Path, **overrides: object) -> Path:
    """A compressed configuration on disk, with the base seeds and pricing."""
    import yaml

    from tests.conftest import make_config

    base = {
        "phases": {"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 4},
        "time": {"horizon_ticks": 12},
        "metrics": {"volatility_window_w": 2, "recovery_hold_ticks_m": 2},
    }
    base.update(overrides)
    config = make_config(**base)
    path = tmp_path / "short.yaml"
    path.write_text(yaml.safe_dump(json.loads(config.model_dump_json())), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def mock_profile(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Mock logs of the primary scenario's pilot seeds, to price a batch from."""
    import dataclasses

    from rpg_economy.campaign import build_plan
    from rpg_economy.campaign.runner import run_campaign
    from rpg_economy.domain import Scenario
    from tests.conftest import make_config

    config = make_config(
        phases={"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 4},
        time={"horizon_ticks": 12},
        metrics={"volatility_window_w": 2, "recovery_hold_ticks_m": 2},
    )
    plan = build_plan(config, phase="pilot", scenarios=(Scenario.RESOURCE_SCARCITY,))
    root = tmp_path_factory.mktemp("profile")
    run_campaign(config, dataclasses.replace(plan, trios=plan.trios[:2]), root)
    return root


def test_campaign_run_requires_a_phase(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A forgotten --phase must not quietly pick a seed set."""
    with pytest.raises(SystemExit) as exit_info:
        main(["campaign-run", "--out", str(tmp_path / "runs")])
    assert exit_info.value.code == 2
    assert "--phase" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_the_pilot_refuses_the_mock(
    tmp_path: Path, mock_profile: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "pilot",
            "--stage",
            "calibration",
            "--out",
            str(tmp_path / "pilot"),
            "--profile-logs",
            str(mock_profile),
            "--spend-limit-set",
            "--confirm-cost",
            "0.01",
        ]
    )
    assert code == 5
    assert "--real" in capsys.readouterr().err
    assert not (tmp_path / "pilot").exists()


def test_the_pilot_refuses_without_a_spend_limit(
    tmp_path: Path, mock_profile: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "pilot",
            "--real",
            "--stage",
            "calibration",
            "--out",
            str(tmp_path / "pilot"),
            "--profile-logs",
            str(mock_profile),
            "--confirm-cost",
            "0.01",
        ]
    )
    assert code == 5
    assert "spend limit" in capsys.readouterr().err
    assert not (tmp_path / "pilot").exists()


def test_the_pilot_refuses_a_cost_it_was_not_shown(
    tmp_path: Path, mock_profile: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "pilot",
            "--real",
            "--stage",
            "calibration",
            "--out",
            str(tmp_path / "pilot"),
            "--profile-logs",
            str(mock_profile),
            "--spend-limit-set",
            "--confirm-cost",
            "999999.99",
        ]
    )
    assert code == 5
    assert "without confirmation" in capsys.readouterr().err
    assert not (tmp_path / "pilot").exists()


def test_pilot_verify_only_fails_a_stage_that_never_ran(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["pilot", "--stage", "calibration", "--out", str(tmp_path), "--verify-only"])
    assert code == 7
    printed = capsys.readouterr()
    report = json.loads(printed.out[printed.out.index("{") :])
    assert report["runs"] == 0
    assert report["expected_runs"] > 0
    assert report["missing"] == report["expected_runs"]
    assert report["complete"] is False


def _smoke_total() -> str:
    from rpg_economy.cli import _smoke_estimate
    from rpg_economy.config import load_config

    return str(_smoke_estimate(load_config(BASE_CONFIG)))


def test_smoke_refuses_without_a_spend_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["smoke", "--out", str(tmp_path / "smoke"), "--confirm-cost", _smoke_total()])
    assert code == 5
    assert "spend limit" in capsys.readouterr().err
    assert not (tmp_path / "smoke").exists()


def test_smoke_refuses_a_cost_it_was_not_shown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["smoke", "--out", str(tmp_path / "smoke"), "--spend-limit-set", "--confirm-cost", "0.00"]
    )
    assert code == 5
    assert "without confirmation" in capsys.readouterr().err
    assert not (tmp_path / "smoke").exists()


def test_smoke_reports_a_provider_failure_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rpg_economy import runner
    from rpg_economy.llm import ProviderError

    def unreachable(*args: object, **kwargs: object) -> None:
        raise ProviderError("Error code: 401 - invalid_api_key")

    # Both entry points replaced: nothing here can reach a provider.
    monkeypatch.setattr(runner, "run_trio", unreachable)
    monkeypatch.setattr(runner, "run_condition", unreachable)

    code = main(
        [
            "smoke",
            "--out",
            str(tmp_path / "smoke"),
            "--spend-limit-set",
            "--confirm-cost",
            _smoke_total(),
        ]
    )
    assert code == 8
    assert "invalid_api_key" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["smoke", "--seed", "2001", "--spend-limit-set", "--confirm-cost", "0.01"],
        ["derisk", "--seed", "2001", "--spend-limit-set", "--confirm-cost", "0.01"],
        ["trio", "--real", "--seed", "2001", "--spend-limit-set", "--confirm-cost", "0.01"],
    ],
)
def test_one_off_real_runs_refuse_a_final_seed(
    tmp_path: Path, argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Running a final seed before the campaign is a look at final data."""
    code = main([*argv, "--out", str(tmp_path / "out")])
    assert code == 5
    assert "--allow-final-seed" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_allow_final_seed_only_lifts_the_seed_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["trio", "--real", "--seed", "2001", "--allow-final-seed", "--out", str(tmp_path / "out")]
    )
    assert code == 5
    err = capsys.readouterr().err
    assert "--allow-final-seed" not in err
    assert "cost estimate" in err


def test_a_mock_trio_on_a_final_seed_is_not_gated(tmp_path: Path) -> None:
    """The gate is about real spend on final data; the offline mock runs freely."""
    layer = tmp_path / "short.yaml"
    layer.write_text(SHORT_LAYER, encoding="utf-8")
    code = main(
        [
            "trio",
            "--layer",
            str(layer),
            "--seed",
            "2001",
            "--no-replay",
            "--out",
            str(tmp_path / "t"),
        ]
    )
    assert code == 0


def test_derisk_without_a_comparison_price_is_refused_cleanly(
    tmp_path: Path, mock_profile: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "derisk",
            "--out",
            str(tmp_path / "derisk"),
            "--profile-logs",
            str(mock_profile),
            "--spend-limit-set",
        ]
    )
    assert code == 4
    assert "--model-pricing" in capsys.readouterr().err
    assert not (tmp_path / "derisk").exists()


def test_derisk_with_a_malformed_price_is_refused_cleanly(
    tmp_path: Path, mock_profile: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "derisk",
            "--out",
            str(tmp_path / "derisk"),
            "--profile-logs",
            str(mock_profile),
            "--model-pricing",
            "gpt-5-mini=cheap",
            "--spend-limit-set",
        ]
    )
    assert code == 4
    assert "invalid --model-pricing" in capsys.readouterr().err


def test_a_real_trio_is_priced_as_a_one_trio_plan(mock_profile: Path) -> None:
    """Same basis as a campaign, the Experiment 2 cap included."""
    from rpg_economy.cli import _trio_estimate, with_real_provider
    from rpg_economy.domain import ExperimentRegime, Scenario
    from tests.conftest import make_config

    cap = 1_000
    config = with_real_provider(
        make_config(budget={"experiment_2_total_tokens": cap}).for_cell(
            scenario=Scenario.RESOURCE_SCARCITY,
            experiment=ExperimentRegime.EQUAL_COST,
            seed=1001,
        )
    )
    estimate = _trio_estimate(config, mock_profile)

    assert estimate.total_runs == 3
    assert estimate.total_tokens <= 3 * cap
    assert estimate.worst_case is not None


# ---------------------------------------------------------------------------
# Shock efficacy covers every scenario the campaign runs
# ---------------------------------------------------------------------------
def _stub_efficacy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from types import SimpleNamespace

    from rpg_economy.analysis import shock_efficacy

    measured: list[str] = []

    def measure(config, seeds, directory=None):
        measured.append(config.run.scenario.value)
        return SimpleNamespace(effects=[], median=lambda name: 0.5, effective=lambda: True)

    monkeypatch.setattr(shock_efficacy, "measure", measure)
    return measured


def test_shock_efficacy_checks_every_configured_scenario(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rpg_economy.config import load_config

    measured = _stub_efficacy(monkeypatch)
    config = load_config(BASE_CONFIG)
    expected = ["resource_scarcity", *(s.value for s in config.campaign.secondary_scenarios)]

    assert main(["economy-check", "--shock-efficacy"]) == 0
    assert measured == expected
    assert len(expected) > 1, "the base configuration names a secondary scenario"
    assert capsys.readouterr().out.count("Shock efficacy:") == len(expected)


def test_shock_efficacy_can_be_narrowed_to_one_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measured = _stub_efficacy(monkeypatch)
    assert main(["economy-check", "--shock-efficacy", "--scenario", "demand_surge"]) == 0
    assert measured == ["demand_surge"]


# ---------------------------------------------------------------------------
# The budget round trip: derive, then plan and report with the derived layer
# ---------------------------------------------------------------------------
def test_the_budget_layer_round_trips_through_plan_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _short_config_file(
        tmp_path,
        seeds={"pilot": list(range(1, 9)), "final": list(range(101, 111))},
        campaign={"secondary_scenarios": ["demand_surge"], "secondary_final_seeds": 8},
    )
    pilot = tmp_path / "pilot"
    assert (
        main(["campaign-run", "--config", str(config), "--phase", "pilot", "--out", str(pilot)])
        == 0
    )
    assert not (pilot / ".campaign.lock").exists(), "the run left its lock behind"

    layer = tmp_path / "budget.yaml"
    assert (
        main(
            [
                "derive-budget",
                "--config",
                str(config),
                "--pilot-logs",
                str(pilot / "resource_scarcity"),
                "--allow-mock",
                "--layer-out",
                str(layer),
            ]
        )
        == 0
    )
    text = layer.read_text(encoding="utf-8")
    assert "experiment_2_total_tokens:" in text
    assert "--allow-mock" in text
    assert "derived_from_mock: true" in text

    # Without the layer the final plan is refused; with it, it plans.
    assert main(["campaign-plan", "--config", str(config), "--phase", "final"]) == 4
    capsys.readouterr()
    assert (
        main(["campaign-plan", "--config", str(config), "--layer", str(layer), "--phase", "final"])
        == 0
    )
    capsys.readouterr()

    final = tmp_path / "final"
    common = ["--config", str(config), "--layer", str(layer)]
    assert main(["campaign-run", *common, "--phase", "final", "--out", str(final)]) == 0
    assert (
        main(["report", str(final), *common, "--allow-mock", "--out", str(tmp_path / "reports")])
        == 0
    )
    assert any((tmp_path / "reports").rglob("*"))


# ---------------------------------------------------------------------------
# pilot --verify-only on a stage that ran whole
# ---------------------------------------------------------------------------
def _billed_log(path: Path, *, tokens: int = 100) -> None:
    """A finished, real, fully billed run log, as the billing verification reads it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "run_metadata", "payload": {}},
        {"type": "llm_call", "payload": {"model": "gpt-5-nano"}},
        {
            "type": "tokens_consumed",
            "payload": {"authoritative_total": tokens, "billed_total": tokens},
        },
        {"type": "run_end", "payload": {}},
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_pilot_verify_only_passes_a_complete_stage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from rpg_economy.campaign.pilot import CALIBRATION, stage_plan
    from rpg_economy.campaign.runner import trio_directory
    from rpg_economy.config import load_config

    plan = stage_plan(load_config(BASE_CONFIG), CALIBRATION)
    for trio in plan.trios:
        for condition in trio.conditions:
            _billed_log(trio_directory(tmp_path / "calibration", trio) / f"{condition.value}.jsonl")

    code = main(["pilot", "--stage", "calibration", "--out", str(tmp_path), "--verify-only"])

    assert code == 0
    printed = capsys.readouterr().out
    report = json.loads(printed[printed.index("{") :])
    assert report["complete"] is True
    assert report["runs"] == report["expected_runs"] == plan.total_runs
    assert report["all_real"] is True
    assert report["all_fully_billed"] is True


def test_the_billing_line_is_localized(capsys: pytest.CaptureFixture[str]) -> None:
    from rpg_economy.cli import _print_billing
    from rpg_economy.i18n import set_locale

    set_locale("es")
    _print_billing(
        {
            "runs": 3,
            "all_real": True,
            "all_fully_billed": False,
            "billed_total": 1234567,
            "set_aside": {"logs": 1, "billed_total": 2500},
        }
    )
    out = capsys.readouterr().out
    assert "1.234.567" in out
    assert "2.500" in out
    assert "sí" in out
    assert "True" not in out and "False" not in out


def test_pilot_header_prints_the_stage_purpose_translated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "--locale",
            "es",
            "pilot",
            "--stage",
            "calibration",
            "--out",
            str(tmp_path),
            "--verify-only",
        ]
    )
    out = capsys.readouterr().out
    assert t("pilot.purpose.calibration", locale="es") in out
    assert "verify the configuration" not in out


# ---------------------------------------------------------------------------
# smoke and derisk reruns: set aside or kept, never a traceback
# ---------------------------------------------------------------------------
def _write_once(path: Path) -> None:
    """Stands in for a run: refuses a non-empty log, as the real writer does."""
    from rpg_economy.logging_ import LogError

    path = Path(path)
    if path.exists() and path.stat().st_size:
        raise LogError(f"refusing to append to non-empty run log {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"stand_in": true}\n', encoding="utf-8")


def _fake_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both smoke runs and their verification replaced: nothing here reaches a provider."""
    from types import SimpleNamespace

    from rpg_economy import runner
    from rpg_economy.campaign import smoke

    def run_trio(config, directory, *, seed=None, **kwargs):
        for condition in "ABC":
            _write_once(Path(directory) / f"{condition}.jsonl")

    def run_condition(config, log_path, **kwargs):
        _write_once(log_path)

    def verify_smoke(path, *, label, planned_horizon, expect_exhaustion):
        return SimpleNamespace(
            label=label,
            model="stand-in",
            llm_calls=1200,
            provider_input=1_500_000,
            provider_output=20_000,
            ledger_total=1_520_000,
            provider_total=1_520_000,
            checks=[SimpleNamespace(passed=True, name="usage", detail="-")],
            passed=True,
            to_payload=lambda: {"label": label},
        )

    monkeypatch.setattr(runner, "run_trio", run_trio)
    monkeypatch.setattr(runner, "run_condition", run_condition)
    monkeypatch.setattr(smoke, "verify_smoke", verify_smoke)


def _smoke_argv(out: Path) -> list[str]:
    return ["smoke", "--out", str(out), "--spend-limit-set", "--confirm-cost", _smoke_total()]


def test_a_smoke_rerun_sets_the_earlier_runs_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_smoke(monkeypatch)
    out = tmp_path / "smoke"

    assert main(_smoke_argv(out)) == 0
    assert main(["--locale", "es", *_smoke_argv(out)]) == 0

    assert {p.name for p in (out / "smoke1" / "previous-1").iterdir()} == {
        "A.jsonl",
        "B.jsonl",
        "C.jsonl",
    }
    assert [p.name for p in (out / "smoke2" / "previous-1").iterdir()] == ["A.jsonl"]
    printed = capsys.readouterr()
    assert "[PASS] usage" in printed.out
    assert "1.500.000" in printed.out
    assert "Traceback" not in printed.err


def test_a_smoke_log_that_cannot_be_written_is_reported_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rpg_economy import runner
    from rpg_economy.logging_ import LogError

    _fake_smoke(monkeypatch)

    def refuses(*args: object, **kwargs: object) -> None:
        raise LogError("refusing to append to non-empty run log")

    monkeypatch.setattr(runner, "run_trio", refuses)

    assert main(_smoke_argv(tmp_path / "smoke")) == 6
    err = capsys.readouterr().err
    assert str(tmp_path / "smoke" / "smoke1") in err
    assert "Traceback" not in err


def test_smoke_report_only_without_runs_is_refused_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["smoke", "--out", str(tmp_path / "smoke"), "--report-only"]) == 6
    err = capsys.readouterr().err
    assert "SMOKE-1" in err
    assert "Traceback" not in err


def test_smoke_report_only_with_a_malformed_log_is_refused_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for step in ("smoke1", "smoke2"):
        (tmp_path / step).mkdir()
        (tmp_path / step / "A.jsonl").write_text("not an event\n", encoding="utf-8")

    assert main(["smoke", "--out", str(tmp_path), "--report-only"]) == 6
    err = capsys.readouterr().err
    assert str(tmp_path / "smoke1" / "A.jsonl") in err
    assert "Traceback" not in err


def _mock_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every de-risk arm replaced by a short mock configuration: offline, and free."""
    from rpg_economy.campaign import derisk
    from tests.conftest import make_config

    def mock_arm(base, *, model, horizon):
        return make_config(
            phases={"base_ticks": 4, "shock_ticks": 4, "recovery_ticks": 4},
            time={"horizon_ticks": 12},
            metrics={"volatility_window_w": 2, "recovery_hold_ticks_m": 2},
            llm={"model": model},
        )

    monkeypatch.setattr(derisk, "derisk_config", mock_arm)


DERISK_PRICING = "gpt-5-mini=0.25/2.00"


def _derisk_total(
    mock_profile: Path,
    *,
    seed: int = 1001,
    models: tuple[str, ...] = ("gpt-5-nano", "gpt-5-mini"),
) -> str:
    from rpg_economy.cli import _derisk_estimate, _parse_model_pricing
    from rpg_economy.config import load_config

    config = load_config(BASE_CONFIG, overrides={"run": {"seed": seed}})
    estimate = _derisk_estimate(
        config,
        models,
        mock_profile,
        horizon=120,
        seed=seed,
        pricing_overrides=_parse_model_pricing(DERISK_PRICING),
    )
    return str(estimate.total_cost)


def test_a_derisk_rerun_keeps_finished_arms_and_sets_aside_the_rest(
    tmp_path: Path,
    mock_profile: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _mock_arms(monkeypatch)
    out = tmp_path / "derisk"
    argv = [
        "derisk",
        "--out",
        str(out),
        "--seed",
        "1001",
        "--profile-logs",
        str(mock_profile),
        "--model-pricing",
        DERISK_PRICING,
        "--spend-limit-set",
        "--confirm-cost",
        _derisk_total(mock_profile),
    ]

    assert main(argv) == 0
    capsys.readouterr()

    # Everything finished: a rerun runs nothing, so it prices and confirms
    # nothing either, and reports again.
    assert main(argv[: argv.index("--spend-limit-set")]) == 0
    printed = capsys.readouterr()
    assert printed.out.count("gpt-5-") >= 2
    assert not list(out.rglob("previous-*"))

    # One arm cut short: only that arm runs again, its old logs set aside, and
    # only that arm is priced -- the whole comparison's total is refused.
    log = out / "gpt-5-mini" / "A.jsonl"
    log.write_bytes(b"".join(log.read_bytes().splitlines(keepends=True)[:-1]))
    assert main(argv) == 5
    capsys.readouterr()
    rerun = [*argv[:-1], _derisk_total(mock_profile, models=("gpt-5-mini",))]
    assert main(rerun) == 0
    assert {p.name for p in (out / "gpt-5-mini" / "previous-1").iterdir()} == {
        "A.jsonl",
        "B.jsonl",
        "C.jsonl",
    }
    assert not (out / "gpt-5-nano" / "previous-1").exists()
    assert "Traceback" not in capsys.readouterr().err


def test_derisk_report_only_with_a_malformed_log_is_refused_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arm = tmp_path / "gpt-5-nano"
    arm.mkdir()
    (arm / "A.jsonl").write_text("not an event\n", encoding="utf-8")

    assert main(["derisk", "--out", str(tmp_path), "--report-only"]) == 6
    assert "Traceback" not in capsys.readouterr().err


def test_the_derisk_estimate_prices_each_arm_as_a_one_trio_plan(mock_profile: Path) -> None:
    """Per cell, with the worst case and the retry multiplier, not pooled across cells."""
    from rpg_economy.campaign.derisk import derisk_config
    from rpg_economy.campaign.parallel import MAX_PROVIDER_ATTEMPTS
    from rpg_economy.cli import _derisk_estimate, _trio_estimate
    from rpg_economy.config import load_config

    config = load_config(BASE_CONFIG, overrides={"run": {"seed": 1001}})
    estimate = _derisk_estimate(
        config,
        (config.llm.model,),
        mock_profile,
        horizon=120,
        seed=1001,
    )

    arm = derisk_config(config, model=config.llm.model, horizon=120)
    arm = arm.for_cell(scenario=arm.run.scenario, experiment=arm.run.experiment, seed=1001)
    one_trio = _trio_estimate(arm, mock_profile)

    assert estimate.total_runs == 3
    assert estimate.total_cost == one_trio.total_cost
    assert estimate.worst_case == one_trio.worst_case
    assert estimate.worst_case is not None
    assert estimate.retry_attempts == MAX_PROVIDER_ATTEMPTS
    assert estimate.worst_case_with_retries == estimate.worst_case * MAX_PROVIDER_ATTEMPTS


# ---------------------------------------------------------------------------
# A budget derived from mock runs never reaches a real batch
# ---------------------------------------------------------------------------
def test_derive_budget_marks_a_mock_budget_in_its_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from rpg_economy.analysis import budget_derivation
    from rpg_economy.config import load_config

    derivation = SimpleNamespace(budget=123_456, runs=[1, 2, 3], to_payload=lambda: {})
    monkeypatch.setattr(budget_derivation, "find_logs", lambda path: [])
    monkeypatch.setattr(
        budget_derivation, "derive_budget", lambda logs, *, config, allow_mock: derivation
    )

    for allow_mock in (True, False):
        layer = tmp_path / f"budget-{allow_mock}.yaml"
        argv = ["derive-budget", "--pilot-logs", str(tmp_path), "--layer-out", str(layer)]
        assert main([*argv, "--allow-mock"] if allow_mock else argv) == 0
        loaded = load_config(BASE_CONFIG, layers=[layer])
        assert loaded.budget.experiment_2_total_tokens == 123_456
        assert loaded.budget.derived_from_mock is allow_mock
        # The code it was derived under travels with it.
        from rpg_economy.logging_.metadata import code_digest

        assert loaded.budget.derived_code_digest == code_digest()


MOCK_BUDGET_LAYER = "budget: {experiment_2_total_tokens: 50000, derived_from_mock: true}\n"


def test_a_real_campaign_refuses_a_budget_derived_under_other_code(
    tmp_path: Path,
    mock_profile: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rpg_economy.campaign import parallel

    monkeypatch.setattr(
        parallel, "run_campaign_parallel", lambda *a, **k: pytest.fail("a stale budget ran")
    )
    layer = tmp_path / "budget.yaml"
    layer.write_text(
        "budget: {experiment_2_total_tokens: 50000, derived_code_digest: not-this-code}\n",
        encoding="utf-8",
    )
    code = main(
        [
            "campaign-run",
            "--real",
            "--config",
            str(_short_config_file(tmp_path)),
            "--layer",
            str(layer),
            "--phase",
            "pilot",
            "--out",
            str(tmp_path / "runs"),
            "--profile-logs",
            str(mock_profile),
            "--spend-limit-set",
            "--confirm-cost",
            "0.01",
        ]
    )
    assert code == 5
    assert t("error.budget.code_changed") in capsys.readouterr().err


@pytest.mark.parametrize("locale", ["en", "es"])
def test_a_real_campaign_refuses_a_mock_derived_budget(
    tmp_path: Path,
    mock_profile: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    locale: str,
) -> None:
    from rpg_economy.campaign import parallel

    monkeypatch.setattr(
        parallel, "run_campaign_parallel", lambda *a, **k: pytest.fail("a mock budget ran")
    )
    layer = tmp_path / "budget.yaml"
    layer.write_text(MOCK_BUDGET_LAYER, encoding="utf-8")

    code = main(
        [
            "--locale",
            locale,
            "campaign-run",
            "--real",
            "--config",
            str(_short_config_file(tmp_path)),
            "--layer",
            str(layer),
            "--phase",
            "pilot",
            "--out",
            str(tmp_path / "runs"),
            "--profile-logs",
            str(mock_profile),
            "--spend-limit-set",
            "--confirm-cost",
            "0.01",
        ]
    )

    assert code == 5
    assert t("error.budget.mock_derived", locale=locale) in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def test_a_real_pilot_refuses_a_mock_derived_budget(
    tmp_path: Path, mock_profile: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layer = tmp_path / "budget.yaml"
    layer.write_text(MOCK_BUDGET_LAYER, encoding="utf-8")

    code = main(
        [
            "pilot",
            "--real",
            "--layer",
            str(layer),
            "--stage",
            "calibration",
            "--out",
            str(tmp_path / "pilot"),
            "--profile-logs",
            str(mock_profile),
            "--spend-limit-set",
            "--confirm-cost",
            "0.01",
        ]
    )

    assert code == 5
    assert t("error.budget.mock_derived") in capsys.readouterr().err
    assert not (tmp_path / "pilot").exists()


def test_the_mock_budget_gate_reads_the_budget_field(tmp_path: Path) -> None:
    from rpg_economy.cli import _refuse_mock_budget
    from rpg_economy.config import load_config

    layer = tmp_path / "budget.yaml"
    layer.write_text(MOCK_BUDGET_LAYER, encoding="utf-8")
    assert _refuse_mock_budget(load_config(BASE_CONFIG, layers=[layer])) is True
    assert _refuse_mock_budget(load_config(BASE_CONFIG)) is False


# ---------------------------------------------------------------------------
# The pilot's progress and failures, in the reader's language
# ---------------------------------------------------------------------------
def _confirmed_total(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """The total a real command asks to be confirmed, read from its refusal."""
    import re

    assert main([*argv, "--confirm-cost", "0"]) == 5
    return re.search(r"--confirm-cost (\S+)", capsys.readouterr().err).group(1)


def _pilot_argv(out: Path, mock_profile: Path) -> list[str]:
    return [
        "pilot",
        "--real",
        "--stage",
        "calibration",
        "--out",
        str(out),
        "--profile-logs",
        str(mock_profile),
        "--spend-limit-set",
    ]


def test_pilot_progress_and_failures_are_localized(
    tmp_path: Path,
    mock_profile: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rpg_economy.campaign import parallel
    from rpg_economy.campaign.parallel import ParallelResult, TrioOutcome

    def fake_run(config, plan, directory, *, workers, resume, progress, **kwargs):
        result = ParallelResult(directory=Path(directory), workers=workers)
        for index, trio in enumerate(plan.trios[:2]):
            outcome = TrioOutcome(
                trio=trio,
                status="failed" if index else "completed",
                detail="ProviderError: 401 invalid_api_key" if index else "",
            )
            result.outcomes.append(outcome)
            progress(outcome, index + 1, 2)
        return result

    monkeypatch.setattr(parallel, "run_campaign_parallel", fake_run)
    argv = ["--locale", "es", *_pilot_argv(tmp_path / "pilot", mock_profile)]
    total = _confirmed_total(argv, capsys)

    assert main([*argv, "--confirm-cost", total]) == 7

    printed = capsys.readouterr()
    progress = [line for line in printed.out.splitlines() if line.startswith(("[1/2]", "[2/2]"))]
    assert len(progress) == 2
    assert all(t("scenario.resource_scarcity", locale="es") in line for line in progress)
    assert t("campaign.status.completed", locale="es") in progress[0]
    assert t("campaign.status.failed", locale="es") in progress[1]
    assert not any("resource_scarcity" in line or "failed" in line for line in progress)
    assert "401 invalid_api_key" in printed.err
    assert "Traceback" not in printed.err


def test_pilot_refuses_a_stage_directory_another_run_holds(
    tmp_path: Path,
    mock_profile: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import os

    from rpg_economy.campaign import parallel
    from rpg_economy.campaign.lock import campaign_lock

    monkeypatch.setattr(
        parallel, "run_campaign_parallel", lambda *a, **k: pytest.fail("a locked stage ran")
    )
    stage = tmp_path / "pilot" / "calibration"
    argv = _pilot_argv(tmp_path / "pilot", mock_profile)
    total = _confirmed_total(argv, capsys)

    # Held for real: another open file description holds the flock.
    with campaign_lock(stage) as held:
        assert main([*argv, "--confirm-cost", total]) == 10
        assert held.path.exists()
    err = capsys.readouterr().err
    assert f"pid {os.getpid()}" in err
    assert "Traceback" not in err
