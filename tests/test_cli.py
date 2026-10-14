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
